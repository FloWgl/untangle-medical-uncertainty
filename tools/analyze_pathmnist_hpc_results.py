#!/usr/bin/env python3
"""Summarize PathMNIST HPC decomposition tensors for thesis tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("docs/pathmnist_audit/pathmnist_audit.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/results/pathmnist"),
    )
    return parser.parse_args()


def output_dir_from_log(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="replace")
    matches = re.findall(r"Output directory is ([^.\n]+)", text)
    if not matches:
        return None
    return Path(matches[-1].strip())


DECOMPOSITIONS = {
    "information-theoretic": "it_au_eu.pt",
    "bregman-estimated": "bregman_eu_au_hat.pt",
}


def tensor_means(
    path: Path,
    decomposition: str = "information-theoretic",
) -> tuple[float, float, int] | None:
    if not path.exists():
        return None
    first, second = torch.load(path, map_location="cpu")
    if decomposition == "bregman-estimated":
        epistemic = first.detach().float()
        aleatoric = second.detach().float()
    else:
        aleatoric = first.detach().float()
        epistemic = second.detach().float()
    mask = torch.isfinite(aleatoric) & torch.isfinite(epistemic)
    count = int(mask.sum())
    if count == 0:
        return math.nan, math.nan, 0
    return float(aleatoric[mask].mean()), float(epistemic[mask].mean()), count


def relative_to_repo(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        return path
    try:
        return path.relative_to(repo_root.resolve())
    except ValueError:
        return path


def read_audit_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if not rows and path.exists():
        print(f"Refusing to overwrite {path} with an empty CSV")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_scarcity(repo_root: Path, audit_csv: Path) -> list[dict[str, object]]:
    tensor_dirs = []
    for tensor in (repo_root / "checkpoints").glob("*/id_test_hard_pathmnist_it_au_eu.pt"):
        tensor_dirs.append((datetime.fromtimestamp(tensor.stat().st_mtime), tensor.parent))

    rows = []
    for row in read_audit_rows(audit_csv):
        if row.get("proper") != "yes":
            continue
        setting = row.get("setting", "")
        if setting != "clean/full" and not setting.startswith("data-scarcity"):
            continue
        log_path = repo_root / row["log_path"]
        output_dir = output_dir_from_log(log_path)
        if output_dir is None and row.get("end") and row["end"] != "Unknown":
            end_time = datetime.fromisoformat(row["end"])
            candidates = sorted(
                (
                    (abs((mtime - end_time).total_seconds()), directory)
                    for mtime, directory in tensor_dirs
                ),
                key=lambda item: item[0],
            )
            if candidates and candidates[0][0] <= 120:
                output_dir = candidates[0][1]
        if output_dir is None:
            continue
        output_dir = relative_to_repo(output_dir, repo_root)
        for decomposition, suffix in DECOMPOSITIONS.items():
            means = tensor_means(
                repo_root / output_dir / f"id_test_hard_pathmnist_{suffix}",
                decomposition,
            )
            if means is None:
                continue
            au_mean, eu_mean, finite_n = means
            rows.append(
                {
                    "method": row["method"],
                    "train_subset": row["train_subset"],
                    "setting": setting,
                    "best_eval_metric": row["best_eval_metric"],
                    "id_test_acc": row["id_test_acc"],
                    "id_test_auroc": row["id_test_auroc"],
                    "decomposition": decomposition,
                    "au_mean": f"{au_mean:.10g}",
                    "eu_mean": f"{eu_mean:.10g}",
                    "finite_n": finite_n,
                    "checkpoint_dir": str(output_dir),
                }
            )
    return rows


def summarize_corruption(repo_root: Path) -> list[dict[str, object]]:
    rows = []
    for decomposition, suffix in DECOMPOSITIONS.items():
        for clean_path in sorted(
            (repo_root / "checkpoints").glob(f"*/id_test_hard_pathmnist-c_{suffix}")
        ):
            checkpoint_dir = clean_path.parent
            clean = tensor_means(clean_path, decomposition)
            if clean is None:
                continue
            corruptions = []
            for path in checkpoint_dir.glob(f"ood_test_hard_pathmnist-c_s1_*_{suffix}"):
                if "_mixed_" in path.name:
                    continue
                name = path.name.removeprefix("ood_test_hard_pathmnist-c_s1_")
                name = name.removesuffix(f"_{suffix}")
                means = tensor_means(path, decomposition)
                if means is None:
                    continue
                corruptions.append((name, means))
            if not corruptions:
                continue
            clean_au, clean_eu, clean_n = clean
            mean_corrupt_au = sum(item[1][0] for item in corruptions) / len(corruptions)
            mean_corrupt_eu = sum(item[1][1] for item in corruptions) / len(corruptions)
            rows.append(
                {
                    "checkpoint_dir": str(relative_to_repo(checkpoint_dir, repo_root)),
                    "decomposition": decomposition,
                    "num_corruptions": len(corruptions),
                    "clean_au_mean": f"{clean_au:.10g}",
                    "clean_eu_mean": f"{clean_eu:.10g}",
                    "mean_corrupted_au": f"{mean_corrupt_au:.10g}",
                    "mean_corrupted_eu": f"{mean_corrupt_eu:.10g}",
                    "delta_au": f"{mean_corrupt_au - clean_au:.10g}",
                    "delta_eu": f"{mean_corrupt_eu - clean_eu:.10g}",
                    "finite_n_clean": clean_n,
                }
            )
    return rows


def summarize_scarcity_margins(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[f"{row['method']}|||{row['decomposition']}"][str(row["train_subset"])] = row

    margins = []
    for method_decomp, by_fraction in sorted(grouped.items()):
        method, decomposition = method_decomp.split("|||", 1)
        full = by_fraction.get("1.0")
        for fraction in ("0.5", "0.1"):
            reduced = by_fraction.get(fraction)
            if full is None or reduced is None:
                continue
            au_delta = float(reduced["au_mean"]) - float(full["au_mean"])
            eu_delta = float(reduced["eu_mean"]) - float(full["eu_mean"])
            margins.append(
                {
                    "method": method,
                    "decomposition": decomposition,
                    "train_subset": fraction,
                    "delta_au": f"{au_delta:.10g}",
                    "delta_eu": f"{eu_delta:.10g}",
                    "scarcity_margin": f"{eu_delta - abs(au_delta):.10g}",
                }
            )
    return margins


def it_only_means(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        if row.get("decomposition") != "information-theoretic":
            continue
        out = dict(row)
        out["it_au_mean"] = out.pop("au_mean")
        out["it_eu_mean"] = out.pop("eu_mean")
        out.pop("decomposition", None)
        filtered.append(out)
    return filtered


def it_only_corruption(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        if row.get("decomposition") != "information-theoretic":
            continue
        out = dict(row)
        out.pop("decomposition", None)
        filtered.append(out)
    return filtered


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    audit_csv = args.audit_csv
    if not audit_csv.is_absolute():
        audit_csv = repo_root / audit_csv
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    scarcity_rows = summarize_scarcity(repo_root, audit_csv)
    write_csv(
        out_dir / "pathmnist_scarcity_all_decompositions_2026-09-16.csv",
        scarcity_rows,
        [
            "method",
            "train_subset",
            "setting",
            "best_eval_metric",
            "id_test_acc",
            "id_test_auroc",
            "decomposition",
            "au_mean",
            "eu_mean",
            "finite_n",
            "checkpoint_dir",
        ],
    )
    write_csv(
        out_dir / "pathmnist_scarcity_it_means_2026-08-20.csv",
        it_only_means(scarcity_rows),
        [
            "method",
            "train_subset",
            "setting",
            "best_eval_metric",
            "id_test_acc",
            "id_test_auroc",
            "it_au_mean",
            "it_eu_mean",
            "finite_n",
            "checkpoint_dir",
        ],
    )
    scarcity_margins = summarize_scarcity_margins(scarcity_rows)
    write_csv(
        out_dir / "pathmnist_scarcity_margins_all_decompositions_2026-09-16.csv",
        scarcity_margins,
        ["method", "decomposition", "train_subset", "delta_au", "delta_eu", "scarcity_margin"],
    )
    write_csv(
        out_dir / "pathmnist_scarcity_margins_2026-08-20.csv",
        [
            {key: value for key, value in row.items() if key != "decomposition"}
            for row in scarcity_margins
            if row.get("decomposition") == "information-theoretic"
        ],
        ["method", "train_subset", "delta_au", "delta_eu", "scarcity_margin"],
    )

    corruption_rows = summarize_corruption(repo_root)
    write_csv(
        out_dir / "pathmnist_c_corruption_all_decompositions_2026-09-16.csv",
        corruption_rows,
        [
            "checkpoint_dir",
            "decomposition",
            "num_corruptions",
            "clean_au_mean",
            "clean_eu_mean",
            "mean_corrupted_au",
            "mean_corrupted_eu",
            "delta_au",
            "delta_eu",
            "finite_n_clean",
        ],
    )
    write_csv(
        out_dir / "pathmnist_c_corruption_it_shift_2026-08-20.csv",
        it_only_corruption(corruption_rows),
        [
            "checkpoint_dir",
            "num_corruptions",
            "clean_au_mean",
            "clean_eu_mean",
            "mean_corrupted_au",
            "mean_corrupted_eu",
            "delta_au",
            "delta_eu",
            "finite_n_clean",
        ],
    )

    print(f"scarcity rows: {len(scarcity_rows)}")
    print(f"corruption rows: {len(corruption_rows)}")
    print(out_dir)


if __name__ == "__main__":
    main()
