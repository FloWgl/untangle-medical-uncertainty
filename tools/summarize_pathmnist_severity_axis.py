#!/usr/bin/env python3
"""Summarize PathMNIST corruption severity-axis outputs from checkpoint folders."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score


METHOD_LABELS = {
    "ce-baseline": "CE baseline",
    "correctness-prediction": "Correctness prediction",
    "deep-correctness-prediction": "Deep correctness prediction",
    "loss-prediction": "Loss prediction",
    "deep-loss-prediction": "Deep loss prediction",
    "mc-dropout": "MC Dropout",
    "edl": "EDL",
    "postnet": "PostNet",
    "sngp": "SNGP",
    "het": "HET",
    "het-xl": "HET-XL",
    "hetclassnn": "HetClassNN",
    "duq": "DUQ",
}

PRIMARY_SCORE = {
    "ce-baseline": "one_minus_max_probs_of_bma",
    "correctness-prediction": "error_probabilities",
    "deep-correctness-prediction": "error_probabilities",
    "loss-prediction": "loss_values",
    "deep-loss-prediction": "loss_values",
    "mc-dropout": "one_minus_expected_max_probs",
    "edl": "dempster_shafer_values",
    "sngp": "one_minus_max_probs_of_bma",
    "het": "one_minus_expected_max_probs",
    "het-xl": "one_minus_expected_max_probs",
    "hetclassnn": "one_minus_expected_max_probs",
    "duq": "duq_values",
}

OOD_RE = re.compile(r"ood_test_hard_pathmnist_s(?P<severity>[1-5])_(?P<corruption>.+)_it_au_eu\.pt$")
CORRUPTIONS = {
    "brightness_down",
    "contrast_up",
    "defocus_blur",
    "motion_blur",
    "jpeg_compression",
    "pixelate",
    "bubble",
    "stain_deposit",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/results/pathmnist"))
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def output_dir_from_row(row: dict[str, str]) -> Path | None:
    for key in ("severity_output_dir", "output_dir"):
        value = row.get(key, "").strip()
        if value:
            return Path(value)
    checkpoint = row.get("checkpoint", "").strip() or row.get("weight_path", "").strip()
    if checkpoint:
        return Path(checkpoint).parent
    return None


def load_tensor(path: Path) -> object:
    return torch.load(path, map_location="cpu", weights_only=False)


def decomposition_pair(path: Path, bregman: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = load_tensor(path)
    first = torch.as_tensor(first).detach().cpu().reshape(-1)
    second = torch.as_tensor(second).detach().cpu().reshape(-1)
    if bregman:
        return second, first
    return first, second


def find_first_key(data: dict[str, object], candidates: tuple[str, ...]) -> torch.Tensor | None:
    for key in candidates:
        value = data.get(key)
        if value is not None:
            return torch.as_tensor(value).detach().cpu().reshape(-1)
    return None


def find_first_value(data: dict[str, object], candidates: tuple[str, ...]) -> object | None:
    for key in candidates:
        if key in data:
            return data[key]
    return None


def score_from_tensor(data: dict[str, object], score_name: str) -> torch.Tensor | None:
    aliases = {
        score_name,
        score_name.replace("_of_bma", ""),
        score_name.replace("one_minus_", ""),
    }
    for key in aliases:
        if key in data:
            return torch.as_tensor(data[key]).detach().cpu().reshape(-1)

    if score_name == "one_minus_max_probs_of_bma":
        probs = find_first_value(data, ("bma_probs", "probs"))
        if probs is not None:
            return 1.0 - torch.as_tensor(probs).detach().cpu().max(dim=-1).values
    if score_name == "one_minus_expected_max_probs":
        probs = find_first_value(data, ("expected_probs", "bma_probs", "probs"))
        if probs is not None:
            return 1.0 - torch.as_tensor(probs).detach().cpu().max(dim=-1).values
    return None


def auroc(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    y_true = torch.cat([torch.zeros_like(clean), torch.ones_like(corrupt)]).numpy()
    y_score = torch.cat([clean, corrupt]).numpy()
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    detailed: list[dict[str, object]] = []

    for row in rows:
        method = row["method"]
        output_dir = output_dir_from_row(row)
        if output_dir is None:
            continue
        clean_path = output_dir / "id_test_hard_pathmnist_it_au_eu.pt"
        if not clean_path.exists():
            continue
        clean_au, clean_eu = decomposition_pair(clean_path)
        clean_total = clean_au + clean_eu
        clean_bregman_path = output_dir / "id_test_hard_pathmnist_bregman_eu_au_hat.pt"
        clean_bregman = (
            decomposition_pair(clean_bregman_path, bregman=True)
            if clean_bregman_path.exists()
            else None
        )
        for ood_path in sorted(output_dir.glob("ood_test_hard_pathmnist_s*_it_au_eu.pt")):
            match = OOD_RE.match(ood_path.name)
            if match is None:
                continue
            corruption = match.group("corruption")
            if corruption not in CORRUPTIONS:
                continue
            corrupt_au, corrupt_eu = decomposition_pair(ood_path)
            corrupt_total = corrupt_au + corrupt_eu
            result: dict[str, object] = {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "severity": match.group("severity"),
                "corruption": corruption,
                "output_dir": str(output_dir),
                "it_au_ood_auroc": f"{auroc(clean_au, corrupt_au):.6f}",
                "it_eu_ood_auroc": f"{auroc(clean_eu, corrupt_eu):.6f}",
                "total_entropy_ood_auroc": f"{auroc(clean_total, corrupt_total):.6f}",
            }
            bregman_path = Path(str(ood_path).replace("_it_au_eu.pt", "_bregman_eu_au_hat.pt"))
            if clean_bregman is not None and bregman_path.exists():
                corrupt_bregman_au, corrupt_bregman_eu = decomposition_pair(
                    bregman_path, bregman=True
                )
                result["bregman_au_ood_auroc"] = f"{auroc(clean_bregman[0], corrupt_bregman_au):.6f}"
                result["bregman_eu_ood_auroc"] = f"{auroc(clean_bregman[1], corrupt_bregman_eu):.6f}"
            detailed.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = args.output_dir / "pathmnist_c_severity_axis_detailed.csv"
    fieldnames = [
        "method",
        "method_label",
        "severity",
        "corruption",
        "it_au_ood_auroc",
        "it_eu_ood_auroc",
        "total_entropy_ood_auroc",
        "bregman_au_ood_auroc",
        "bregman_eu_ood_auroc",
        "output_dir",
    ]
    with detailed_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed)

    by_method_severity: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in detailed:
        by_method_severity.setdefault((str(row["method"]), str(row["severity"])), []).append(row)

    summary_path = args.output_dir / "pathmnist_c_severity_axis_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "method_label",
                "severity",
                "n_corruptions",
                "it_au_ood_auroc_mean",
                "it_eu_ood_auroc_mean",
                "total_entropy_ood_auroc_mean",
                "bregman_au_ood_auroc_mean",
                "bregman_eu_ood_auroc_mean",
            ],
        )
        writer.writeheader()
        for (method, severity), group in sorted(by_method_severity.items()):
            metric_names = (
                "it_au_ood_auroc",
                "it_eu_ood_auroc",
                "total_entropy_ood_auroc",
                "bregman_au_ood_auroc",
                "bregman_eu_ood_auroc",
            )
            means = {
                name: [float(row[name]) for row in group if row.get(name)]
                for name in metric_names
            }
            writer.writerow(
                {
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "severity": severity,
                    "n_corruptions": len(group),
                    **{
                        f"{name}_mean": f"{sum(values) / len(values):.6f}" if values else ""
                        for name, values in means.items()
                    },
                }
            )

    print(f"Wrote {len(detailed)} detailed rows to {detailed_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
