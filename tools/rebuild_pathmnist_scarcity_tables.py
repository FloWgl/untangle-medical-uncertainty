#!/usr/bin/env python3
"""Rebuild PathMNIST scarcity tables from logged commands and result tensors."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "docs/results/pathmnist"
LOG_GLOB = "pathmnist-reduced_1765084_*.out"
DECOMPOSITIONS = {
    "information-theoretic": "id_test_hard_pathmnist_it_au_eu.pt",
    "bregman-estimated": "id_test_hard_pathmnist_bregman_eu_au_hat.pt",
}
CLEAN_FILES = (
    "pathmnist_clean_main_decompositions_2026-09-17.csv",
    "pathmnist_clean_gap_decompositions_2026-09-17.csv",
    "pathmnist_clean_shapefix_decompositions_2026-09-17.csv",
    "pathmnist_clean_posthoc_base_decompositions_2026-09-17.csv",
)


def option(tokens: list[str], name: str) -> str:
    return tokens[tokens.index(name) + 1]


def tensor_means(path: Path, decomposition: str) -> tuple[float, float, int]:
    first, second = torch.load(path, map_location="cpu", weights_only=True)
    if decomposition == "bregman-estimated":
        epistemic, aleatoric = first.float(), second.float()
    else:
        aleatoric, epistemic = first.float(), second.float()
    finite = torch.isfinite(aleatoric) & torch.isfinite(epistemic)
    return (
        float(aleatoric[finite].mean()),
        float(epistemic[finite].mean()),
        int(finite.sum()),
    )


def scarcity_rows() -> pd.DataFrame:
    rows = []
    for out_log in sorted((ROOT / "logs/slurm").glob(LOG_GLOB)):
        text = out_log.read_text(errors="replace")
        command = re.search(r"^Command: (.+)$", text, re.MULTILINE)
        if command is None:
            continue
        tokens = shlex.split(command.group(1))
        method = option(tokens, "--method-name")
        if method == "het-xl" and "--use-het" in tokens:
            method = "het"
        level = float(option(tokens, "--train-subset"))
        task = int(out_log.stem.rsplit("_", 1)[1])
        err_log = out_log.with_suffix(".err")
        err_text = err_log.read_text(errors="replace")
        output_match = re.findall(r"Output directory is ([^.\n]+)", err_text)
        if not output_match or "Tests took" not in err_text:
            continue
        output_dir = Path(output_match[-1])
        for decomposition, filename in DECOMPOSITIONS.items():
            au, eu, finite_n = tensor_means(ROOT / output_dir / filename, decomposition)
            rows.append({
                "intervention": "scarcity",
                "method": method,
                "level": level,
                "decomposition": decomposition,
                "au_mean": au,
                "eu_mean": eu,
                "finite_n": finite_n,
                "task": task,
                "output_dir": str(output_dir),
                "log": str(err_log.relative_to(ROOT)),
            })
    return pd.DataFrame(rows).sort_values(["level", "task", "decomposition"], ascending=[False, True, False])


def clean_rows() -> pd.DataFrame:
    clean = pd.concat([pd.read_csv(PROGRESS / name) for name in CLEAN_FILES])
    # Task 1 is the designated CE baseline; the remaining CE rows are ensemble members.
    clean = clean[(clean.method != "ce-baseline") | (clean.task == 1)]
    edl_dir = Path("checkpoints/20260603-160030-087887-timm_resnet_50-224")
    edl_rows = []
    for decomposition, filename in DECOMPOSITIONS.items():
        au, eu, finite_n = tensor_means(ROOT / edl_dir / filename, decomposition)
        edl_rows.append({
            "intervention": "clean",
            "method": "edl",
            "level": 1.0,
            "decomposition": decomposition,
            "au_mean": au,
            "eu_mean": eu,
            "finite_n": finite_n,
            "task": 7,
            "output_dir": str(edl_dir),
            "log": "logs/slurm/pathmnist-train_1676043_7.err",
        })
    return pd.concat([clean, pd.DataFrame(edl_rows)]).drop_duplicates(
        ["method", "decomposition"], keep="first"
    )


def delta_rows(scarcity: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    merged = scarcity.merge(
        clean[["method", "decomposition", "au_mean", "eu_mean"]],
        on=["method", "decomposition"],
        suffixes=("_intervention", "_clean"),
        validate="many_to_one",
    )
    return pd.DataFrame({
        "intervention": "scarcity",
        "method": merged.method,
        "level": merged.level,
        "decomposition": merged.decomposition,
        "clean_au": merged.au_mean_clean,
        "intervention_au": merged.au_mean_intervention,
        "delta_au": merged.au_mean_intervention - merged.au_mean_clean,
        "clean_eu": merged.eu_mean_clean,
        "intervention_eu": merged.eu_mean_intervention,
        "delta_eu": merged.eu_mean_intervention - merged.eu_mean_clean,
        "finite_n": merged.finite_n,
        "source_output_dir": merged.output_dir,
    })


def main() -> None:
    scarcity = scarcity_rows()
    expected = {(method, level) for method in scarcity.method.unique() for level in (0.5, 0.1)}
    observed = set(zip(scarcity.method, scarcity.level))
    if expected != observed or len(scarcity) != 56:
        raise RuntimeError(f"Incomplete scarcity grid: missing {sorted(expected - observed)}")
    deltas = delta_rows(scarcity, clean_rows())
    scarcity.to_csv(PROGRESS / "pathmnist_scarcity_all_decompositions_2026-09-17.csv", index=False)
    deltas.to_csv(PROGRESS / "pathmnist_scarcity_deltas_all_decompositions_2026-09-17.csv", index=False)
    print(f"Wrote {len(scarcity)} scarcity rows and {len(deltas)} delta rows")


if __name__ == "__main__":
    main()
