#!/usr/bin/env python3
"""Add Bregman AU/EU OOD AUROC values to the CIFAR-10C detailed results."""

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / "docs/results/cifar10/cifar10c_ood_auroc_detailed.csv"
OUTPUT = ROOT / "docs/results/cifar10/cifar10c_bregman_ood_auroc_summary.csv"


def pair(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    eu, au = torch.load(path, map_location="cpu", weights_only=True)
    return au.detach().float().reshape(-1), eu.detach().float().reshape(-1)


def auroc(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    finite_clean = clean[torch.isfinite(clean)]
    finite_corrupt = corrupt[torch.isfinite(corrupt)]
    labels = torch.cat([torch.zeros_like(finite_clean), torch.ones_like(finite_corrupt)]).numpy()
    scores = torch.cat([finite_clean, finite_corrupt]).numpy()
    return float(roc_auc_score(labels, scores))


def main() -> None:
    detailed = pd.read_csv(DETAIL)
    rows = []
    for _, row in detailed.iterrows():
        directory = ROOT / row.output_dir
        clean_path = directory / "id_test_soft_cifar10_bregman_eu_au_hat.pt"
        corrupt_path = directory / (
            f"ood_test_soft_cifar10_s{int(row.severity)}_{row.corruption}_bregman_eu_au_hat.pt"
        )
        if not clean_path.exists() or not corrupt_path.exists():
            continue
        clean_au, clean_eu = pair(clean_path)
        corrupt_au, corrupt_eu = pair(corrupt_path)
        rows.append(
            {
                "phase": row.phase,
                "train_fraction": row.train_fraction,
                "method": row.method,
                "queue_run": row.queue_run,
                "severity": int(row.severity),
                "corruption": row.corruption,
                "bregman_au_ood_auroc": auroc(clean_au, corrupt_au),
                "bregman_eu_ood_auroc": auroc(clean_eu, corrupt_eu),
            }
        )
    values = pd.DataFrame(rows)
    summary = (
        values.groupby(["phase", "train_fraction", "method", "queue_run"], as_index=False)
        .agg(
            severity2_bregman_au_ood_auroc_mean=("bregman_au_ood_auroc", lambda x: x[values.loc[x.index, "severity"] == 2].mean()),
            severity2_bregman_eu_ood_auroc_mean=("bregman_eu_ood_auroc", lambda x: x[values.loc[x.index, "severity"] == 2].mean()),
            all_bregman_au_ood_auroc_mean=("bregman_au_ood_auroc", "mean"),
            all_bregman_eu_ood_auroc_mean=("bregman_eu_ood_auroc", "mean"),
            n_corruption_points=("corruption", "size"),
        )
    )
    summary.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(summary)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
