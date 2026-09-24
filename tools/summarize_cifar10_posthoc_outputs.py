#!/usr/bin/env python3
"""Summarize finished CIFAR-10 post-hoc output directories."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import torch


def finite_mean(values: torch.Tensor) -> float:
    flat = values.detach().float().cpu().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return math.nan
    return float(finite.mean().item())


def load_pair(path: Path) -> tuple[float, float, int, int]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        raise ValueError(f"Expected tensor pair in {path}")
    au = obj[0].detach().float().cpu().reshape(-1)
    eu = obj[1].detach().float().cpu().reshape(-1)
    finite = torch.isfinite(au) & torch.isfinite(eu)
    return finite_mean(au), finite_mean(eu), int(finite.sum().item()), int(au.numel())


def parse_test_metrics(log_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    pattern = re.compile(r"Test metric ([^:]+): ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")
    if not log_path.is_file():
        return metrics
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def mean_metric(metrics: dict[str, float], *, contains: str, severity: str | None = None) -> float:
    values = []
    for key, value in metrics.items():
        if contains not in key:
            continue
        if severity is not None and f"_s{severity}_" not in key:
            continue
        if "_mixed_soft_cifar10_" not in key:
            continue
        values.append(value)
    if not values:
        return math.nan
    return sum(values) / len(values)


def summarize(method: str, output_dir: Path, log_path: Path) -> dict[str, str]:
    row: dict[str, str] = {"method": method, "output_dir": str(output_dir), "log": str(log_path)}
    for name, filename in {
        "it": "id_test_soft_cifar10_it_au_eu.pt",
        "bregman": "id_test_soft_cifar10_bregman_au_eu.pt",
        "mahalanobis": "id_test_soft_cifar10_mahalanobis_au_eu.pt",
    }.items():
        path = output_dir / filename
        if not path.is_file():
            row[f"{name}_au_mean"] = ""
            row[f"{name}_eu_mean"] = ""
            row[f"{name}_finite"] = ""
            row[f"{name}_count"] = ""
            continue
        au_mean, eu_mean, finite, count = load_pair(path)
        row[f"{name}_au_mean"] = f"{au_mean:.10g}"
        row[f"{name}_eu_mean"] = f"{eu_mean:.10g}"
        row[f"{name}_finite"] = str(finite)
        row[f"{name}_count"] = str(count)

    metrics = parse_test_metrics(log_path)
    for score in [
        "one_minus_max_probs_of_bma_auroc_oodness",
        "one_minus_expected_max_probs_auroc_oodness",
        "jensen_shannon_divergences_auroc_oodness",
        "mahalanobis_values_auroc_oodness",
    ]:
        row[f"ood_s2_mean_{score}"] = f"{mean_metric(metrics, contains=score, severity='2'):.10g}"
        row[f"ood_all_mean_{score}"] = f"{mean_metric(metrics, contains=score):.10g}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("items", nargs="+", help="method=output_dir=log_path")
    args = parser.parse_args()

    rows = []
    for item in args.items:
        parts = item.split("=", 2)
        if len(parts) != 3:
            raise SystemExit(f"Expected method=output_dir=log_path, got {item}")
        rows.append(summarize(parts[0], Path(parts[1]), Path(parts[2])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
