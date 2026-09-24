#!/usr/bin/env python3
"""Aggregate available CIFAR-10 results into a CSV inventory.

This script scans `checkpoints/` for checkpoint directories, records whether
`checkpoint_best.pt` exists, lists any `id_test_*.pt` metric files, and
attempts to guess the method name by scanning `logs/` for the checkpoint directory
name. It writes a CSV to `results/cifar10_metrics_inventory.csv`.

Optional: `--evaluate-missing` can run `tools/eval_uncertainty.py` for entries
that don't have any `id_test_*.pt` files if a method mapping is provided with
`--method-map mapping.csv` (format: checkpoint_dir,method,extra_args).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import shlex
import sys


def guess_method_from_logs(checkpoint_name: str, logs_dir: Path) -> str | None:
    """Heuristic: search logs for the checkpoint_name and extract --method-name."""
    if not logs_dir.exists():
        return None
    for p in logs_dir.iterdir():
        if not p.is_file():
            continue
        try:
            s = p.read_text(errors="ignore")
        except Exception:
            continue
        if checkpoint_name in s:
            # try to find '--method-name' occurrence
            idx = s.find("--method-name")
            if idx != -1:
                rest = s[idx:idx+200]
                parts = rest.split()
                if len(parts) >= 2:
                    return parts[1].strip()
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--logs-dir", type=Path, default=Path("logs"))
    p.add_argument("--out-csv", type=Path, default=Path("results/cifar10_metrics_inventory.csv"))
    p.add_argument("--evaluate-missing", action="store_true")
    p.add_argument("--method-map", type=Path, default=None, help="Optional CSV with mapping: checkpoint_dir,method,extra_args")
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    mapping = {}
    if args.method_map and args.method_map.exists():
        with args.method_map.open() as fh:
            rdr = csv.reader(fh)
            for row in rdr:
                if not row:
                    continue
                mapping[row[0]] = (row[1] if len(row) > 1 else "", row[2] if len(row) > 2 else "")

    rows = []
    for d in sorted(args.checkpoints_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        best = (d / "checkpoint_best.pt").exists()
        metric_files = [f.name for f in d.iterdir() if f.is_file() and f.name.startswith("id_test_")]
        method_guess = mapping.get(name, (None, ""))[0] or guess_method_from_logs(name, args.logs_dir)

        rows.append({
            "checkpoint_dir": name,
            "checkpoint_best": best,
            "num_metric_files": len(metric_files),
            "metric_files": ";".join(sorted(metric_files)),
            "method_guess": method_guess or "",
        })

    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "checkpoint_dir",
                "checkpoint_best",
                "num_metric_files",
                "metric_files",
                "method_guess",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote inventory to {args.out_csv}")

    if args.evaluate_missing:
        # Evaluate entries without metric files if mapping provided
        for r in rows:
            if r["num_metric_files"] == 0:
                name = r["checkpoint_dir"]
                if name in mapping:
                    method, extra = mapping[name]
                else:
                    method = r["method_guess"]
                    extra = ""
                if not method:
                    print(f"Skipping {name}: no method mapping or guess available")
                    continue
                ck = args.checkpoints_dir / name / "checkpoint_best.pt"
                if not ck.exists():
                    print(f"Skipping {name}: checkpoint_best.pt missing")
                    continue
                cmd = ["python","tools/eval_uncertainty.py","--checkpoint", str(ck), "--method", method, "--dataset", "hard/cifar10", "--data-dir", "./data", "--data-dir-id", "./data", "--output-dir", "results/cifar10_eval"]
                if extra:
                    cmd += shlex.split(extra)
                print("Running:", " ".join(shlex.quote(c) for c in cmd))
                subprocess.run(cmd)


if __name__ == "__main__":
    main()
