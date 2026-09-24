#!/usr/bin/env python3
"""Summarize PathMNIST scarcity or label-noise decomposition outputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shlex
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--log-template", required=True)
    parser.add_argument("--intervention", choices=("scarcity", "label-noise"), required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def option(parts: list[str], name: str, default: str = "") -> str:
    try:
        return parts[parts.index(name) + 1]
    except (ValueError, IndexError):
        return default


def method_name(parts: list[str]) -> str:
    method = option(parts, "--method-name")
    if method == "het-xl" and "--use-het" in parts:
        return "het"
    return method


def output_dir(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    matches = re.findall(r"Output directory is ([^.\n]+)", log_path.read_text(errors="replace"))
    return Path(matches[-1].strip()) if matches else None


def means(path: Path, bregman: bool = False) -> tuple[float, float, int] | None:
    if not path.exists():
        return None
    first, second = torch.load(path, map_location="cpu", weights_only=True)
    first = first.detach().float().reshape(-1)
    second = second.detach().float().reshape(-1)
    aleatoric, epistemic = (second, first) if bregman else (first, second)
    mask = torch.isfinite(aleatoric) & torch.isfinite(epistemic)
    if not mask.any():
        return math.nan, math.nan, 0
    return float(aleatoric[mask].mean()), float(epistemic[mask].mean()), int(mask.sum())


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    commands = [line.strip() for line in args.commands.read_text().splitlines() if line.strip()]
    for task, command in enumerate(commands, start=1):
        parts = shlex.split(command)
        log_path = Path(args.log_template.format(task=task))
        directory = output_dir(log_path)
        if directory is None:
            continue
        setting = (
            option(parts, "--train-subset", "1.0")
            if args.intervention == "scarcity"
            else option(parts, "--label-noise-fraction", "0.0")
        )
        for decomposition, suffix, is_bregman in (
            ("information-theoretic", "it_au_eu.pt", False),
            ("bregman-estimated", "bregman_eu_au_hat.pt", True),
        ):
            summary = means(directory / f"id_test_hard_pathmnist_{suffix}", is_bregman)
            if summary is None:
                continue
            au_mean, eu_mean, finite_n = summary
            rows.append(
                {
                    "intervention": args.intervention,
                    "method": method_name(parts),
                    "level": setting,
                    "decomposition": decomposition,
                    "au_mean": f"{au_mean:.10g}",
                    "eu_mean": f"{eu_mean:.10g}",
                    "finite_n": finite_n,
                    "task": task,
                    "output_dir": str(directory),
                    "log": str(log_path),
                }
            )

    fields = [
        "intervention",
        "method",
        "level",
        "decomposition",
        "au_mean",
        "eu_mean",
        "finite_n",
        "task",
        "output_dir",
        "log",
    ]
    handle = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            handle.close()


if __name__ == "__main__":
    main()
