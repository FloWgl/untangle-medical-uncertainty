#!/usr/bin/env python3
"""Build a PathMNIST checkpoint manifest from completed TinyGPU SLURM logs."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
from pathlib import Path


def parse_method(command: str) -> str | None:
    parts = shlex.split(command)
    if "--method-name" not in parts:
        return None
    method = parts[parts.index("--method-name") + 1]
    if method == "het-xl" and "--use-het" in parts:
        return "het"
    return method


def parse_seed(command: str) -> str:
    parts = shlex.split(command)
    if "--seed" not in parts:
        return ""
    return parts[parts.index("--seed") + 1]


def command_from_stdout(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("Command: "):
            return line.removeprefix("Command: ").strip()
    return None


def output_dir_from_stderr(path: Path) -> Path | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    if "Tests took" not in text:
        return None
    matches = re.findall(r"Output directory is ([^.]+)\.", text)
    if not matches:
        return None
    return Path(matches[-1])


def collect_rows(repo_root: Path, job_ids: list[str]) -> list[dict[str, str]]:
    rows_by_method: dict[str, dict[str, str]] = {}
    slurm_dir = repo_root / "logs" / "slurm"
    for job_id in job_ids:
        for stdout in sorted(slurm_dir.glob(f"*_{job_id}_*.out")):
            match = re.search(rf"_{re.escape(job_id)}_(\d+)\.out$", stdout.name)
            if not match:
                continue
            task_id = match.group(1)
            stderr = stdout.with_suffix(".err")
            command = command_from_stdout(stdout)
            output_dir = output_dir_from_stderr(stderr)
            if command is None or output_dir is None:
                continue
            method = parse_method(command)
            if method is None:
                continue
            checkpoint_best = repo_root / output_dir / "checkpoint_best.pt"
            checkpoint_last = repo_root / output_dir / "checkpoint_last.pt"
            if not checkpoint_best.is_file():
                continue
            row = {
                "method": method,
                "checkpoint": str(checkpoint_best),
                "checkpoint_best": str(checkpoint_best),
                "checkpoint_last": str(checkpoint_last) if checkpoint_last.is_file() else "",
                "job_id": job_id,
                "task_id": task_id,
                "seed": parse_seed(command),
                "output_dir": str(repo_root / output_dir),
            }
            # Prefer the first successfully completed task for each method. The
            # job-id order supplied by the watcher places fresh reruns first.
            rows_by_method.setdefault(method, row)
    return [rows_by_method[key] for key in sorted(rows_by_method)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--job-ids", required=True, help="Comma-separated SLURM job IDs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    job_ids = [part.strip() for part in args.job_ids.split(",") if part.strip()]
    rows = collect_rows(args.repo_root.resolve(), job_ids)
    if not rows:
        raise SystemExit("No completed PathMNIST checkpoints found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "checkpoint",
                "checkpoint_best",
                "checkpoint_last",
                "job_id",
                "task_id",
                "seed",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
