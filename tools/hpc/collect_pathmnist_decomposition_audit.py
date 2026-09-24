#!/usr/bin/env python3
"""Build an audit CSV for fresh PathMNIST decomposition reruns."""

from __future__ import annotations

import argparse
import csv
import re
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--job-ids", required=True, help="Comma-separated SLURM job IDs")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def command_from_stdout(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("Command: "):
            return line.removeprefix("Command: ").strip()
    return None


def option(parts: list[str], name: str, default: str = "") -> str:
    if name not in parts:
        return default
    index = parts.index(name) + 1
    if index >= len(parts):
        return default
    return parts[index]


def parse_method(parts: list[str]) -> str:
    method = option(parts, "--method-name")
    if method == "het-xl" and "--use-het" in parts:
        return "het"
    return method


def output_dir_from_log(text: str) -> str:
    matches = re.findall(r"Output directory is ([^.]+)\.", text)
    return matches[-1] if matches else ""


def metric(text: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}: ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)", text)
    return match.group(1) if match else ""


def best_metric(text: str) -> tuple[str, str]:
    match = re.search(r"Best eval metric: ([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?: \(epoch (\d+)\))?", text)
    if not match:
        return "", ""
    return match.group(1), match.group(2) or ""


def collect(repo_root: Path, job_ids: list[str]) -> list[dict[str, str]]:
    rows = []
    slurm_dir = repo_root / "logs" / "slurm"
    for job_id in job_ids:
        for stdout in sorted(slurm_dir.glob(f"*_{job_id}_*.out")):
            task_match = re.search(rf"_{re.escape(job_id)}_(\d+)\.out$", stdout.name)
            if not task_match:
                continue
            stderr = stdout.with_suffix(".err")
            command = command_from_stdout(stdout)
            if command is None or not stderr.is_file():
                continue
            parts = shlex.split(command)
            text = stderr.read_text(errors="replace")
            train_subset = option(parts, "--train-subset", "1.0")
            label_noise = option(parts, "--label-noise-fraction")
            if label_noise:
                setting = f"label-noise {label_noise}"
            elif train_subset == "1.0":
                setting = "clean/full"
            else:
                setting = f"data-scarcity {train_subset}"
            output_dir = output_dir_from_log(text)
            proper = "yes" if "Tests took" in text and output_dir else "no"
            best_eval_metric, best_epoch = best_metric(text)
            rows.append(
                {
                    "job_id": job_id,
                    "task": task_match.group(1),
                    "job_name": "",
                    "state": "COMPLETED" if proper == "yes" else "",
                    "exit_code": "0:0" if proper == "yes" else "",
                    "proper": proper,
                    "method": parse_method(parts),
                    "setting": setting,
                    "train_subset": train_subset,
                    "label_noise": label_noise,
                    "dataset_id": option(parts, "--dataset-id"),
                    "best_eval_metric": best_eval_metric,
                    "best_epoch": best_epoch,
                    "last_eval_accuracy": "",
                    "id_test_acc": metric(text, "Test metric id_test_hard_bma_accuracy_original"),
                    "id_test_auroc": metric(text, "Test metric id_test_confidence_auroc_hard_bma_correctness_original"),
                    "tests_took": "yes" if "Tests took" in text else "no",
                    "traceback": "yes" if "Traceback (most recent call last)" in text else "no",
                    "elapsed": "",
                    "end": "Unknown",
                    "log_path": str(stderr.relative_to(repo_root)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    job_ids = [part.strip() for part in args.job_ids.split(",") if part.strip()]
    rows = collect(repo_root, job_ids)
    if not rows:
        raise SystemExit("No PathMNIST decomposition audit rows found.")

    fields = [
        "job_id",
        "task",
        "job_name",
        "state",
        "exit_code",
        "proper",
        "method",
        "setting",
        "train_subset",
        "label_noise",
        "dataset_id",
        "best_eval_metric",
        "best_epoch",
        "last_eval_accuracy",
        "id_test_acc",
        "id_test_auroc",
        "tests_took",
        "traceback",
        "elapsed",
        "end",
        "log_path",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
