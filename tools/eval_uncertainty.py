#!/usr/bin/env python3
"""Run training script in evaluation-only mode for a checkpoint.

This forwards args to `train.py` to ensure wrappers are constructed the same
way they were during training. Use `--extra-args` to pass method-specific
flags (e.g., SNGP options).
"""
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import shlex


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--method", type=str, required=True)
    p.add_argument("--dataset", type=str, default="hard/pathmnist")
    p.add_argument("--data-dir", type=str, default="/tmp/pathmnist_real")
    p.add_argument("--data-dir-id", type=str, default="/tmp/pathmnist_real")
    p.add_argument("--output-dir", type=Path, default=Path("results/pathmnist_uncertainty_decomposition"))
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Additional args forwarded to train.py")

    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"{args.checkpoint.parent.name}__{args.method}__{ts}.log"
    log_path = args.output_dir / log_name

    base_cmd = [
        "python",
        "train.py",
        "--dataset", args.dataset,
        "--dataset-id", args.dataset,
        "--data-dir", args.data_dir,
        "--data-dir-id", args.data_dir_id,
        "--dataset-download",
        "--method-name", args.method,
        "--initial-checkpoint-path", str(args.checkpoint),
        "--evaluate-on-test-sets",
        "--discard-ood-test-sets",
        "--epochs", "0",
        "--num-workers", "2",
        "--num-eval-workers", "2",
        "--storage-device", "cuda",
        "--batch-size", "128",
        "--img-size", "224",
    ]

    if args.extra_args:
        # extra_args comes as a list like ['--gp-kernel-scale', '1.0']
        base_cmd.extend(args.extra_args)

    print("Running:", " ".join(shlex.quote(c) for c in base_cmd))
    print("Logging to:", log_path)

    with open(log_path, "wb") as fh:
        proc = subprocess.run(base_cmd, stdout=fh, stderr=subprocess.STDOUT)

    if proc.returncode == 0:
        print("Evaluation finished successfully.")
    else:
        print(f"Evaluation exited with code {proc.returncode}.")

    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
