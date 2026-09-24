#!/usr/bin/env python3
"""Generate SLURM command manifests for the PathMNIST benchmark."""

from __future__ import annotations

import argparse
import csv
import shlex
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


PATHMNIST_C_TRANSFORMS = (
    "brightness_down",
    "contrast_up",
    "defocus_blur",
    "motion_blur",
    "jpeg_compression",
    "pixelate",
    "bubble",
    "stain_deposit",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    args: tuple[str, ...]
    trainable: bool = True
    posthoc: bool = False
    checkpoint_key: str | None = None


TRAINABLE_METHODS = (
    MethodSpec("ce-baseline", ("--method-name", "ce-baseline", "--loss", "cross-entropy")),
    MethodSpec(
        "correctness-prediction",
        (
            "--method-name",
            "correctness-prediction",
            "--loss",
            "correctness-prediction",
            "--lambda-uncertainty-loss",
            "0.01",
        ),
    ),
    MethodSpec(
        "deep-correctness-prediction",
        (
            "--method-name",
            "deep-correctness-prediction",
            "--loss",
            "correctness-prediction",
            "--lambda-uncertainty-loss",
            "0.01",
        ),
    ),
    MethodSpec(
        "loss-prediction",
        (
            "--method-name",
            "loss-prediction",
            "--loss",
            "loss-prediction",
            "--lambda-uncertainty-loss",
            "0.01",
        ),
    ),
    MethodSpec(
        "deep-loss-prediction",
        (
            "--method-name",
            "deep-loss-prediction",
            "--loss",
            "loss-prediction",
            "--lambda-uncertainty-loss",
            "0.01",
        ),
    ),
    MethodSpec(
        "mc-dropout",
        (
            "--method-name",
            "mc-dropout",
            "--loss",
            "cross-entropy",
            "--dropout-probability",
            "0.05",
            "--num-mc-samples",
            "10",
        ),
    ),
    MethodSpec(
        "edl",
        (
            "--method-name",
            "edl",
            "--loss",
            "edl",
            "--edl-start-epoch",
            "0",
            "--edl-scaler",
            "1.0",
            "--edl-activation",
            "exp",
        ),
    ),
    MethodSpec(
        "sngp",
        (
            "--method-name",
            "sngp",
            "--loss",
            "cross-entropy",
            "--use-spectral-normalization",
            "--spectral-normalization-iteration",
            "1",
            "--spectral-normalization-bound",
            "6",
            "--num-random-features",
            "1024",
            "--gp-kernel-scale",
            "1.0",
            "--gp-output-bias",
            "0.0",
            "--gp-random-feature-type",
            "orf",
            "--gp-cov-momentum",
            "-1",
            "--gp-cov-ridge-penalty",
            "1.0",
            "--gp-input-dim",
            "128",
        ),
    ),
    MethodSpec(
        "postnet",
        (
            "--method-name",
            "postnet",
            "--loss",
            "uce",
            "--latent-dim",
            "6",
            "--num-hidden-features",
            "256",
            "--num-density-components",
            "6",
            "--uce-regularization-factor",
            "1e-5",
        ),
    ),
    MethodSpec(
        "het",
        (
            "--method-name",
            "het-xl",
            "--loss",
            "bma-cross-entropy",
            "--use-het",
            "--num-mc-samples",
            "10",
            "--temperature",
            "1.5",
        ),
    ),
    MethodSpec(
        "het-xl",
        (
            "--method-name",
            "het-xl",
            "--loss",
            "bma-cross-entropy",
            "--matrix-rank",
            "15",
            "--num-mc-samples",
            "10",
            "--temperature",
            "1.5",
        ),
    ),
    MethodSpec(
        "hetclassnn",
        (
            "--method-name",
            "hetclassnn",
            "--loss",
            "bma-cross-entropy",
            "--dropout-probability",
            "0.05",
            "--num-mc-samples",
            "10",
            "--num-mc-samples-integral",
            "1000",
        ),
    ),
    MethodSpec(
        "shallow-ensemble",
        (
            "--method-name",
            "shallow-ensemble",
            "--loss",
            "bma-cross-entropy",
            "--num-heads",
            "10",
        ),
    ),
    MethodSpec(
        "duq",
        (
            "--method-name",
            "duq",
            "--loss",
            "duq",
            "--rbf-length-scale",
            "0.1",
            "--ema-momentum",
            "0.999",
            "--lambda-gradient-penalty",
            "0.75",
        ),
    ),
)


POSTHOC_METHODS = (
    MethodSpec(
        "temperature-scaling",
        ("--method-name", "temperature-scaling", "--loss", "cross-entropy"),
        trainable=False,
        posthoc=True,
        checkpoint_key="ce-baseline",
    ),
    MethodSpec(
        "mahalanobis",
        (
            "--method-name",
            "mahalanobis",
            "--loss",
            "cross-entropy",
            "--magnitude",
            "0.001",
        ),
        trainable=False,
        posthoc=True,
        checkpoint_key="ce-baseline",
    ),
    MethodSpec(
        "ddu",
        (
            "--method-name",
            "ddu",
            "--loss",
            "cross-entropy",
            "--use-spectral-normalization",
            "--spectral-normalization-iteration",
            "1",
            "--spectral-normalization-bound",
            "6",
            "--use-spectral-normalized-batch-norm",
        ),
        trainable=False,
        posthoc=True,
        checkpoint_key="sngp",
    ),
    MethodSpec(
        "swag",
        (
            "--method-name",
            "swag",
            "--loss",
            "cross-entropy",
            "--use-low-rank-cov",
            "--max-rank",
            "20",
            "--num-checkpoints-per-epoch",
            "4",
        ),
        trainable=False,
        posthoc=True,
        checkpoint_key="ce-baseline",
    ),
    MethodSpec(
        "laplace",
        (
            "--method-name",
            "laplace",
            "--loss",
            "cross-entropy",
            "--num-mc-samples",
            "10",
            "--num-mc-samples-cv",
            "50",
            "--pred-type",
            "glm",
            "--hessian-structure",
            "kron",
        ),
        trainable=False,
        posthoc=True,
        checkpoint_key="ce-baseline",
    ),
)


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(
        str(part) if "$" in str(part) else shlex.quote(str(part)) for part in parts
    )


def add_common_args(
    *,
    args: argparse.Namespace,
    dataset_id: str,
    data_dir_id: str,
    epochs: int,
    train_subset: float,
    label_noise_fraction: float = 0.0,
) -> list[str]:
    command = [
        args.python,
        "train.py",
        "--dataset",
        "hard/pathmnist",
        "--dataset-id",
        dataset_id,
        "--data-dir",
        args.pathmnist_dir,
        "--data-dir-id",
        data_dir_id,
        "--dataset-download",
        "--model-name",
        args.model_name,
        "--num-classes",
        "9",
        "--opt",
        "adamw",
        "--weight-decay",
        str(args.weight_decay),
        "--lr-base",
        str(args.lr_base),
        "--batch-size",
        str(args.batch_size),
        "--accumulation-steps",
        str(args.accumulation_steps),
        "--epochs",
        str(epochs),
        "--seed",
        str(args.seed),
        "--img-size",
        str(args.img_size),
        "--mean",
        args.mean,
        "--std",
        args.std,
        "--train-subset",
        str(train_subset),
        "--evaluate-on-test-sets",
        "--storage-device",
        args.storage_device,
        "--num-workers",
        str(args.num_workers),
        "--num-eval-workers",
        str(args.num_eval_workers),
    ]
    if label_noise_fraction > 0.0:
        command.extend(
            [
                "--label-noise-fraction",
                str(label_noise_fraction),
                "--label-noise-mode",
                "symmetric",
                "--label-noise-seed",
                str(args.label_noise_seed),
            ]
        )
    return command


def command_for_method(
    *,
    args: argparse.Namespace,
    method: MethodSpec,
    epochs: int,
    train_subset: float,
    label_noise_fraction: float = 0.0,
    seed: int | None = None,
    dataset_id: str = "hard/pathmnist",
    data_dir_id: str | None = None,
    discard_ood: bool = True,
    severities: tuple[int, ...] | None = None,
    weight_paths: str | None = None,
) -> str:
    cmd = add_common_args(
        args=args,
        dataset_id=dataset_id,
        data_dir_id=data_dir_id or args.pathmnist_dir,
        epochs=epochs,
        train_subset=train_subset,
        label_noise_fraction=label_noise_fraction,
    )
    cmd.extend(method.args)
    if seed is not None:
        seed_index = cmd.index("--seed") + 1
        cmd[seed_index] = str(seed)
    if discard_ood:
        cmd.append("--discard-ood-test-sets")
    else:
        transforms = ",".join(PATHMNIST_C_TRANSFORMS)
        severity_args = tuple(severities or (1,))
        cmd.extend(
            [
                "--severities",
                ",".join(str(severity) for severity in severity_args),
                "--ood-transforms-eval",
                transforms,
                "--ood-transforms-test",
                transforms,
            ]
        )
    if weight_paths is not None:
        cmd.extend(["--weight-paths", weight_paths])
    if args.amp:
        cmd.append("--amp")
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.log_wandb:
        cmd.append("--log-wandb")
    return shell_join(cmd)


def read_checkpoint_manifest(path: Path) -> dict[str, str]:
    checkpoints: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row.get("method")
            checkpoint = row.get("checkpoint") or row.get("weight_path")
            if method and checkpoint:
                checkpoints[method] = checkpoint
    return checkpoints


def selected_trainable_methods(args: argparse.Namespace) -> tuple[MethodSpec, ...]:
    if not args.methods:
        return TRAINABLE_METHODS
    wanted = set(args.methods.split(","))
    return tuple(method for method in TRAINABLE_METHODS if method.name in wanted)


def generate_commands(args: argparse.Namespace) -> list[str]:
    commands: list[str] = []
    trainable_methods = selected_trainable_methods(args)

    if args.phase in {"train", "all"}:
        for method in trainable_methods:
            commands.append(
                command_for_method(
                    args=args,
                    method=method,
                    epochs=args.epochs,
                    train_subset=1.0,
                )
            )
        if not args.skip_ensemble_seeds:
            for seed in args.ensemble_seeds:
                commands.append(
                    command_for_method(
                        args=args,
                        method=TRAINABLE_METHODS[0],
                        epochs=args.epochs,
                        train_subset=1.0,
                        seed=seed,
                    )
                )

    if args.phase in {"reduced", "all"}:
        for subset in args.train_subsets:
            for method in trainable_methods:
                commands.append(
                    command_for_method(
                        args=args,
                        method=method,
                        epochs=args.epochs,
                        train_subset=subset,
                    )
                )

    if args.phase in {"label-noise", "all"}:
        for noise_fraction in args.label_noise_fractions:
            for method in trainable_methods:
                commands.append(
                    command_for_method(
                        args=args,
                        method=method,
                        epochs=args.epochs,
                        train_subset=1.0,
                        label_noise_fraction=noise_fraction,
                    )
                )

    if args.phase in {"corruption", "corruption-severity", "all"}:
        if args.checkpoint_manifest is None:
            msg = (
                "--checkpoint-manifest is required for corruption, "
                "corruption-severity, or all phases"
            )
            raise SystemExit(msg)
        checkpoints = read_checkpoint_manifest(args.checkpoint_manifest)
        for method in trainable_methods:
            checkpoint = checkpoints.get(method.name)
            if checkpoint is None:
                continue
            if args.phase == "corruption-severity":
                dataset_id = "hard/pathmnist"
                data_dir_id = args.pathmnist_dir
                severities = tuple(args.corruption_severities)
            else:
                dataset_id = "hard/pathmnist-c"
                data_dir_id = args.pathmnist_c_dir
                severities = (1,)
            commands.append(
                command_for_method(
                    args=args,
                    method=method,
                    epochs=0,
                    train_subset=1.0,
                    dataset_id=dataset_id,
                    data_dir_id=data_dir_id,
                    discard_ood=False,
                    severities=severities,
                    weight_paths=checkpoint,
                )
            )

    if args.phase in {"posthoc", "all"}:
        if args.checkpoint_manifest is None:
            msg = "--checkpoint-manifest is required for posthoc or all phases"
            raise SystemExit(msg)
        checkpoints = read_checkpoint_manifest(args.checkpoint_manifest)
        for method in POSTHOC_METHODS:
            checkpoint = checkpoints.get(method.checkpoint_key or "")
            if checkpoint is None:
                continue
            commands.append(
                command_for_method(
                    args=args,
                    method=method,
                    epochs=0,
                    train_subset=1.0,
                    weight_paths=checkpoint,
                )
            )

    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "train",
            "reduced",
            "label-noise",
            "corruption",
            "corruption-severity",
            "posthoc",
            "all",
        ),
        default="train",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--python", default="${PYTHON_BIN:-python}")
    parser.add_argument(
        "--pathmnist-dir",
        default="${DATA_ROOT:?set DATA_ROOT}/pathmnist_cache/pathmnist",
    )
    parser.add_argument(
        "--pathmnist-c-dir",
        default="${DATA_ROOT:?set DATA_ROOT}/pathmnist_c/pathmnist",
    )
    parser.add_argument("--methods", help="Comma-separated trainable method names")
    parser.add_argument("--model-name", default="timm/resnet_50")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-seeds", type=int, nargs="*", default=(43, 44, 45, 46))
    parser.add_argument(
        "--skip-ensemble-seeds",
        action="store_true",
        help="Do not append CE ensemble member seeds in the train phase.",
    )
    parser.add_argument("--train-subsets", type=float, nargs="*", default=(0.5, 0.1))
    parser.add_argument(
        "--label-noise-fractions",
        type=float,
        nargs="*",
        default=(0.1, 0.2, 0.4),
        help="Training-label noise fractions used by the label-noise phase.",
    )
    parser.add_argument(
        "--label-noise-seed",
        type=int,
        default=1042,
        help="Seed used to select and flip noisy training labels.",
    )
    parser.add_argument(
        "--corruption-severities",
        type=int,
        nargs="*",
        default=(1, 2, 3, 4, 5),
        help=(
            "Severity levels for the synthetic/on-the-fly PathMNIST corruption "
            "axis. This is used only by phase=corruption-severity."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--lr-base", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--mean", default="0.485,0.456,0.406")
    parser.add_argument("--std", default="0.229,0.224,0.225")
    parser.add_argument("--storage-device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-eval-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = generate_commands(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(commands) + ("\n" if commands else ""))
    print(f"Wrote {len(commands)} commands to {args.output}")


if __name__ == "__main__":
    main()
