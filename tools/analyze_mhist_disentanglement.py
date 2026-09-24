#!/usr/bin/env python3
"""Analyze MHIST uncertainty specialization against votes and data scarcity."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


PHASE_TO_FRACTION = {
    "full-data": 1.0,
    "scarce-50pct": 0.5,
    "scarce-10pct": 0.1,
}

DEGENERATE_STD_THRESHOLD = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("docs/results/mhist/mhist_training_results.csv"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/MHIST/annotations.csv"),
    )
    parser.add_argument(
        "--posthoc-status",
        type=Path,
        action="append",
        default=[],
        help="Optional completed MHIST post-hoc status.tsv. Can be passed more than once.",
    )
    parser.add_argument(
        "--extra-status",
        type=Path,
        action="append",
        default=[],
        help="Optional completed MHIST extra-methods status.tsv. Can be passed more than once.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/results/mhist/disentanglement"),
    )
    return parser.parse_args()


def binary_vote_entropy(num_positive_votes: int, num_raters: int = 7) -> float:
    probability = num_positive_votes / num_raters
    if probability in {0.0, 1.0}:
        return 0.0
    entropy = -probability * math.log(probability) - (
        1 - probability
    ) * math.log(1 - probability)
    return entropy / math.log(2)


def load_rater_targets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    entropies = []
    disagreements = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Partition"].strip().lower() != "test":
                continue
            votes = int(row["Number of Annotators who Selected SSA (Out of 7)"])
            entropies.append(binary_vote_entropy(votes))
            disagreements.append(min(votes, 7 - votes) / 3)
    return np.asarray(entropies), np.asarray(disagreements)


def output_dir_from_log(path: Path) -> Path:
    text = path.read_text(errors="replace")
    matches = re.findall(r"Output directory is ([^.\n]+)", text)
    if not matches:
        raise ValueError(f"No output directory recorded in {path}")
    return Path(matches[-1])


def preferred_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    exact_phases = {
        row["phase"]
        for row in rows
        if row["method"] == "correctness-prediction"
        and row["parameter_source"] == "exact-cdc24149"
    }
    return [
        row
        for row in rows
        if not (
            row["method"] == "correctness-prediction"
            and row["phase"] in exact_phases
            and row["parameter_source"] != "exact-cdc24149"
        )
    ]


def posthoc_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["status"] != "done":
                    continue
                if row["stage"] not in {
                    "deep-ensemble",
                    "fast-deep-ensemble",
                    "temperature-scaling",
                    "swag",
                    "laplace",
                }:
                    continue
                rows.append({
                    "phase": row["phase"],
                    "method": row["stage"],
                    "parameter_source": "posthoc",
                    "log": row["log"],
                })
    return rows


def extra_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["status"] != "done":
                    continue
                rows.append({
                    "phase": row["phase"],
                    "method": row["method"],
                    "parameter_source": "transferred-cifar-config-extra",
                    "log": row["log"],
                })
    return rows


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if (
        np.std(x) < DEGENERATE_STD_THRESHOLD
        or np.std(y) < DEGENERATE_STD_THRESHOLD
    ):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranked = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranked[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranked


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if (
        np.std(x) < DEGENERATE_STD_THRESHOLD
        or np.std(y) < DEGENERATE_STD_THRESHOLD
    ):
        return float("nan")
    return pearson(ranks(x), ranks(y))


def load_decompositions(output_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    prefix = output_dir / "id_test_soft_mhist"
    paths = {
        "information-theoretic": Path(f"{prefix}_it_au_eu.pt"),
        "bregman-estimated": Path(f"{prefix}_bregman_eu_au_hat.pt"),
    }
    decompositions = {}
    for name, path in paths.items():
        if not path.exists():
            continue
        first, second = torch.load(path, map_location="cpu", weights_only=True)
        if name == "information-theoretic":
            aleatoric, epistemic = first, second
        else:
            epistemic, aleatoric = first, second
        decompositions[name] = (
            aleatoric.detach().float().numpy(),
            epistemic.detach().float().numpy(),
        )
    return decompositions


def finite(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.10g}"


def finite_pair(
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    label_entropy: np.ndarray,
    vote_disagreement: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    mask = (
        np.isfinite(aleatoric)
        & np.isfinite(epistemic)
        & np.isfinite(label_entropy)
        & np.isfinite(vote_disagreement)
    )
    return (
        aleatoric[mask],
        epistemic[mask],
        label_entropy[mask],
        vote_disagreement[mask],
        int(mask.sum()),
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    label_entropy, vote_disagreement = load_rater_targets(args.annotations)

    rows = preferred_rows(args.results_csv)
    rows.extend(posthoc_rows(args.posthoc_status))
    rows.extend(extra_rows(args.extra_status))
    deduped_rows = {}
    for row in rows:
        deduped_rows[
            (row["phase"], row["method"], row["parameter_source"])
        ] = row
    rows = list(deduped_rows.values())
    detailed = []

    for row in rows:
        phase = row["phase"]
        if phase not in PHASE_TO_FRACTION:
            continue
        log_path = Path(row["log"])
        output_dir = output_dir_from_log(log_path)
        for decomposition, (aleatoric, epistemic) in load_decompositions(
            output_dir
        ).items():
            if len(aleatoric) != len(label_entropy):
                raise ValueError(
                    f"{output_dir}: {len(aleatoric)} predictions for "
                    f"{len(label_entropy)} MHIST test targets"
                )
            (
                finite_aleatoric,
                finite_epistemic,
                finite_label_entropy,
                finite_vote_disagreement,
                num_finite_samples,
            ) = finite_pair(aleatoric, epistemic, label_entropy, vote_disagreement)
            detailed.append({
                "phase": phase,
                "train_fraction": PHASE_TO_FRACTION[phase],
                "method": row["method"],
                "parameter_source": row["parameter_source"],
                "decomposition": decomposition,
                "num_samples": len(aleatoric),
                "num_finite_samples": num_finite_samples,
                "mean_aleatoric": float(finite_aleatoric.mean()),
                "mean_epistemic": float(finite_epistemic.mean()),
                "std_aleatoric": float(finite_aleatoric.std()),
                "std_epistemic": float(finite_epistemic.std()),
                "pearson_aleatoric_label_entropy": pearson(
                    finite_aleatoric, finite_label_entropy
                ),
                "spearman_aleatoric_label_entropy": spearman(
                    finite_aleatoric, finite_label_entropy
                ),
                "pearson_epistemic_label_entropy": pearson(
                    finite_epistemic, finite_label_entropy
                ),
                "spearman_epistemic_label_entropy": spearman(
                    finite_epistemic, finite_label_entropy
                ),
                "spearman_aleatoric_vote_disagreement": spearman(
                    finite_aleatoric, finite_vote_disagreement
                ),
                "spearman_epistemic_vote_disagreement": spearman(
                    finite_epistemic, finite_vote_disagreement
                ),
                "output_dir": str(output_dir),
                "log": str(log_path),
            })

    detailed_path = args.out_dir / "mhist_disentanglement_detailed.csv"
    fieldnames = list(detailed[0]) if detailed else []
    with detailed_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in detailed:
            writer.writerow({
                key: finite(value) if isinstance(value, float) else value
                for key, value in row.items()
            })

    grouped: dict[tuple[str, str, str], dict[float, dict[str, object]]] = (
        defaultdict(dict)
    )
    for row in detailed:
        key = (
            str(row["method"]),
            str(row["parameter_source"]),
            str(row["decomposition"]),
        )
        grouped[key][float(row["train_fraction"])] = row

    summary = []
    for (method, source, decomposition), fractions in sorted(grouped.items()):
        full = fractions.get(1.0)
        if full is None:
            continue
        for fraction in (0.5, 0.1):
            scarce = fractions.get(fraction)
            if scarce is None:
                continue
            aleatoric_shift = float(scarce["mean_aleatoric"]) - float(
                full["mean_aleatoric"]
            )
            epistemic_shift = float(scarce["mean_epistemic"]) - float(
                full["mean_epistemic"]
            )
            summary.append({
                "method": method,
                "parameter_source": source,
                "decomposition": decomposition,
                "scarce_fraction": fraction,
                "full_mean_aleatoric": full["mean_aleatoric"],
                "scarce_mean_aleatoric": scarce["mean_aleatoric"],
                "aleatoric_scarcity_shift": aleatoric_shift,
                "full_mean_epistemic": full["mean_epistemic"],
                "scarce_mean_epistemic": scarce["mean_epistemic"],
                "epistemic_scarcity_shift": epistemic_shift,
                "full_spearman_aleatoric_label_entropy": full[
                    "spearman_aleatoric_label_entropy"
                ],
                "scarce_spearman_aleatoric_label_entropy": scarce[
                    "spearman_aleatoric_label_entropy"
                ],
                "full_spearman_epistemic_label_entropy": full[
                    "spearman_epistemic_label_entropy"
                ],
                "scarce_spearman_epistemic_label_entropy": scarce[
                    "spearman_epistemic_label_entropy"
                ],
                "scarcity_specialization_margin": epistemic_shift
                - abs(aleatoric_shift),
            })

    summary_path = args.out_dir / "mhist_disentanglement_scarcity.csv"
    summary_fields = list(summary[0]) if summary else []
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({
                key: finite(value) if isinstance(value, float) else value
                for key, value in row.items()
            })

    metadata_path = args.out_dir / "mhist_analysis_metadata.md"
    metadata_path.write_text(
        "# MHIST disentanglement analysis\n\n"
        f"- Test samples: {len(label_entropy)}\n"
        "- Aleatoric target: normalized binary entropy of seven pathologist votes.\n"
        "- Alternate ambiguity target: minority-vote fraction normalized to [0, 1].\n"
        "- Epistemic intervention: change from 100% training data to 50% and 10%.\n"
        "- Primary decomposition: expected entropy (aleatoric) and mutual "
        "information/Jensen-Shannon divergence (epistemic).\n"
        "- Cross-sensitivity: epistemic correlation with rater entropy and "
        "aleatoric drift under scarcity.\n"
        "- Undefined correlations are left blank when a component is constant.\n"
        "- Non-finite decomposition entries are excluded pairwise and counted in "
        "num_finite_samples.\n"
        "- Extra-method status files can be included with --extra-status once "
        "the supplemental MHIST queue finishes.\n"
    )
    print(f"Wrote {detailed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
