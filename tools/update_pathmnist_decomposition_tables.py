#!/usr/bin/env python3
"""Refresh PathMNIST decomposition tables in the thesis from summary CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


METHOD_LABELS = {
    "ce-baseline": "CE baseline",
    "correctness-prediction": "Correctness prediction",
    "deep-correctness-prediction": "Deep correctness prediction",
    "loss-prediction": "Loss prediction",
    "deep-loss-prediction": "Deep loss prediction",
    "mc-dropout": "MC Dropout",
    "edl": "EDL",
    "sngp": "SNGP",
    "het": "HET",
    "het-xl": "HET-XL",
    "hetclassnn": "HetClassNN",
    "shallow-ensemble": "Shallow Ensemble",
    "duq": "DUQ",
    "postnet": "PostNet",
}

DECOMP_LABELS = {
    "information-theoretic": "IT",
    "bregman-estimated": "Bregman",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: str, digits: int = 4, signed: bool = True) -> str:
    number = float(value)
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.{digits}f}"


def scarcity_table(rows: list[dict[str, str]]) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["method"],
            row["decomposition"],
            float(row["train_subset"]),
        ),
    )
    body = "\n".join(
        " & ".join(
            [
                METHOD_LABELS.get(row["method"], row["method"]),
                DECOMP_LABELS.get(row["decomposition"], row["decomposition"]),
                "50\\%" if row["train_subset"] == "0.5" else "10\\%",
                fnum(row["delta_au"]),
                fnum(row["delta_eu"]),
            ]
        )
        + r" \\"
        for row in ordered
    )
    return rf"""\begin{{table}}[htbp]
\centering
\scriptsize
\caption[PathMNIST data-scarcity shifts]{{PathMNIST data-scarcity AU/EU shifts from extracted decomposition tensors.}}
\label{{tab:pathmnist-scarcity-results}}
\begin{{tabular}}{{@{{}}lllrr@{{}}}}
\toprule
Method & Decomp. & Split & $\Delta AU$ & $\Delta EU$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def corruption_table(rows: list[dict[str, str]]) -> tuple[str, str]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["decomposition"]].append(row)

    table_rows = []
    interpretations = []
    for decomposition, decomp_rows in sorted(grouped.items()):
        n = len(decomp_rows)
        clean_au = sum(float(row["clean_au_mean"]) for row in decomp_rows) / n
        corrupt_au = sum(float(row["mean_corrupted_au"]) for row in decomp_rows) / n
        clean_eu = sum(float(row["clean_eu_mean"]) for row in decomp_rows) / n
        corrupt_eu = sum(float(row["mean_corrupted_eu"]) for row in decomp_rows) / n
        delta_au = corrupt_au - clean_au
        delta_eu = corrupt_eu - clean_eu
        label = DECOMP_LABELS.get(decomposition, decomposition)
        table_rows.append(
            f"{label} & {n} & {clean_au:.4f} & {corrupt_au:.4f} & "
            f"{clean_eu:.4f} & {corrupt_eu:.4f} \\\\"
        )
        interpretations.append(
            f"{label}: $\\Delta AU={delta_au:+.4f}$ and "
            f"$\\Delta EU={delta_eu:+.4f}$"
        )

    table = rf"""\begin{{table}}[htbp]
\centering
\scriptsize
\caption[Aggregate PathMNIST-C corruption shift]{{Aggregate shift for the fixed official PathMNIST-C corruption set.}}
\label{{tab:pathmnist-corruption-shift}}
\begin{{tabular}}{{@{{}}llrrrr@{{}}}}
\toprule
Decomp. & Checkpoints & Clean AU & Corrupted AU & Clean EU & Corrupted EU \\
\midrule
{chr(10).join(table_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}"""
    text = (
        "Across the extracted PathMNIST-C checkpoint set, the aggregate "
        "decomposition shifts remain small ("
        + "; ".join(interpretations)
        + "). This table therefore does not support a strong aggregate "
        "decomposition-shift conclusion. It is a tensor-level corruption "
        "summary; the method-level OOD ranking in "
        "Table~\\ref{tab:pathmnist-c-method-ood} gives the more direct "
        "distribution-shift readout."
    )
    return table, text


def replace_table(tex: str, label: str, replacement: str) -> str:
    pattern = (
        r"\\begin\{table\}\[htbp\]\n"
        r".*?"
        + re.escape(rf"\label{{{label}}}")
        + r".*?"
        r"\\end\{table\}"
    )
    new, count = re.subn(pattern, replacement, tex, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"Could not replace table {label}")
    return new


def replace_corruption_text(tex: str, replacement: str) -> str:
    pattern = (
        r"Across the extracted PathMNIST-C checkpoint set,.*?"
        r"readout\."
    )
    new, count = re.subn(pattern, replacement, tex, count=1, flags=re.S)
    if count != 1:
        raise SystemExit("Could not replace PathMNIST-C corruption interpretation text")
    return new


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    thesis_path = repo_root / "docs/thesis/sections/pathmnist_results.tex"
    out_dir = repo_root / "docs/results/pathmnist"
    scarcity_rows = read_rows(out_dir / "pathmnist_scarcity_margins_all_decompositions_2026-09-16.csv")
    corruption_rows = read_rows(out_dir / "pathmnist_c_corruption_all_decompositions_2026-09-16.csv")

    changed = False
    tex = thesis_path.read_text()
    if scarcity_rows:
        tex = replace_table(tex, "tab:pathmnist-scarcity-results", scarcity_table(scarcity_rows))
        changed = True
    if corruption_rows:
        table, text = corruption_table(corruption_rows)
        tex = replace_table(tex, "tab:pathmnist-corruption-shift", table)
        tex = replace_corruption_text(tex, text)
        changed = True

    if not changed:
        print("No non-empty PathMNIST decomposition CSVs found; thesis unchanged.")
        return
    thesis_path.write_text(tex)
    print(f"Updated {thesis_path}")


if __name__ == "__main__":
    main()
