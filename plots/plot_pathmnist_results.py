"""Plot PathMNIST result tensors saved by `train.py`/`validate.py`."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def describe_file(path: Path) -> tuple[str, str]:
    name = path.name
    if name.endswith("it_au_eu.pt"):
        return "AU (expected entropies)", "EU (Jensen-Shannon divergences)"
    if name.endswith("bregman_eu_au_hat.pt"):
        return "EU^b (expected divergences)", "AU^b (expected entropies)"
    if name.endswith("kendall_gal_au_eu_prob.pt"):
        return "AU (expected entropies)", "EU (variance of probs)"
    if name.endswith("kendall_gal_au_eu_logit.pt"):
        return "AU (expected entropies)", "EU (variance of logits)"
    if name.endswith("kendall_gal_au_eu_internal_prob.pt"):
        return "AU (expected entropies)", "EU (internal variance of probs)"
    if name.endswith("kendall_gal_au_eu_internal_logit.pt"):
        return "AU (expected entropies)", "EU (internal variance of logits)"
    if name.endswith("ddu_au_eu.pt"):
        return "AU (expected entropies)", "EU (GMM negative log density)"
    if name.endswith("mahalanobis_au_eu.pt"):
        return "AU (expected entropies)", "EU (Mahalanobis values)"
    return "tensor 1", "tensor 2"


def load_pair(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, tuple) or len(data) != 2:
        raise ValueError(f"Expected a 2-tuple in {path}, got {type(data).__name__}")
    first, second = data
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError(f"Expected tensors in {path}")
    return first.flatten().float(), second.flatten().float()


def plot_pair(path: Path, output_dir: Path | None, max_points: int | None) -> Path:
    x_name, y_name = describe_file(path)
    x, y = load_pair(path)

    if max_points is not None and len(x) > max_points:
        indices = torch.randperm(len(x))[:max_points]
        x = x[indices]
        y = y[indices]

    x_np = x.numpy()
    y_np = y.numpy()

    pearson = pearsonr(x_np, y_np).statistic
    spearman = spearmanr(x_np, y_np).statistic

    output_dir = output_dir or path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_scatter.png"

    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.scatter(x_np, y_np, s=6, alpha=0.35, edgecolors="none")
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(
        f"{path.stem}\nPearson={pearson:.3f}, Spearman={spearman:.3f}, n={len(x_np)}"
    )
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"Saved {output_path}")
    print(f"Pearson={pearson:.6f}")
    print(f"Spearman={spearman:.6f}")
    print(f"n={len(x_np)}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PathMNIST tensor pairs saved by untangle runs"
    )
    parser.add_argument("files", nargs="+", type=Path, help="One or more .pt files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots in; defaults to each file's folder",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional random subsample size for dense scatter plots",
    )
    args = parser.parse_args()

    for file_path in args.files:
        plot_pair(file_path, args.output_dir, args.max_points)


if __name__ == "__main__":
    main()