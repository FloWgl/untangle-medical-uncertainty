#!/usr/bin/env python3
"""Compare IT and estimated Bregman decomposition results across experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "decomposition_formula_comparison.csv"
EPS = 1e-6
NO_MEASURED_EU = {
    "ce-baseline", "correctness-prediction", "deep-correctness-prediction",
    "loss-prediction", "deep-loss-prediction", "ddu", "duq",
    "temperature-scaling", "mahalanobis",
}


def rho(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def direction(x):
    return np.where(x > EPS, 1, np.where(x < -EPS, -1, 0))


def intervention_row(name, data, level, intended):
    subset = data[np.isclose(data["level"], level)]
    subset = subset[~subset.method.isin(NO_MEASURED_EU)]
    wide = subset.pivot(index="method", columns="decomposition", values=["delta_au", "delta_eu"])
    au = wide["delta_au"]["information-theoretic"].to_numpy()
    it = wide["delta_eu"]["information-theoretic"].to_numpy()
    br = wide["delta_eu"]["bregman-estimated"].to_numpy()
    if intended == "EU":
        it_verdict, br_verdict = it > au, br > au
    else:
        it_verdict, br_verdict = au > it, au > br
    return {
        "probe": name,
        "n": len(wide),
        "spearman_it_vs_bregman": rho(it, br),
        "direction_agreement": np.mean(direction(it) == direction(br)),
        "probe_verdict_agreement": np.mean(it_verdict == br_verdict),
        "median_abs_it": np.median(np.abs(it)),
        "median_abs_bregman": np.median(np.abs(br)),
    }


def mhist_rows():
    source = ROOT / "docs/results/mhist/disentanglement/mhist_disentanglement_detailed.csv"
    data = pd.read_csv(source)
    means = data.pivot(index=["method", "train_fraction"], columns="decomposition", values=["mean_aleatoric", "mean_epistemic"])
    full = means.xs(1.0, level="train_fraction")
    low = means.xs(0.1, level="train_fraction")
    common = full.index.intersection(low.index)
    low_rows = data[np.isclose(data.train_fraction, 0.1)]
    nondegenerate = low_rows.groupby("method").apply(
        lambda group: bool(
            (group["std_epistemic"] >= EPS).all()
        ),
        include_groups=False,
    )
    common = common.intersection(nondegenerate[nondegenerate].index)
    au = low.loc[common, ("mean_aleatoric", "information-theoretic")].to_numpy() - full.loc[common, ("mean_aleatoric", "information-theoretic")].to_numpy()
    it = low.loc[common, ("mean_epistemic", "information-theoretic")].to_numpy() - full.loc[common, ("mean_epistemic", "information-theoretic")].to_numpy()
    br = low.loc[common, ("mean_epistemic", "bregman-estimated")].to_numpy() - full.loc[common, ("mean_epistemic", "bregman-estimated")].to_numpy()
    scarcity = {
        "probe": "MHIST scarcity (100% to 10%)",
        "n": len(common),
        "spearman_it_vs_bregman": rho(it, br),
        "direction_agreement": np.mean(direction(it) == direction(br)),
        "probe_verdict_agreement": np.mean((it > au) == (br > au)),
        "median_abs_it": np.median(np.abs(it)),
        "median_abs_bregman": np.median(np.abs(br)),
    }
    full_data = data[np.isclose(data.train_fraction, 1.0)].pivot(
        index="method", columns="decomposition",
        values=["spearman_aleatoric_label_entropy", "spearman_epistemic_label_entropy"])
    rater_au = full_data[("spearman_aleatoric_label_entropy", "information-theoretic")].to_numpy()
    rater_it = full_data[("spearman_epistemic_label_entropy", "information-theoretic")].to_numpy()
    rater_br = full_data[("spearman_epistemic_label_entropy", "bregman-estimated")].to_numpy()
    valid = np.isfinite(rater_au) & np.isfinite(rater_it) & np.isfinite(rater_br)
    rater = {
        "probe": "MHIST rater disagreement",
        "n": int(valid.sum()),
        "spearman_it_vs_bregman": rho(rater_it[valid], rater_br[valid]),
        "direction_agreement": np.mean(direction(rater_it[valid]) == direction(rater_br[valid])),
        "probe_verdict_agreement": np.mean((rater_au[valid] > rater_it[valid]) == (rater_au[valid] > rater_br[valid])),
        "median_abs_it": np.median(np.abs(rater_it[valid])),
        "median_abs_bregman": np.median(np.abs(rater_br[valid])),
    }
    return scarcity, rater


def cifar_scarcity_row():
    data = pd.read_csv(ROOT / "docs/results/cifar10/cifar10_scarcity_decomposition_means.csv")
    data = data[~data.method.isin(NO_MEASURED_EU)]
    full = data[np.isclose(data.train_fraction, 1.0)].set_index("queue_run")
    low = data[np.isclose(data.train_fraction, 0.1)].set_index("queue_run")
    common = full.index.intersection(low.index)
    au = low.loc[common, "it_aleatoric_mean"].to_numpy() - full.loc[common, "it_aleatoric_mean"].to_numpy()
    it = low.loc[common, "it_epistemic_mean"].to_numpy() - full.loc[common, "it_epistemic_mean"].to_numpy()
    br = low.loc[common, "bregman_epistemic_mean"].to_numpy() - full.loc[common, "bregman_epistemic_mean"].to_numpy()
    return {
        "probe": "CIFAR-10 scarcity (100% to 10%)",
        "n": len(common),
        "spearman_it_vs_bregman": rho(it, br),
        "direction_agreement": np.mean(direction(it) == direction(br)),
        "probe_verdict_agreement": np.mean((it > au) == (br > au)),
        "median_abs_it": np.median(np.abs(it)),
        "median_abs_bregman": np.median(np.abs(br)),
    }


def ood_rows():
    path = pd.read_csv(ROOT / "docs/results/pathmnist/pathmnist_c_severity_axis_summary.csv")
    path = path[~path.method.isin(NO_MEASURED_EU)]
    path_it = path.it_eu_ood_auroc_mean.to_numpy()
    path_br = path.bregman_eu_ood_auroc_mean.to_numpy()
    path_row = {
        "probe": "PathMNIST-C EU OOD AUROC",
        "n": len(path),
        "spearman_it_vs_bregman": rho(path_it, path_br),
        "direction_agreement": np.mean(direction(path_it - 0.5) == direction(path_br - 0.5)),
        "probe_verdict_agreement": np.mean((path_it > 0.5) == (path_br > 0.5)),
        "median_abs_it": np.median(np.abs(path_it - 0.5)),
        "median_abs_bregman": np.median(np.abs(path_br - 0.5)),
    }
    it = pd.read_csv(ROOT / "docs/results/cifar10/cifar10c_ood_auroc_summary.csv")
    br = pd.read_csv(ROOT / "docs/results/cifar10/cifar10c_bregman_ood_auroc_summary.csv")
    merged = it.merge(br, on=["phase", "train_fraction", "method", "queue_run"])
    merged = merged[~merged.method.isin(NO_MEASURED_EU)]
    cit = merged.all_severities_it_epistemic_ood_auroc_mean.to_numpy()
    cbr = merged.all_bregman_eu_ood_auroc_mean.to_numpy()
    cifar_row = {
        "probe": "CIFAR-10C EU OOD AUROC",
        "n": len(merged),
        "spearman_it_vs_bregman": rho(cit, cbr),
        "direction_agreement": np.mean(direction(cit - 0.5) == direction(cbr - 0.5)),
        "probe_verdict_agreement": np.mean((cit > 0.5) == (cbr > 0.5)),
        "median_abs_it": np.median(np.abs(cit - 0.5)),
        "median_abs_bregman": np.median(np.abs(cbr - 0.5)),
    }
    return path_row, cifar_row


def main():
    scarcity = pd.read_csv(ROOT / "docs/results/pathmnist/pathmnist_scarcity_deltas_all_decompositions_2026-09-17.csv")
    noise = pd.read_csv(ROOT / "docs/results/pathmnist/pathmnist_label_noise_deltas_all_decompositions_2026-09-17.csv")
    rows = [
        intervention_row("PathMNIST scarcity (100% to 10%)", scarcity, 0.1, "EU"),
        intervention_row("PathMNIST label noise (40%)", noise, 0.4, "AU"),
        *mhist_rows(),
        cifar_scarcity_row(),
        *ood_rows(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.6f")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
