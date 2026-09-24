# Untangle Paper Hyperparameters - Extraction Summary

## Overview

Successfully extracted hyperparameters from **18 ImageNet methods** and **12 CIFAR-10 methods** by fetching the official W&B sweep configurations and consolidating them.

## Sources

| Resource | Link | Purpose |
|---|---|---|
| **Paper Repository** | https://github.com/bmucsanyi/untangle | Official code + sweep IDs in README |
| **Paper (arXiv)** | https://arxiv.org/abs/2402.19460 | Original paper |
| **W&B Project** | https://wandb.ai/bmucsanyi/untangle/sweeps | Official sweep configurations |
| **Extraction Method** | Browser navigation + Playwright | Visited each sweep's overview page and extracted `Sweep configuration` section |

## Extracted Hyperparameters by Dataset

### ImageNet (18 methods extracted)

See the consolidated table in `docs/EXPERIMENT_SETTINGS_CONSOLIDATED.md` for the full per-method list. Extracted ImageNet methods include:

- CE Baseline (5rugaun5)
- Correctness Prediction (v2yi16cs)
- HetClassNN (qbzsnbl8)
- HET-XL (px1j9kqg)
- HET (jzl73q5k)
- Loss Prediction (xqseuogg)
- MC Dropout (kvdgsjmc)
- EDL (08wija9l)
- Shallow Ensemble (adp0fyi8)
- DDU (ixrv6bih)
- SNGP / GP variants (1yzdpvwt, xibtpo9s)
- PostNet (yfsrcusv)
- Deep Ensemble (54kpysjy, final)
- Laplace (42thx27s, final)
- Mahalanobis (8a3palks, final)
- Temperature Scaling (jfnn98e3, final)
- SWAG (o04c996o)


### CIFAR-10 (12 methods extracted)

See the consolidated table in `docs/EXPERIMENT_SETTINGS_CONSOLIDATED.md` for the full per-method list. Extracted CIFAR-10 methods include:

- CE Baseline (cedfgnqz)
- Correctness Prediction (azcfycns)
- MC Dropout (k4xc00mf)
- HetClassNN (pb3a6oru)
- HET-XL (olapo0kg)
- HET (e6rpfaue)
- Loss Prediction (9fb4ansn)
- Shallow Ensemble (k6scvi1c)
- DDU (5w1yjrmj)
- GP (1692idyk)
- SNGP (jkzjb5vz)
- PostNet (8ugac5sn)
- EDL (5lhhpt1y)

## Additional Methods Not Yet Extracted

All sweeps referenced in the original docs have now been fetched and consolidated. The raw YAMLs are stored in `docs/wandb_sweeps/` and the canonical consolidated view is `docs/EXPERIMENT_SETTINGS_CONSOLIDATED.md`.

## Documentation Files Created

### Full Hyperparameter Specifications

- **Canonical consolidated file:** [EXPERIMENT_SETTINGS_CONSOLIDATED.md](EXPERIMENT_SETTINGS_CONSOLIDATED.md) — contains full per-method tables, key hyperparameters and W&B sweep links for all datasets.
- **Raw sweep backups:** `docs/wandb_sweeps/` — original YAMLs for every fetched W&B sweep are preserved.

> Note: The dataset-specific full files (`IMAGENET_EXPERIMENT_SETTINGS_FULL.md` and `CIFAR10_EXPERIMENT_SETTINGS_FULL.md`) have been removed to avoid duplication. The consolidated file is the single source of truth.

## Key Findings

### Hyperparameter Patterns

1. **ImageNet**
   - All methods use resnet_50 or resnet_50-based variant
   - Cosine annealing scheduler with 5-epoch warmup
   - AMP training with float16
   - Gradient accumulation of 16 or 32 steps
   - 50 epoch training

2. **CIFAR-10**
   - All methods use wide_resnet_c_preact_26_10
   - Multistep scheduler with milestone decay
   - No AMP (standard float32)
   - No gradient accumulation
   - 200 epoch training (4x longer than ImageNet)

3. **Method-Specific Patterns**
   - Uncertainty prediction methods (Correctness Prediction, Loss Prediction) use additional swept parameters (lambda_uncertainty_loss)
   - Deterministic + MC methods (HetClassNN, MC-Dropout) use MC sampling counts
   - EDL uses different, narrower learning rate ranges
   - HET variants use significantly higher batch accumulation

## Extraction Methodology

1. **Visited W&B sweep overview pages** in a browser
2. **Extracted the "Sweep configuration" YAML section** using Playwright
3. **Parsed parameter distributions and fixed values** from the YAML
4. **Compiled into structured documentation** with source links

This ensures accuracy by using official W&B configurations rather than reverse-engineering from paper text.

## Usage

To use these hyperparameters for reproduction:

```bash
# ImageNet CE Baseline example
python train.py \
  --method-name ce-baseline \
  --dataset hard/imagenet \
  --lr 0.001 \
  --epochs 50 \
  --batch-size 128 \
  --opt lamb \
  --weight-decay 0.01

# CIFAR-10 MC-Dropout example
python train.py \
  --method-name mc-dropout \
  --dataset hard/cifar10 \
  --lr 0.09 \
  --epochs 200 \
  --batch-size 128 \
  --opt nesterov \
  --dropout-probability 0.1
```

For swept parameters, test multiple values within the documented ranges and select based on validation performance.
