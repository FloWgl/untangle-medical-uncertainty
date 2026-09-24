# Consolidated Experiment Settings (ImageNet, CIFAR-10, PathMNIST)

This document consolidates extracted hyperparameters for ImageNet and CIFAR-10 experiments from the project's W&B sweep configurations. Parameters were verified against the saved sweep YAMLs in `docs/wandb_sweeps/`.

For the end-to-end execution protocol, use:

- `REPRODUCIBILITY.md`
- `docs/experiment_matrix_cifar10_pathmnist.csv`
- `tools/hpc/untangle_job_array.sbatch`

Layout:
- Sections by dataset (ImageNet, CIFAR-10)
- Methods grouped by type where helpful (Baseline, Ensembles, GP/SNGP, Uncertainty prediction, Calibration / Post-hoc, Other)
- For each method: `Sweep ID`, key hyperparameters, and W&B link

---

## ImageNet (hard/imagenet)

Common settings:
- Image size: 224×224, crop-pct: 0.875
- Dataset id for soft labels: `soft/imagenet`
- Scheduler: cosine with warmup (warmup_lr=0.0001, warmup_epochs=5) unless noted
- AMP: enabled (float16) for most ImageNet sweeps
- Accumulation: typically 16 (some methods use 32)
- Default seed: 42

### Baseline

| Method | Sweep ID | lr (range) | epochs | batch | opt | wd | Model | Key flags | W&B |
|---|---:|---:|---:|---:|---:|---:|---|---|
| CE Baseline | 5rugaun5 | log_uniform(0.0005–0.02) | 50 | 128 | lamb | 0.01 / 0.02 | resnet_50 | cosine; accum=16; AMP fp16 | https://wandb.ai/bmucsanyi/untangle/sweeps/5rugaun5 |

### Uncertainty prediction / Auxiliary heads

| Method | Sweep ID | lr | epochs | batch | opt | model | Key params | W&B |
|---|---:|---:|---:|---:|---:|---|---|---|
| Correctness Prediction | v2yi16cs | log_uniform(0.0005–0.02) | 50 | 128 | lamb | resnet_50 | lambda_unc: log(0.001–0.1); MLP 3×1024 | https://wandb.ai/bmucsanyi/untangle/sweeps/v2yi16cs |
| Loss Prediction | xqseuogg | log_uniform(0.0005–0.02) | 50 | 128 | lamb | resnet_50 | lambda_unc: log(0.001–0.1); MLP 3×1024 | https://wandb.ai/bmucsanyi/untangle/sweeps/xqseuogg |
| PostNet (flow-based post-hoc) | yfsrcusv | log_uniform(0.0005–0.02) | 50 | 128 | lamb | untangle/resnet_50 | latent-dim: 6,10; hidden: 512/1024; num-density-components=6; loss=UCE; uce-reg=1e-5 | https://wandb.ai/bmucsanyi/untangle/sweeps/yfsrcusv |

### Ensembles & Deterministic methods

| Method | Sweep ID | lr | epochs | batch | opt | notes | W&B |
|---|---:|---:|---:|---:|---:|---|---|
| Shallow Ensemble | adp0fyi8 | log_uniform(0.0005–0.02) | 50 | 128 | lamb | num-heads=10; accum=16; amp=true; pretrained=true | https://wandb.ai/bmucsanyi/untangle/sweeps/adp0fyi8 |
| Deep Ensemble (fast) | 54kpysjy | final-only | final-only | 128 | lamb | ensemble built from listed weight paths (see sweep) | https://wandb.ai/bmucsanyi/untangle/sweeps/54kpysjy |

### GP / SNGP / Bayesian-ish

| Method | Sweep ID | lr | epochs | batch | opt | gp / spectral params | W&B |
|---|---:|---:|---:|---:|---:|---|---|
| SNGP / GP variant | 1yzdpvwt | log_uniform(0.0005–0.02) | 50 | 128 | lamb | num-random-features=1024; gp-kernel-scale=1; num-mc-samples=1000; likelihood=gaussian | https://wandb.ai/bmucsanyi/untangle/sweeps/1yzdpvwt |
| SNGP (alt) | xibtpo9s | log_uniform(0.0005–0.02) | 50 | 128 | lamb | use-spectral-normalization=true; spectral-normalization-bound=6; num-random-features=1024 | https://wandb.ai/bmucsanyi/untangle/sweeps/xibtpo9s |
| DDU | ixrv6bih | log_uniform(0.0005–0.02) | 50 | 128 | lamb | use-spectral-normalization=true; spectral-normalization-bound=3; model-kwargs: downsample_type=avg_pad,act_layer=nn.LeakyReLU | https://wandb.ai/bmucsanyi/untangle/sweeps/ixrv6bih |

### Calibration / Post-hoc / Other

| Method | Sweep ID | lr | epochs | batch | notes | W&B |
|---|---:|---:|---:|---:|---|---|
| Temperature Scaling | jfnn98e3 | final-only | final-only | 128 | final model paths in sweep; use-temperature-scaling flag | https://wandb.ai/bmucsanyi/untangle/sweeps/jfnn98e3 |
| Laplace | 42thx27s | final-only | final-only | 128 | hessian-structure=kron; num-mc-samples=900 | https://wandb.ai/bmucsanyi/untangle/sweeps/42thx27s |
| Mahalanobis | 8a3palks | final-only | final-only | 64 | magnitude=0.001; module-name-regex=^(layer[1-4]|global_pool)$ | https://wandb.ai/bmucsanyi/untangle/sweeps/8a3palks |
| SWAG | o04c996o | lr=0.001 (fixed) | 10 | 128 | num-checkpoints-per-epoch=4; max-rank=20; use-low-rank-cov=true | https://wandb.ai/bmucsanyi/untangle/sweeps/o04c996o |

---

## CIFAR-10 (hard/cifar10)

Common settings:
- Image size: 32×32, crop-pct: 1.0
- Scheduler: multistep (milestones vary) with warmup_lr=0.005, warmup_epochs=1
- Optimizer: Nesterov SGD (momentum=0.9) for most CIFAR sweeps
- Typical epochs: 200 (some 250), batch size: 128
- No AMP; typically no gradient accumulation

### Baseline & Deterministic

| Method | Sweep ID | lr | epochs | batch | opt | wd | Model | W&B |
|---|---:|---:|---:|---:|---:|---:|---|---|
| CE Baseline (CIFAR) | cedfgnqz | uniform(0.05–0.14) | 200 | 128 | nesterov | log_uniform(1e-5–0.005) | wide_resnet_c_preact_26_10 | https://wandb.ai/bmucsanyi/untangle/sweeps/cedfgnqz |

### Uncertainty prediction / HET variants

| Method | Sweep ID | lr | epochs | batch | key params | W&B |
|---|---:|---:|---:|---:|---|---|
| HetClassNN | pb3a6oru | uniform(0.05–0.14) | 200 | 128 | dropout=0.1; MC=30; use-filterwise-dropout True/False; multistep sched | https://wandb.ai/bmucsanyi/untangle/sweeps/pb3a6oru |
| HET-XL | olapo0kg | uniform(0.05–0.14) | 200 | 128 | matrix-rank=6; MC=1000 | https://wandb.ai/bmucsanyi/untangle/sweeps/olapo0kg |
| HET | e6rpfaue | uniform(0.05–0.14) | 200 | 128 | matrix-rank swept; use-het flags | https://wandb.ai/bmucsanyi/untangle/sweeps/e6rpfaue |

### Ensembles & GP / SNGP / DDU

| Method | Sweep ID | lr | epochs | batch | key params | W&B |
|---|---:|---:|---:|---:|---|---|
| Shallow Ensemble | k6scvi1c | uniform(0.05–0.14) | 200 | 128 | ensemble heads; weight paths listed in sweep | https://wandb.ai/bmucsanyi/untangle/sweeps/k6scvi1c |
| DDU | 5w1yjrmj | uniform(0.05–0.14) | 250 | 128 | use-spectral-normalization=true; spectral-bound=3 | https://wandb.ai/bmucsanyi/untangle/sweeps/5w1yjrmj |
| GP (CIFAR) | 1692idyk | uniform(0.05–0.14) | 200 | 128 | GP-related params (random features, kernel) in sweep | https://wandb.ai/bmucsanyi/untangle/sweeps/1692idyk |
| SNGP | jkzjb5vz | uniform(0.05–0.14) | 200 | 128 | SNGP params present in sweep | https://wandb.ai/bmucsanyi/untangle/sweeps/jkzjb5vz |

### PostNet / EDL / Calibration

| Method | Sweep ID | lr | epochs | batch | key params | W&B |
|---|---:|---:|---:|---:|---|---|
| PostNet | 8ugac5sn | uniform(0.05–0.14) | 200 | 128 | flow/postnet specific params in sweep | https://wandb.ai/bmucsanyi/untangle/sweeps/8ugac5sn |
| EDL | 5lhhpt1y | uniform(0.05–0.14) | 200 | 128 | EDL-specific scaler / activation in sweep | https://wandb.ai/bmucsanyi/untangle/sweeps/5lhhpt1y |
| Loss Prediction | 9fb4ansn | uniform(0.05–0.14) | 200 | 128 | lambda-uncertainty-loss: log_uniform(0.01–0.1); mlp-depth=3; hidden=1024 | https://wandb.ai/bmucsanyi/untangle/sweeps/9fb4ansn |
| Laplace (final) | nnle8epz | final-only | final-only | 128 | hessian-structure=kron; num-mc-samples=1000; weight-paths provided | https://wandb.ai/bmucsanyi/untangle/sweeps/nnle8epz |
| Temperature Scaling (final) | lh04ospw | final-only | final-only | 128 | use-temperature-scaling=true; weight-paths provided | https://wandb.ai/bmucsanyi/untangle/sweeps/lh04ospw |
| Mahalanobis (final) | h0m0bybl | final-only | final-only | 128 | magnitude=0.001; module-name-regex=^(layer[1-3].0.act1|act)$; weight-paths provided | https://wandb.ai/bmucsanyi/untangle/sweeps/h0m0bybl |
| Deep Ensemble (final) | bn1hsbqz | final-only | final-only | 128 | ensemble weight-paths provided in sweep | https://wandb.ai/bmucsanyi/untangle/sweeps/bn1hsbqz |
| SWAG | zsiqsl6u | lr=0.01 | 40 | 128 | max-rank=20; num-checkpoints-per-epoch=1; use-low-rank-cov=true | https://wandb.ai/bmucsanyi/untangle/sweeps/zsiqsl6u |

---

## Verification notes
- All ImageNet and CIFAR sweep YAMLs referenced above were read from `docs/wandb_sweeps/` and used to fill the per-method key parameters shown above.
- For final-only sweeps (ensembles, Laplace, Mahalanobis, Temperature Scaling) the sweep stores `weight-paths` pointing to final checkpoints — reproduce by using those paths or the flags shown in the table.
- CIFAR-10 replication should be run before interpreting PathMNIST transfer results. The verification and full experiment commands are documented in `REPRODUCIBILITY.md`.

---

## PathMNIST (hard/pathmnist)

PathMNIST is not a published benchmark point in the original paper. It is the
medical transfer target in this repo. The fixed protocol is documented in
`docs/PATHMNIST_EXPERIMENT_SETTINGS.md` and summarized here.

Common settings:

- Dataset: `hard/pathmnist`
- Corruption/OOD dataset: `hard/pathmnist-c`
- Model: `timm/resnet_50`
- Image size: 224
- Optimizer: AdamW
- Weight decay: `2e-5`
- Base learning rate: `0.001`
- Effective learning rate with batch 128 and accumulation 16: `0.008`
- Epochs: 90
- Batch size: 128
- Accumulation steps: 16
- Seed: 42
- Normalization: ImageNet mean/std

PathMNIST-C note:

- The Zenodo NPZ files are fixed corruptions and do not encode CIFAR-C severity
  levels 1-5. Use `--severities 1` when evaluating NPZ PathMNIST-C files.

Transfer method roster:

| Method | Train/post-hoc | Key PathMNIST flags | Status note |
|---|---|---|---|
| CE baseline | train | `--method-name ce-baseline --loss cross-entropy` | Baseline for all post-hoc methods. |
| Correctness prediction | train | `--method-name correctness-prediction --loss correctness-prediction --lambda-uncertainty-loss 0.01` | Fixed auxiliary-head setting. |
| Deep correctness prediction | train | `--method-name deep-correctness-prediction --loss correctness-prediction --lambda-uncertainty-loss 0.01` | Previously completed after queued rerun; verify checkpoint manifest. |
| Loss prediction | train | `--method-name loss-prediction --loss loss-prediction --lambda-uncertainty-loss 0.01` | Existing protocol shows completed run. |
| Deep loss prediction | train | `--method-name deep-loss-prediction --loss loss-prediction --lambda-uncertainty-loss 0.01` | Existing protocol shows completed rerun. |
| EDL | train | `--method-name edl --loss edl --edl-start-epoch 0 --edl-scaler 1.0 --edl-activation exp --num-classes 9` | `--num-classes 9` is required. |
| SNGP | train | spectral normalization + GP flags from `docs/PATHMNIST_EXPERIMENT_SETTINGS.md` | Existing protocol shows completed rerun. |
| PostNet | train | `--method-name postnet --latent-dim 6 --num-hidden-features 256 --num-density-components 6 --uce-regularization-factor 1e-5` | Verify latest checkpoint/log. |
| HET / HET-XL / HetClassNN | train | HET flags from `docs/PATHMNIST_EXPERIMENT_SETTINGS.md` | Earlier runs hit OOM; reserve for larger GPU or reduced batch. |
| Shallow ensemble | train | `--method-name shallow-ensemble --num-heads 10` | Earlier runs hit OOM. |
| Deep ensemble | post-hoc | `--method-name deep-ensemble --weight-paths ...` | Needs multiple trained checkpoints. |
| Temperature scaling | post-hoc | `--method-name temperature-scaling --weight-paths ...` | Needs trained checkpoint. |
| Mahalanobis | post-hoc | `--method-name mahalanobis --weight-paths ... --magnitude 0.001` | Needs covariance/feature pass. |
| DDU | post-hoc/train-compatible | DDU spectral flags | Needs compatible checkpoint. |
| SWAG | post-hoc/fine-tune | `--method-name swag --use-low-rank-cov --max-rank 20` | Needs checkpoint collection. |
| Laplace | post-hoc | `--method-name laplace --hessian-structure kron` | Dependency stack is risky in current environment. |

---
