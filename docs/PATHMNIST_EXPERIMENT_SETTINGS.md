# PathMNIST Experiment Settings

This file collects the experiment families supported by the repository, the shared
PathMNIST settings, and the method-specific flags needed to reproduce or adapt the
benchmark on hard-label PathMNIST.

PathMNIST context
-----------------

- Dataset names: `hard/pathmnist` and `hard/pathmnist-c`
- Dataset source: exported PathMNIST ImageFolder cache for ID and Zenodo NPZ PathMNIST-C for OOD
- Paper context: the original paper benchmarked ImageNet and CIFAR-10 style datasets,
  so PathMNIST is an adaptation target rather than a published-paper benchmark point.

Shared benchmark defaults
-------------------------

These are the repository defaults that define the baseline PathMNIST recipe unless a
method needs a specific override:

- `--model-name timm/resnet_50`
- `--loss cross-entropy`
- `--opt adamw`
- `--weight-decay 2e-5`
- `--lr-base 0.001`
- `--batch-size 128`
- `--accumulation-steps 16`
- `--epochs 90`
- `--seed 42`
- `--img-size 224`
- `--mean 0.485,0.456,0.406`
- `--std 0.229,0.224,0.225`
- `--dataset hard/pathmnist`
- `--dataset-id hard/pathmnist`

The learning rate is derived from the repo rule:

`lr = lr_base * (batch_size * accumulation_steps / 256)`

With the defaults above, the effective learning rate is `0.008`.

Experiment roster
-----------------

The list below is organized by runtime style and includes the settings that matter
for each method on PathMNIST.

### 1. Training-time methods

These methods train a model directly on PathMNIST.

#### CE Baseline

- `--method-name ce-baseline`
- `--loss cross-entropy`
- No additional method-specific flags required

#### MC-Dropout

- `--method-name mc-dropout`
- `--dropout-probability 0.05`
- `--num-mc-samples 10`
- Optional: `--use-filterwise-dropout`

#### Correctness Prediction

- `--method-name correctness-prediction`
- `--loss correctness-prediction`
- `--lambda-uncertainty-loss 0.01`
- Optional: `--use-top5-correctness` for datasets with at least 5 classes; PathMNIST has
  enough classes for this, but top-1 is usually the default comparable choice

#### Deep Correctness Prediction

- `--method-name deep-correctness-prediction`
- `--loss correctness-prediction`
- `--lambda-uncertainty-loss 0.01`
- `--num-hooks` if you want to restrict feature taps
- `--module-name-regex '^(act1|layer[1-4])$'` by default

#### Loss Prediction

- `--method-name loss-prediction`
- `--loss loss-prediction`
- `--lambda-uncertainty-loss 0.01`
- Optional: `--detach-uncertainty-target`

#### Deep Loss Prediction

- `--method-name deep-loss-prediction`
- `--loss loss-prediction`
- `--lambda-uncertainty-loss 0.01`
- `--num-hooks` if needed
- `--module-name-regex '^(act1|layer[1-4])$'` by default

#### EDL

- `--method-name edl`
- `--loss edl`
- `--edl-start-epoch 0`
- `--edl-scaler 1.0`
- `--edl-activation exp`

#### HET

- `--method-name het-xl`
- `--use-het`
- `--num-mc-samples 10`
- `--temperature 1.5`
- Optional: `--matrix-rank 15` for HET-XL behavior

#### HET-XL

- `--method-name het-xl`
- Do not pass `--use-het`
- `--matrix-rank 15`
- `--num-mc-samples 10`
- `--temperature 1.5`

#### HetClassNN

- `--method-name hetclassnn`
- `--dropout-probability 0.05`
- `--num-mc-samples 10`
- `--num-mc-samples-integral 1000`
- Optional: `--use-filterwise-dropout`

#### SNGP

- `--method-name sngp`
- `--use-spectral-normalization`
- `--spectral-normalization-iteration 1`
- `--spectral-normalization-bound 6`
- `--num-random-features 1024`
- `--gp-kernel-scale 1.0`
- `--gp-output-bias 0.0`
- `--gp-random-feature-type orf`
- `--gp-cov-momentum -1`
- `--gp-cov-ridge-penalty 1.0`
- `--gp-input-dim 128`
- Optional: `--use-spectral-normalized-batch-norm`, `--use-tight-norm-for-pointwise-convs`,
  `--use-input-normalized-gp`

#### PostNet

- `--method-name postnet`
- `--latent-dim 6`
- `--num-hidden-features 256`
- `--num-density-components 6`
- Optional: `--use-batched-flow`
- `--uce-regularization-factor 1e-5`

#### Shallow Ensemble

- `--method-name shallow-ensemble`
- `--num-heads 10`

#### Deep Ensemble

- `--method-name deep-ensemble`
- `--weight-paths` must contain one checkpoint per ensemble member
- No special training-time loss; each member is typically trained with the baseline recipe

#### DUQ

- `--method-name duq`
- `--loss duq`
- `--rbf-length-scale 0.1`
- `--ema-momentum 0.999`
- `--lambda-gradient-penalty 0.75`
- GPU memory use may require a smaller batch size than the shared default

### 2. Post-hoc methods

These methods need at least one checkpoint from a trained model.

#### Temperature Scaling

- `--method-name temperature-scaling`
- `--weight-paths <checkpoint>`
- Optional: `--use-temperature-scaling` is not needed here; the wrapper itself handles it

#### Mahalanobis

- `--method-name mahalanobis`
- `--weight-paths <checkpoint>`
- `--magnitude 0.001`
- `--num-hooks` if you want to limit feature hooks
- `--module-name-regex '^(act1|layer[1-4])$'` by default
- Optional: `--module-type` for explicit hook selection

#### SWAG

- `--method-name swag`
- `--weight-paths <checkpoint>`
- `--use-low-rank-cov`
- `--max-rank 20`
- `--num-checkpoints-per-epoch 4`

#### DDU

- `--method-name ddu`
- `--weight-paths <checkpoint>`
- `--use-spectral-normalization`
- `--spectral-normalization-iteration 1`
- `--spectral-normalization-bound 6`
- `--use-spectral-normalized-batch-norm`
- Optional: `--use-tight-norm-for-pointwise-convs`

#### Laplace

- `--method-name laplace`
- `--weight-paths <checkpoint>`
- `--num-mc-samples 10`
- `--num-mc-samples-cv 50`
- `--pred-type glm`
- `--hessian-structure kron`

### 3. Methods that are feasible but need more care

These are supported by the code, but they require either extra checkpoints, more memory,
or additional post-hoc data passes.

- Deep Ensemble: needs multiple trained checkpoints
- Fast Deep Ensemble: needs multiple trained checkpoints
- SWAG: needs checkpoints across training
- DDU: needs a trained checkpoint plus feature extraction over the training set
- Mahalanobis: needs a trained checkpoint and covariance estimation
- Laplace: requires the pinned `laplace-torch` and `curvlinops-for-pytorch` versions
- DUQ: may require smaller batch size or a smaller backbone to avoid OOM

### 4. What does not transfer from the paper

These are not PathMNIST-specific experiment settings; they are paper-only metrics or
outputs that depend on soft labels or external datasets.

- Soft-label metrics
- Ground-truth Bregman targets derived from soft label distributions
- CIFAR-10H-specific evaluation
- ImageNet-ReaL-specific evaluation

Reporting-ready commands
------------------------

Baseline hard-label PathMNIST:

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /path/to/pathmnist_cache/pathmnist \
  --data-dir-id /path/to/pathmnist_cache/pathmnist \
  --dataset-download \
  --method-name ce-baseline \
  --loss cross-entropy \
  --model-name timm/resnet_50 \
  --opt adamw \
  --weight-decay 2e-5 \
  --lr-base 0.001 \
  --batch-size 128 \
  --accumulation-steps 16 \
  --epochs 90 \
  --seed 42 \
  --img-size 224 \
  --evaluate-on-test-sets \
  --discard-ood-test-sets
```

Correctness prediction on hard-label PathMNIST:

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /path/to/pathmnist_cache/pathmnist \
  --data-dir-id /path/to/pathmnist_cache/pathmnist \
  --dataset-download \
  --method-name correctness-prediction \
  --loss correctness-prediction \
  --lambda-uncertainty-loss 0.01 \
  --model-name timm/resnet_50 \
  --opt adamw \
  --weight-decay 2e-5 \
  --lr-base 0.001 \
  --batch-size 128 \
  --accumulation-steps 16 \
  --epochs 90 \
  --seed 42 \
  --img-size 224 \
  --evaluate-on-test-sets \
  --discard-ood-test-sets
```

Reference results
-----------------

Completed experiment summaries are stored under `docs/results/pathmnist/`.
Smoke-test measurements and cluster-specific failures are not part of the stable
experiment specification.
