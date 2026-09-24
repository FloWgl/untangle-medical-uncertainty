PathMNIST export and local cache
================================

This document describes how to export clean PathMNIST to a local ImageFolder
layout and how to use PathMNIST-C directly from the official Zenodo NPZ files.

Script
------
The repository includes a CLI script at `tools/export_medmnist_to_imagefolder.py` that
exports clean PathMNIST.

When you use the exported data with the training code, prefer the hard-label
dataset names `hard/pathmnist` and `hard/pathmnist-c`.

Basic examples
--------------
- Export only clean PathMNIST (train/val/test) into a local cache:

```bash
python tools/export_medmnist_to_imagefolder.py \
  --output-dir /path/to/pathmnist_cache \
  --download \
  --dataset pathmnist \
  --splits train,val,test
```

Notes
-----
- The `--download` flag allows the script to let `medmnist` download the data if needed.
- The script creates the following layout under `--output-dir`:

```
/path/to/pathmnist_cache/
  pathmnist/
    train/<class_idx>/*.png
    val/<class_idx>/*.png
    test/<class_idx>/*.png
```

- If you prefer using medmnist's cache rather than exported ImageFolder data, set
  the environment variable `PATHMNIST_PREFER_MEDMNIST=1` before running code that
  instantiates the `PathMNIST` dataset in the repo.
- The repo also accepts bare `pathmnist` / `pathmnist-c` as compatibility aliases,
  but the hard-prefixed names are the canonical ones.

When to use
-----------
- Use the exported ImageFolder structure when you want to train/evaluate repeatedly
  without depending on `medmnist` network downloads and to keep experiments reproducible.
- Use the official Zenodo NPZ files for PathMNIST-C corruptions.

Benchmark recipe
----------------
The repository's default benchmark recipe is encoded in the CLI defaults. For a
baseline hard-label PathMNIST run, the exact settings are:

- `--dataset hard/pathmnist`
- `--dataset-id hard/pathmnist`
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

The effective learning rate is computed as `lr = lr_base * (batch_size * accumulation_steps / 256)`.
With the defaults above, that gives `lr = 0.008`.

Copy-paste command:

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /path/to/pathmnist_cache \
  --data-dir-id /path/to/pathmnist_cache \
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

If you want the exact same recipe for another method, keep the baseline training
settings above and only change `--method-name` plus the method-specific flags.

Using Zenodo PathMNIST-C NPZ files directly
-------------------------------------------
The Zenodo archive for MedMNIST-C PathMNIST (for example `pathmnist.zip`) contains
corruptions as `.npz` files such as:

- `brightness_down.npz`
- `contrast_up.npz`
- `defocus_blur.npz`
- `motion_blur.npz`
- `jpeg_compression.npz`
- `pixelate.npz`
- `bubble.npz`
- `stain_deposit.npz`

The repository can read these files directly for `hard/pathmnist-c` when
`--data-dir-id` points to the extracted NPZ directory (for example
`/path/to/tmp_pathmnist_c/pathmnist`).

Important:

- NPZ corruptions are treated as fixed, paper-provided corruptions.
- Use `--severities 1` (the NPZ format does not encode CIFAR-C style severity 1..5).
- Set `--ood-transforms-test` (and typically `--ood-transforms-eval`) to NPZ file stems.

Example command:

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist-c \
  --data-dir /path/to/pathmnist_cache \
  --data-dir-id /path/to/tmp_pathmnist_c/pathmnist \
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
  --severities 1 \
  --ood-transforms-eval brightness_down,contrast_up,defocus_blur,motion_blur,jpeg_compression,pixelate,bubble,stain_deposit \
  --ood-transforms-test brightness_down,contrast_up,defocus_blur,motion_blur,jpeg_compression,pixelate,bubble,stain_deposit \
  --evaluate-on-test-sets
```

Paper-to-PathMNIST adaptation
-----------------------------
The paper benchmarked the method families on ImageNet and CIFAR-10 style datasets,
so PathMNIST is an adaptation target rather than a dataset with published paper
numbers. The transferable part is the method family and training/evaluation
structure; the non-transferable part is any metric or decomposition that depends on
soft annotations.

What carries over cleanly:

- CE Baseline
- Temperature Scaling
- MC-Dropout
- Correctness Prediction
- Loss Prediction
- EDL
- HET / HET-XL
- HetClassNN
- SNGP
- PostNet
- Shallow Ensemble
- Deep Ensemble
- DDU
- Mahalanobis
- SWAG

What needs extra care:

- Laplace: depends on an unavailable `curvlinops` dependency in this environment
- DUQ: can run, but the default configuration hit GPU memory pressure in smoke tests
- Deep Ensemble / SWAG / post-hoc methods: require one or more trained checkpoints

What does not transfer:

- Soft-label-only metrics and analyses
- Any Bregman / soft decomposition outputs that require soft annotations
- ImageNet- or CIFAR-10-specific label resources like ReaL or CIFAR-10H

PathMNIST paper-style next run
------------------------------
To continue with the same benchmark recipe but a different method, keep the PathMNIST
defaults above and switch only the method-specific flags. For the next runnable
method, use correctness prediction:

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /path/to/pathmnist_cache \
  --data-dir-id /path/to/pathmnist_cache \
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

Persistent run log
------------------
This section records the exact commands and observed results from the PathMNIST
adaptation work so they can be reused in reporting.

### Completed runs

#### CE baseline smoke run

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /tmp/pathmnist_real \
  --data-dir-id /tmp/pathmnist_real \
  --dataset-download \
  --evaluate-on-test-sets \
  --discard-ood-test-sets \
  --epochs 1 \
  --num-workers 2 \
  --num-eval-workers 2 \
  --batch-size 64 \
  --accumulation-steps 1 \
  --log-interval 20 \
  --storage-device cuda
```

- Status: completed
- Eval accuracy: 0.7543
- Eval metric: 0.8213
- Epoch time: 231.03 s

#### Temperature scaling post-hoc run

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /tmp/pathmnist_real \
  --data-dir-id /tmp/pathmnist_real \
  --dataset-download \
  --method-name temperature-scaling \
  --weight-paths checkpoints/20260429-203407-465301-timm_resnet_50-224/checkpoint_0.pt \
  --evaluate-on-test-sets \
  --discard-ood-test-sets \
  --epochs 0 \
  --num-workers 2 \
  --num-eval-workers 2 \
  --batch-size 64 \
  --accumulation-steps 1 \
  --log-interval 20 \
  --storage-device cuda
```

- Status: completed
- Result: successful post-hoc evaluation, no training epochs

#### MC-Dropout smoke run

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /tmp/pathmnist_real \
  --data-dir-id /tmp/pathmnist_real \
  --dataset-download \
  --method-name mc-dropout \
  --evaluate-on-test-sets \
  --discard-ood-test-sets \
  --epochs 1 \
  --num-workers 2 \
  --num-eval-workers 2 \
  --batch-size 64 \
  --accumulation-steps 1 \
  --log-interval 20 \
  --storage-device cuda
```

- Status: completed
- Eval accuracy: 0.7255
- Eval metric: 0.7793
- Epoch time: 324.55 s

### Failed or blocked runs

#### Laplace

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /tmp/pathmnist_real \
  --data-dir-id /tmp/pathmnist_real \
  --dataset-download \
  --method-name laplace \
  --weight-paths checkpoints/20260429-203407-465301-timm_resnet_50-224/checkpoint_0.pt \
  --evaluate-on-test-sets \
  --discard-ood-test-sets \
  --epochs 0 \
  --num-workers 2 \
  --num-eval-workers 2 \
  --batch-size 64 \
  --accumulation-steps 1 \
  --log-interval 20 \
  --storage-device cuda
```

- Status: blocked
- Error: `curvlinops._base` missing in the installed `laplace` package

#### DUQ

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /tmp/pathmnist_real \
  --data-dir-id /tmp/pathmnist_real \
  --dataset-download \
  --method-name duq \
  --evaluate-on-test-sets \
  --discard-ood-test-sets \
  --epochs 1 \
  --num-workers 2 \
  --num-eval-workers 2 \
  --batch-size 64 \
  --accumulation-steps 1 \
  --log-interval 20 \
  --storage-device cuda
```

- Status: blocked
- Error: CUDA out of memory at training time with batch size 64

### Current run

#### Correctness prediction, paper-style defaults

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /tmp/pathmnist_real \
  --data-dir-id /tmp/pathmnist_real \
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
  --discard-ood-test-sets \
  --num-workers 2 \
  --num-eval-workers 2 \
  --storage-device cuda
```

- Status: running at the time of writing
- Purpose: paper-style correctness prediction on hard-label PathMNIST

#### Loss prediction, queued next

The next experiment in the PathMNIST pipeline is loss prediction with the same
shared recipe and hard/pathmnist dataset namespace. The queued command is:

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /tmp/pathmnist_real \
  --data-dir-id /tmp/pathmnist_real \
  --dataset-download \
  --method-name loss-prediction \
  --loss loss-prediction \
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
  --discard-ood-test-sets \
  --num-workers 2 \
  --num-eval-workers 2 \
  --storage-device cuda
```

The `--detach-uncertainty-target` flag is intentionally left at its default value
here so the queued command stays aligned with the extracted paper-style settings.

Troubleshooting
---------------
- If a corruption requires ImageMagick/wand (e.g., `motion_blur` or `snow`) and it's not
  installed, the exporter will skip that corruption unless you pass `--strict`, which will
  make the script fail early.

Contact
-------
If you need additional export formats or custom corruptions, I can extend the script to
write TFRecord/LMDB datasets or include additional preprocessing steps.
