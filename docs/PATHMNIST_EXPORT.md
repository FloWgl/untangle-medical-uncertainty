# PathMNIST Data Setup

This document describes the local data layouts used by the clean PathMNIST and
PathMNIST-C experiments.

## Clean PathMNIST

Export the MedMNIST train, validation, and test splits to an ImageFolder layout:

```bash
python tools/export_medmnist_to_imagefolder.py \
  --output-dir /path/to/pathmnist_cache \
  --download \
  --dataset pathmnist \
  --splits train,val,test
```

The resulting structure is:

```text
/path/to/pathmnist_cache/
  pathmnist/
    train/<class_idx>/*.png
    val/<class_idx>/*.png
    test/<class_idx>/*.png
```

Use `hard/pathmnist` as both `--dataset` and `--dataset-id`. Bare
`pathmnist` remains a compatibility alias. Set `PATHMNIST_PREFER_MEDMNIST=1`
to load the MedMNIST cache directly when an ImageFolder export also exists.

A baseline command is:

```bash
python train.py \
  --dataset hard/pathmnist \
  --dataset-id hard/pathmnist \
  --data-dir /path/to/pathmnist_cache/pathmnist \
  --data-dir-id /path/to/pathmnist_cache/pathmnist \
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

The effective learning rate is
`lr_base * (batch_size * accumulation_steps / 256)`. Keep the shared training
settings fixed when comparing methods and add only the method-specific options
listed in `PATHMNIST_EXPERIMENT_SETTINGS.md`.

## Official PathMNIST-C Files

The MedMNIST-C archive contains one NPZ file per corruption:

- `brightness_down.npz`
- `contrast_up.npz`
- `defocus_blur.npz`
- `motion_blur.npz`
- `jpeg_compression.npz`
- `pixelate.npz`
- `bubble.npz`
- `stain_deposit.npz`

Download and submit evaluation with:

```bash
DATA_ROOT=/path/to/datasets \
CHECKPOINT_MANIFEST=/path/to/checkpoints.csv \
  bash tools/hpc/download_official_pathmnist_c_and_submit.sh
```

The loader accepts the official NPZ schemas and resolves the evaluation images
and labels from the archive. These files represent the fixed MedMNIST-C
corruptions; they are evaluated with `--severities 1` because the NPZ files do
not encode a CIFAR-C-style severity index.

## Generated Severity Axis

The separate severity-axis experiment applies the same eight corruption families
at levels 1 through 5 during evaluation. It uses clean PathMNIST images rather
than expecting severity-indexed NPZ files.

Create a local manifest from
`configs/pathmnist_checkpoint_manifest.example.csv`, then run:

```bash
CHECKPOINT_MANIFEST=configs/pathmnist_checkpoint_manifest.csv \
DATA_ROOT=/path/to/datasets \
  bash tools/hpc/submit_pathmnist_severity_axis.sh
```

See `pathmnist_severity_axis_experiment.md` for the protocol and
`docs/results/pathmnist/pathmnist_c_severity_axis_summary.csv` for the reference
summary.

## Method Requirements

Trainable methods use the common PathMNIST recipe wherever their architecture
allows it. Post-hoc methods additionally require compatible checkpoints:

- Deep Ensemble and Fast Deep Ensemble require multiple member checkpoints.
- Temperature Scaling and Laplace require a trained base checkpoint.
- SWAG requires checkpoints collected during training.
- DDU and Mahalanobis require a pass over training features.
- Laplace uses the dependency versions pinned in `pyproject.toml`.

Complete commands should be generated through
`tools/hpc/generate_pathmnist_hpc_commands.py`; local paths and checkpoint
manifests are deliberately supplied at runtime.
