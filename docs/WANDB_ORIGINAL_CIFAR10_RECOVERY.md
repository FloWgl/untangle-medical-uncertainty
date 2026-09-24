# Original CIFAR-10 W&B Config Recovery

This records the recovered W&B metadata for the CIFAR-10 post-hoc base models.

## Result

The original base checkpoints used by the post-hoc CIFAR-10 methods come from final CE baseline sweep `uo3gu133`.

The portable configuration is represented by the tracked sweep YAMLs in
`docs/wandb_sweeps/` and the reconstructed launchers in `tools/launchers/`.
Raw W&B downloads are not versioned.

## Common Training Config

- `dataset`: `hard/cifar10`
- `dataset-id`: `soft/cifar10`
- `model-name`: `untangle/wide_resnet_c_preact_26_10`
- `method-name`: `ce-baseline`
- `epochs`: `200`
- `batch-size`: `128`
- `accumulation-steps`: `1`
- `lr`: `0.1395872780316071`
- `weight-decay`: `0.004022874547456138`
- `opt`: `nesterov`
- `momentum`: `0.9`
- `crop-pct`: `1`
- `hflip`: `0.5`
- `img-size`: `32`
- `padding`: `2`
- `mean`: `0.4914,0.4822,0.4465`
- `std`: `0.2023,0.1994,0.2010`
- `sched-kwargs`: `sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1`
- `storage-device`: `cuda`

## Run To Checkpoint Mapping

| Run | Display name | Seed | Best epoch | Checkpoint directory |
|---|---|---:|---:|---|
| `jjyl7phf` | `misty-sweep-1` | 0 | 164 | `20241025-125108-441033-untangle_wide_resnet_c_preact_26_10-32` |
| `utahcjyb` | `astral-sweep-2` | 1 | 159 | `20241025-183701-517629-untangle_wide_resnet_c_preact_26_10-32` |
| `1bs3bwlk` | `fiery-sweep-3` | 2 | 187 | `20241026-001813-337221-untangle_wide_resnet_c_preact_26_10-32` |
| `rtm42hct` | `prime-sweep-4` | 3 | 186 | `20241026-060010-289890-untangle_wide_resnet_c_preact_26_10-32` |
| `z28hw0ij` | `fearless-sweep-5` | 4 | 159 | `20241026-114039-621072-untangle_wide_resnet_c_preact_26_10-32` |

## Post-Hoc Link

The checkpoint directory names above match the author paths embedded in the post-hoc CIFAR-10 sweep YAMLs:

- `bn1hsbqz`: deep ensemble, uses `checkpoint_best.pt`
- `lh04ospw`: temperature scaling, uses `checkpoint_best.pt`
- `nnle8epz`: Laplace, uses `checkpoint_best.pt`
- `h0m0bybl`: Mahalanobis, uses `checkpoint_last.pt`
- `zsiqsl6u`: SWAG, uses `checkpoint_last.pt`

## Checkpoint Availability

W&B run files contain config/log/summary/requirements metadata, but not the checkpoint binaries. The original `.pt` files are therefore not recoverable from W&B alone. The closest reproducible path is to retrain the five CE members with:

```bash
bash tools/launchers/run_bn1hsbqz_members_reconstructed.sh --train-subset 1.0 --discard-ood-test-sets
```

On HPC, use `configs/cifar10_hpc_ensemble_member_commands.txt` with the existing SLURM array wrapper.
