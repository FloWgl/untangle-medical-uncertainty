# PathMNIST Corruption Severity-Axis Experiment

This experiment is separate from the fixed official PathMNIST-C NPZ evaluation.
The official NPZ files used earlier are treated as one fixed corruption set. This
severity-axis experiment instead applies the repository's PathMNIST-C-style
corruptions on the fly to clean PathMNIST images at severities 1 through 5.

## Launch

Prepare a checkpoint manifest with at least these columns:

```csv
method,checkpoint
ce-baseline,checkpoints/.../checkpoint_best.pt
correctness-prediction,checkpoints/.../checkpoint_best.pt
```

Then submit on the SLURM system:

```bash
CHECKPOINT_MANIFEST=configs/pathmnist_checkpoint_manifest.csv \
DATA_ROOT="$DATA_ROOT" \
PYTHON_BIN="${PYTHON_BIN:-python3}" \
tools/hpc/submit_pathmnist_severity_axis.sh
```

The launcher writes commands to `logs/pathmnist_hpc_<run-id>/commands.txt` and
submits one array task per checkpointed method. Each task evaluates severities
`1,2,3,4,5` for:

- `brightness_down`
- `contrast_up`
- `defocus_blur`
- `motion_blur`
- `jpeg_compression`
- `pixelate`
- `bubble`
- `stain_deposit`

## Summarise

After the array finishes, add each severity-evaluation output directory to the
manifest as `severity_output_dir`, or point `checkpoint` to the output directory's
`checkpoint_best.pt` if the evaluation wrote one. Then run:

```bash
python3 tools/summarize_pathmnist_severity_axis.py \
  --manifest configs/pathmnist_checkpoint_manifest.csv \
  --output-dir docs/results/pathmnist
```

The summarizer writes:

- `docs/results/pathmnist/pathmnist_c_severity_axis_detailed.csv`
- `docs/results/pathmnist/pathmnist_c_severity_axis_summary.csv`

Use these files only after confirming that all intended methods finished.
