# Experiment Reproducibility

This repository contains the code and portable metadata required to reproduce
the CIFAR-10, PathMNIST, PathMNIST-C, and MHIST experiments. Training outputs,
datasets, checkpoints, cluster logs, thesis sources, presentations, and assistant
or editor state are intentionally not versioned.

## 1. Environment

Use Python 3.11 or 3.12. A CUDA-enabled PyTorch installation is recommended for
training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

The `Singularity` definition provides the container starting point used by the
original benchmark. Package requirements are declared in both `pyproject.toml`
and `requirements.txt`. `curvlinops-for-pytorch` is pinned because the Laplace
implementation depends on its API.

## 2. Data

Do not commit downloaded datasets. Place them under a local data root and pass
that path through the command-line options or `DATA_ROOT`.

### CIFAR-10

CIFAR-10 is downloaded by `torchvision` when requested by the training pipeline.
The reconstruction launchers are in `tools/launchers/`.

### PathMNIST

Export the MedMNIST data into the expected image-folder layout:

```bash
python tools/export_medmnist_to_imagefolder.py \
  --download \
  --output-dir "$DATA_ROOT/pathmnist_cache/pathmnist"
```

PathMNIST-C setup and the official-corruption submission entry point are in
`tools/hpc/download_official_pathmnist_c_and_submit.sh`. The severity-axis
experiment generates corruptions at evaluation time and does not require a
severity-indexed NPZ archive.

The exporter requires the `medmnist` package and writes only clean PathMNIST.
Use `--max-samples 5` for a small export check or `--max-samples 0` for the
complete dataset. Set `PATHMNIST_PREFER_MEDMNIST=1` to force direct MedMNIST
loading when an image-folder export also exists.

### MHIST

MHIST requires accepting the dataset's access terms. After receiving the
download URLs, run:

```bash
python tools/prepare_mhist.py \
  --root "$DATA_ROOT/MHIST" \
  --images-url "$MHIST_IMAGES_URL" \
  --annotations-url "$MHIST_ANNOTATIONS_URL" \
  --md5-url "$MHIST_MD5_URL"
```

The URLs are credentials for a temporary download and must not be committed.

## 3. PathMNIST Experiment Matrix

`tools/hpc/generate_pathmnist_hpc_commands.py` is the canonical command
generator. It supports these phases:

- `train`: full-data training.
- `reduced`: 50% and 10% data-scarcity training.
- `label-noise`: 10%, 20%, and 40% synthetic training-label noise.
- `corruption`: fixed PathMNIST-C evaluation.
- `corruption-severity`: severities 1 through 5.
- `posthoc`: post-hoc uncertainty methods from checkpoint manifests.
- `all`: the complete generated matrix.

Generate commands without submitting them:

```bash
python tools/hpc/generate_pathmnist_hpc_commands.py \
  --phase all \
  --output /tmp/pathmnist_commands.txt \
  --pathmnist-dir "$DATA_ROOT/pathmnist_cache/pathmnist" \
  --pathmnist-c-dir "$DATA_ROOT/pathmnist_c/pathmnist"
```

Submit a SLURM array:

```bash
DATA_ROOT="$DATA_ROOT" PHASE=train \
  bash tools/hpc/submit_pathmnist_hpc.sh

DATA_ROOT="$DATA_ROOT" \
  bash tools/hpc/submit_pathmnist_label_noise_pipeline.sh
```

For the severity axis, create a local manifest from
`configs/pathmnist_checkpoint_manifest.example.csv`, then run:

```bash
CHECKPOINT_MANIFEST=configs/pathmnist_checkpoint_manifest.csv \
DATA_ROOT="$DATA_ROOT" \
  bash tools/hpc/submit_pathmnist_severity_axis.sh
```

Cluster-specific binaries, partitions, GPU resources, runtimes, and Python
paths can be overridden with `SBATCH_BIN`, `SLURM_PARTITION`, `SLURM_GRES`,
`SLURM_TIME`, and `PYTHON_BIN`.

## 4. CIFAR-10 and MHIST

Recovered CIFAR-10 launchers live in `tools/launchers/`. The portable command
lists in `configs/` cover the full-data and scarcity runs. Post-hoc methods need
an ensemble manifest copied from
`configs/cifar10_posthoc_weight_paths.env.example` after the five CE members
finish.

MHIST dataset integration is implemented in `untangle/datasets/mhist.py`.
The queue entry points are the `tools/run_mhist_*_queue.sh` scripts. Set
`PYTHON_BIN`, the local dataset path, and scheduler variables for the target
cluster before launching.

## 5. Analysis and Reference Results

Core analysis entry points include:

```bash
python tools/analyze_pathmnist_hpc_results.py --help
python tools/analyze_mhist_disentanglement.py --help
python tools/compare_decomposition_formulas.py --help
python tools/summarize_pathmnist_intervention_outputs.py --help
python tools/summarize_pathmnist_severity_axis.py --help
python tools/summarize_cifar10_posthoc_outputs.py --help
```

Lightweight CSV summaries under `docs/results/cifar10/`,
`docs/results/pathmnist/`, and `docs/results/mhist/` provide reference values
for checking reproduced runs. Raw tensors, checkpoints, logs, and W&B downloads
are excluded because they are generated artifacts or too large for Git.

Only final result tables are versioned. Scheduler snapshots, queue state,
retry reports, and dated progress reports are deliberately excluded because
they describe a particular execution environment rather than the experiment
protocol.

The normalized CIFAR-10 values in
`docs/results/cifar10/paper_outcomes/paper_results_normalized.csv` are the
preferred source for programmatic comparison. The accompanying `raw_tables/`
files preserve the extracted text from Tables H.1--H.4 and I.1--I.2. These
artifacts do not contain exact values for every main-paper figure; comparisons
without extracted table values therefore use recovered W&B run metrics.

The MHIST analysis uses normalized binary entropy of seven pathologist votes as
its primary ambiguity target and normalized minority-vote fraction as an
alternative. Changes from 100% training data to 50% and 10% form the epistemic
scarcity intervention. Undefined correlations remain blank when an uncertainty
component is constant, and non-finite decomposition entries are excluded
pairwise and reported through `num_finite_samples`.

## 6. Verification

Run the focused test suite before launching expensive jobs:

```bash
pytest tests/test_pathmnist_smoke.py tests/test_mhist_smoke.py
pytest tests/test_duq_wrapper.py tests/test_swag_wrapper.py
```

For every run, record the Git commit, command manifest, random seed, dataset
version, scheduler settings, and checkpoint path outside the repository. This is
enough to trace a result without committing machine-specific paths or large
artifacts.

## Repository Boundary

The repository intentionally excludes:

- thesis and presentation sources or generated documents;
- Codex, Copilot, editor, or chat-session state;
- downloaded datasets and temporary exports;
- model checkpoints, raw uncertainty tensors, and W&B artifacts;
- scheduler logs, process IDs, and archived working directories;
- machine-specific checkpoint manifests containing local paths.
