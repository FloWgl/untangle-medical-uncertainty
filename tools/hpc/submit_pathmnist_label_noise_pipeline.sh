#!/usr/bin/env bash
# Submit the PathMNIST synthetic-label-noise experiment on TinyGPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SBATCH_BIN="${SBATCH_BIN:-sbatch.tinygpu}"
SLURM_PARTITION="${SLURM_PARTITION:-a100}"
SLURM_GRES="${SLURM_GRES:-gpu:a100:1}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python}"

METHODS="${METHODS:-ce-baseline,mc-dropout,edl,sngp,het,het-xl,hetclassnn,shallow-ensemble,duq,correctness-prediction,loss-prediction}"
LABEL_NOISE_FRACTIONS="${LABEL_NOISE_FRACTIONS:-0.1 0.2 0.4}"
LABEL_NOISE_SEED="${LABEL_NOISE_SEED:-1042}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_STAMP}-pathmnist-label-noise}"
LOG_DIR="$REPO_ROOT/logs/pathmnist_hpc_${RUN_ID}"
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status.log"

{
  echo "Submitting PathMNIST synthetic-label-noise experiment."
  echo "run_id=${RUN_ID}"
  echo "methods=${METHODS}"
  echo "label_noise_fractions=${LABEL_NOISE_FRACTIONS}"
  echo "label_noise_seed=${LABEL_NOISE_SEED}"
} | tee "$STATUS_FILE"

PHASE="label-noise" \
RUN_ID="$RUN_ID" \
METHODS="$METHODS" \
SKIP_ENSEMBLE_SEEDS=1 \
DATA_ROOT="$DATA_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
SBATCH_BIN="$SBATCH_BIN" \
SLURM_PARTITION="$SLURM_PARTITION" \
SLURM_GRES="$SLURM_GRES" \
SLURM_TIME="$SLURM_TIME" \
LABEL_NOISE_FRACTIONS="$LABEL_NOISE_FRACTIONS" \
LABEL_NOISE_SEED="$LABEL_NOISE_SEED" \
bash tools/hpc/submit_pathmnist_hpc.sh | tee -a "$STATUS_FILE"

cat <<EOF_SUMMARY | tee -a "$STATUS_FILE"

Synthetic-label-noise pipeline submitted.
commands=$LOG_DIR/commands.txt
status=$STATUS_FILE

Interpretation:
- label noise is injected only into the training labels;
- validation, clean test, and optional corruption test labels remain unchanged;
- the run tests whether aleatoric-oriented methods react to controlled label noise
  more than epistemic-oriented components react.
EOF_SUMMARY
