#!/usr/bin/env bash
# Submit remaining PathMNIST gaps: missing label-noise variants and post-hoc methods.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
SBATCH_BIN="${SBATCH_BIN:-sbatch.tinygpu}"
SLURM_PARTITION="${SLURM_PARTITION:-a100}"
SLURM_GRES="${SLURM_GRES:-gpu:a100:1}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
ENV_SETUP="${ENV_SETUP:-$REPO_ROOT/hpc_env.sh}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_LABEL_NOISE="${RUN_LABEL_NOISE:-1}"
RUN_POSTHOC="${RUN_POSTHOC:-1}"

LABEL_NOISE_METHODS="${LABEL_NOISE_METHODS:-deep-correctness-prediction,deep-loss-prediction,postnet}"
LABEL_NOISE_FRACTIONS="${LABEL_NOISE_FRACTIONS:-0.1 0.2 0.4}"
LABEL_NOISE_SEED="${LABEL_NOISE_SEED:-1042}"
POSTHOC_MANIFEST="${POSTHOC_MANIFEST:-configs/pathmnist_posthoc_weight_paths_${RUN_STAMP}.env}"

LOG_DIR="$REPO_ROOT/logs/pathmnist_hpc_${RUN_STAMP}-missing-gaps"
STATUS_FILE="$LOG_DIR/status.log"
mkdir -p "$LOG_DIR" "$REPO_ROOT/configs"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$STATUS_FILE"
}

log "Submitting PathMNIST missing-gap pipeline."
log "run_stamp=${RUN_STAMP}"
log "label_noise_methods=${LABEL_NOISE_METHODS}"
log "label_noise_fractions=${LABEL_NOISE_FRACTIONS}"
log "posthoc_manifest=${POSTHOC_MANIFEST}"

if [[ "$RUN_LABEL_NOISE" == "1" ]]; then
  log "Submitting missing label-noise variants."
  RUN_ID="${RUN_STAMP}-pathmnist-label-noise-missing" \
  METHODS="$LABEL_NOISE_METHODS" \
  LABEL_NOISE_FRACTIONS="$LABEL_NOISE_FRACTIONS" \
  LABEL_NOISE_SEED="$LABEL_NOISE_SEED" \
  SBATCH_BIN="$SBATCH_BIN" \
  SLURM_PARTITION="$SLURM_PARTITION" \
  SLURM_GRES="$SLURM_GRES" \
  SLURM_TIME="$SLURM_TIME" \
  ENV_SETUP="$ENV_SETUP" \
  DATA_ROOT="$DATA_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash tools/hpc/submit_pathmnist_label_noise_pipeline.sh | tee -a "$STATUS_FILE"
else
  log "Skipping label-noise submission because RUN_LABEL_NOISE=${RUN_LABEL_NOISE}."
fi

if [[ "$RUN_POSTHOC" == "1" ]]; then
  if [[ ! -f "$POSTHOC_MANIFEST" ]]; then
    log "Building post-hoc manifest from completed PathMNIST Slurm logs."
    python3 tools/hpc/build_pathmnist_posthoc_env_manifest.py \
      --repo-root "$REPO_ROOT" \
      --output "$POSTHOC_MANIFEST" | tee -a "$STATUS_FILE"
  fi
  log "Submitting post-hoc methods from ${POSTHOC_MANIFEST}."
  RUN_ID="${RUN_STAMP}-pathmnist-posthoc-paperstyle" \
  SBATCH_BIN="$SBATCH_BIN" \
  SLURM_PARTITION="$SLURM_PARTITION" \
  SLURM_GRES="$SLURM_GRES" \
  SLURM_TIME="$SLURM_TIME" \
  ENV_SETUP="$ENV_SETUP" \
  DATA_ROOT="$DATA_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash tools/hpc/submit_pathmnist_posthoc_from_env_manifest.sh "$POSTHOC_MANIFEST" | tee -a "$STATUS_FILE"
else
  log "Skipping post-hoc submission because RUN_POSTHOC=${RUN_POSTHOC}."
fi

log "Missing-gap pipeline submitted."
