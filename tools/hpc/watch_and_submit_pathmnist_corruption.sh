#!/usr/bin/env bash
# Wait for PathMNIST full-data jobs, then submit PathMNIST-C corruption evals.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

WAIT_JOB_IDS="${WAIT_JOB_IDS:?set WAIT_JOB_IDS to comma-separated job IDs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-pathmnist-corruption}"
SBATCH_BIN="${SBATCH_BIN:-sbatch.tinygpu}"
SLURM_PARTITION="${SLURM_PARTITION:-a100}"
SLURM_GRES="${SLURM_GRES:-gpu:a100:1}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
ENV_SETUP="${ENV_SETUP:-$REPO_ROOT/hpc_env.sh}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
METHODS="${METHODS:-ce-baseline,correctness-prediction,deep-correctness-prediction,loss-prediction,deep-loss-prediction,mc-dropout,edl,sngp,het,het-xl,hetclassnn,shallow-ensemble,duq}"

COMMAND_DIR="$REPO_ROOT/logs/pathmnist_hpc_${RUN_ID}"
MANIFEST="$REPO_ROOT/configs/pathmnist_checkpoint_manifest_${RUN_ID}.csv"
STATUS_FILE="$COMMAND_DIR/status.log"

mkdir -p "$COMMAND_DIR" "$REPO_ROOT/configs" "$REPO_ROOT/logs/slurm"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$STATUS_FILE"
}

wait_for_jobs() {
  IFS=',' read -r -a ids <<< "$WAIT_JOB_IDS"
  for job_id in "${ids[@]}"; do
    job_id="${job_id//[[:space:]]/}"
    [[ -z "$job_id" ]] && continue
    log "Waiting for job ${job_id} to leave the TinyGPU queue."
    while squeue.tinygpu -h -j "$job_id" | grep -q .; do
      sleep 300
    done
    log "Job ${job_id} is no longer queued."
  done
}

build_manifest() {
  log "Building checkpoint manifest ${MANIFEST}"
  python3 tools/hpc/build_pathmnist_checkpoint_manifest.py \
    --repo-root "$REPO_ROOT" \
    --job-ids "$WAIT_JOB_IDS" \
    --output "$MANIFEST" | tee -a "$STATUS_FILE"
}

submit_corruption() {
  log "Submitting PathMNIST-C corruption phase from ${MANIFEST}"
  RUN_ID="$RUN_ID" \
  PHASE=corruption \
  METHODS="$METHODS" \
  CHECKPOINT_MANIFEST="$MANIFEST" \
  DATA_ROOT="$DATA_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  SBATCH_BIN="$SBATCH_BIN" \
  SLURM_PARTITION="$SLURM_PARTITION" \
  SLURM_GRES="$SLURM_GRES" \
  SLURM_TIME="$SLURM_TIME" \
  ENV_SETUP="$ENV_SETUP" \
  bash tools/hpc/submit_pathmnist_hpc.sh | tee -a "$STATUS_FILE"
}

wait_for_jobs
build_manifest
submit_corruption
log "PathMNIST-C corruption watcher finished."
