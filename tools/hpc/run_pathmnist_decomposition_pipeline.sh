#!/usr/bin/env bash
# Submit PathMNIST reruns and export portable IT/Bregman result tables.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)-pathmnist-decomposition}"
METHODS="${METHODS:-ce-baseline,correctness-prediction,deep-correctness-prediction,loss-prediction,deep-loss-prediction,mc-dropout,edl,sngp,het,het-xl,hetclassnn,shallow-ensemble,duq,postnet}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"
SKIP_ENSEMBLE_SEEDS="${SKIP_ENSEMBLE_SEEDS:-1}"

LOG_DIR="$REPO_ROOT/logs/pathmnist_hpc_${RUN_STAMP}"
STATUS_FILE="$LOG_DIR/status.log"
AUDIT_CSV="$REPO_ROOT/docs/pathmnist_audit/pathmnist_decomposition_rerun_${RUN_STAMP}.csv"
CHECKPOINT_MANIFEST="$REPO_ROOT/configs/pathmnist_decomposition_rerun_${RUN_STAMP}_checkpoints.csv"

mkdir -p "$LOG_DIR" logs/slurm configs docs/pathmnist_audit docs/results/pathmnist

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$STATUS_FILE"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 is required. Run this script on the HPC login node where SLURM is available." >&2
    exit 127
  fi
}

submit_phase() {
  local phase="$1"
  local run_id="$2"
  local output
  log "Submitting phase=${phase} run_id=${run_id}" >&2
  output="$(
    RUN_ID="$run_id" \
    PHASE="$phase" \
    METHODS="$METHODS" \
    DATA_ROOT="$DATA_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    SLURM_TIME="$SLURM_TIME" \
    SKIP_ENSEMBLE_SEEDS="$SKIP_ENSEMBLE_SEEDS" \
    CHECKPOINT_MANIFEST="${CHECKPOINT_MANIFEST:-}" \
    bash tools/hpc/submit_pathmnist_hpc.sh 2>&1
  )"
  printf '%s\n' "$output" | tee -a "$STATUS_FILE" >&2
  local job_id
  job_id="$(printf '%s\n' "$output" | sed -n 's/Submitted batch job //p' | tail -1)"
  if [[ -z "$job_id" ]]; then
    echo "Could not parse SLURM job id for phase=${phase}" >&2
    exit 3
  fi
  printf '%s' "$job_id"
}

wait_for_job() {
  local job_id="$1"
  log "Waiting for job ${job_id}"
  while true; do
    if squeue -j "$job_id" -h >/tmp/pathmnist_squeue_${job_id}.txt 2>/dev/null; then
      if [[ ! -s "/tmp/pathmnist_squeue_${job_id}.txt" ]]; then
        break
      fi
    fi
    sleep 120
  done
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$job_id" --format=JobID,State,ExitCode,Elapsed -P | tee -a "$STATUS_FILE" || true
  fi
}

require_command sbatch
require_command squeue

log "Starting PathMNIST IT+Bregman decomposition pipeline."
log "methods=${METHODS}"
log "data_root=${DATA_ROOT}"

TRAIN_JOB_ID="$(submit_phase train "${RUN_STAMP}-train")"
REDUCED_JOB_ID="$(submit_phase reduced "${RUN_STAMP}-reduced")"

wait_for_job "$TRAIN_JOB_ID"
wait_for_job "$REDUCED_JOB_ID"

log "Collecting fresh audit CSV."
"$PYTHON_BIN" tools/hpc/collect_pathmnist_decomposition_audit.py \
  --repo-root "$REPO_ROOT" \
  --job-ids "${TRAIN_JOB_ID},${REDUCED_JOB_ID}" \
  --output "$AUDIT_CSV" | tee -a "$STATUS_FILE"

log "Building full-data checkpoint manifest for corruption evaluation."
"$PYTHON_BIN" tools/hpc/build_pathmnist_checkpoint_manifest.py \
  --repo-root "$REPO_ROOT" \
  --job-ids "$TRAIN_JOB_ID" \
  --output "$CHECKPOINT_MANIFEST" | tee -a "$STATUS_FILE"

CORRUPTION_JOB_ID="$(
CHECKPOINT_MANIFEST="$CHECKPOINT_MANIFEST" \
  submit_phase corruption "${RUN_STAMP}-corruption"
)"
wait_for_job "$CORRUPTION_JOB_ID"

log "Summarizing IT and Bregman decomposition tensors."
"$PYTHON_BIN" tools/analyze_pathmnist_hpc_results.py \
  --repo-root "$REPO_ROOT" \
  --audit-csv "$AUDIT_CSV" \
  --out-dir "$REPO_ROOT/docs/results/pathmnist" | tee -a "$STATUS_FILE"

log "Done. Audit: $AUDIT_CSV"
log "Checkpoint manifest: $CHECKPOINT_MANIFEST"
log "Result tables: $REPO_ROOT/docs/results/pathmnist"
