#!/usr/bin/env bash
# Submit the remaining feasible PathMNIST experiment pipeline on TinyGPU.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SBATCH_BIN="${SBATCH_BIN:-sbatch.tinygpu}"
SLURM_PARTITION="${SLURM_PARTITION:-a100}"
SLURM_GRES="${SLURM_GRES:-gpu:a100:1}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
ENV_SETUP="${ENV_SETUP:-$REPO_ROOT/hpc_env.sh}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Existing jobs launched on 2026-07-14. Override if this script is reused.
SHAPEFIX_JOB_ID="${SHAPEFIX_JOB_ID:-1745138}"
POSTHOC_BASE_JOB_ID="${POSTHOC_BASE_JOB_ID:-1745150}"

MISSING_FULL_METHODS="${MISSING_FULL_METHODS:-correctness-prediction,deep-correctness-prediction,loss-prediction,deep-loss-prediction,mc-dropout,het,het-xl,duq}"
SCARCITY_METHODS="${SCARCITY_METHODS:-ce-baseline,correctness-prediction,deep-correctness-prediction,loss-prediction,deep-loss-prediction,mc-dropout,edl,sngp,het,het-xl,hetclassnn,shallow-ensemble,duq}"
CORRUPTION_METHODS="${CORRUPTION_METHODS:-ce-baseline,correctness-prediction,deep-correctness-prediction,loss-prediction,deep-loss-prediction,mc-dropout,edl,sngp,het,het-xl,hetclassnn,shallow-ensemble,duq}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$REPO_ROOT/logs/pathmnist_hpc_${RUN_STAMP}-remaining-pipeline"
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status.log"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$STATUS_FILE"
}

submit_phase() {
  local phase="$1"
  local run_id="$2"
  local methods="$3"
  local skip_ensemble="$4"
  local output

  log "Submitting ${phase} run ${run_id}: ${methods}"
  output="$(
    RUN_ID="$run_id" \
    PHASE="$phase" \
    METHODS="$methods" \
    SKIP_ENSEMBLE_SEEDS="$skip_ensemble" \
    DATA_ROOT="$DATA_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    SBATCH_BIN="$SBATCH_BIN" \
    SLURM_PARTITION="$SLURM_PARTITION" \
    SLURM_GRES="$SLURM_GRES" \
    SLURM_TIME="$SLURM_TIME" \
    ENV_SETUP="$ENV_SETUP" \
    bash tools/hpc/submit_pathmnist_hpc.sh
  )"
  printf '%s\n' "$output" | tee -a "$STATUS_FILE"
  local job_id
  job_id="$(printf '%s\n' "$output" | sed -n 's/Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1)"
  if [[ -z "$job_id" ]]; then
    log "Could not parse job id for ${phase} run ${run_id}."
    return 1
  fi
  printf '%s\n' "$job_id"
}

MISSING_FULL_JOB_ID="$(
  submit_phase train "${RUN_STAMP}-pathmnist-missing-full" "$MISSING_FULL_METHODS" 1 | tail -1
)"
SCARCITY_JOB_ID="$(
  submit_phase reduced "${RUN_STAMP}-pathmnist-scarcity" "$SCARCITY_METHODS" 1 | tail -1
)"

CORRUPTION_WAIT_JOB_IDS="${CORRUPTION_WAIT_JOB_IDS:-${MISSING_FULL_JOB_ID},${SHAPEFIX_JOB_ID},${POSTHOC_BASE_JOB_ID},1676043}"
CORRUPTION_RUN_ID="${RUN_STAMP}-pathmnist-corruption"
CORRUPTION_LOG_DIR="$REPO_ROOT/logs/pathmnist_hpc_${CORRUPTION_RUN_ID}"
mkdir -p "$CORRUPTION_LOG_DIR"

log "Starting detached corruption watcher for jobs ${CORRUPTION_WAIT_JOB_IDS}."
nohup env \
  WAIT_JOB_IDS="$CORRUPTION_WAIT_JOB_IDS" \
  RUN_ID="$CORRUPTION_RUN_ID" \
  METHODS="$CORRUPTION_METHODS" \
  DATA_ROOT="$DATA_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  SBATCH_BIN="$SBATCH_BIN" \
  SLURM_PARTITION="$SLURM_PARTITION" \
  SLURM_GRES="$SLURM_GRES" \
  SLURM_TIME="$SLURM_TIME" \
  ENV_SETUP="$ENV_SETUP" \
  bash tools/hpc/watch_and_submit_pathmnist_corruption.sh \
  > "$CORRUPTION_LOG_DIR/watcher.nohup.log" 2>&1 &
WATCHER_PID="$!"
log "Corruption watcher PID: ${WATCHER_PID}"

cat <<EOF_SUMMARY | tee -a "$STATUS_FILE"
Pipeline submitted.
missing_full_job_id=${MISSING_FULL_JOB_ID}
scarcity_job_id=${SCARCITY_JOB_ID}
shapefix_job_id=${SHAPEFIX_JOB_ID}
posthoc_base_job_id=${POSTHOC_BASE_JOB_ID}
posthoc_watcher_log=$REPO_ROOT/logs/pathmnist_hpc_20260714-pathmnist-posthoc-paperstyle/status.log
corruption_watcher_log=$CORRUPTION_LOG_DIR/status.log
pipeline_status_log=$STATUS_FILE

Blocked outside this TinyGPU pipeline:
- Exact full 90-epoch PostNet, because the 24h TinyGPU wall-time limit timed out the run.
EOF_SUMMARY
