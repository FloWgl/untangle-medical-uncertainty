#!/usr/bin/env bash
# Generate and submit a PathMNIST SLURM job array.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PHASE="${PHASE:-train}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-pathmnist-${PHASE}}"
COMMAND_DIR="${COMMAND_DIR:-$REPO_ROOT/logs/pathmnist_hpc_${RUN_ID}}"
CMD_FILE="${CMD_FILE:-$COMMAND_DIR/commands.txt}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
SLURM_GRES="${SLURM_GRES:-gpu:1}"
SLURM_TIME="${SLURM_TIME:-48:00:00}"

mkdir -p "$COMMAND_DIR" logs/slurm checkpoints results

GENERATOR_ARGS=(
  --phase "$PHASE"
  --output "$CMD_FILE"
)

if [[ -n "${CHECKPOINT_MANIFEST:-}" ]]; then
  GENERATOR_ARGS+=(--checkpoint-manifest "$CHECKPOINT_MANIFEST")
fi

if [[ -n "${DATA_ROOT:-}" ]]; then
  GENERATOR_ARGS+=(
    --pathmnist-dir "$DATA_ROOT/pathmnist_cache/pathmnist"
    --pathmnist-c-dir "$DATA_ROOT/pathmnist_c/pathmnist"
  )
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  GENERATOR_ARGS+=(--python "$PYTHON_BIN")
fi

if [[ -n "${METHODS:-}" ]]; then
  GENERATOR_ARGS+=(--methods "$METHODS")
fi

if [[ "${SKIP_ENSEMBLE_SEEDS:-0}" == "1" ]]; then
  GENERATOR_ARGS+=(--skip-ensemble-seeds)
fi

if [[ -n "${EPOCHS:-}" ]]; then
  GENERATOR_ARGS+=(--epochs "$EPOCHS")
fi

if [[ -n "${BATCH_SIZE:-}" ]]; then
  GENERATOR_ARGS+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "${ACCUMULATION_STEPS:-}" ]]; then
  GENERATOR_ARGS+=(--accumulation-steps "$ACCUMULATION_STEPS")
fi

if [[ -n "${LABEL_NOISE_FRACTIONS:-}" ]]; then
  # shellcheck disable=SC2206
  LABEL_NOISE_FRACTION_ARRAY=($LABEL_NOISE_FRACTIONS)
  GENERATOR_ARGS+=(--label-noise-fractions "${LABEL_NOISE_FRACTION_ARRAY[@]}")
fi

if [[ -n "${LABEL_NOISE_SEED:-}" ]]; then
  GENERATOR_ARGS+=(--label-noise-seed "$LABEL_NOISE_SEED")
fi

if [[ -n "${CORRUPTION_SEVERITIES:-}" ]]; then
  # shellcheck disable=SC2206
  CORRUPTION_SEVERITY_ARRAY=($CORRUPTION_SEVERITIES)
  GENERATOR_ARGS+=(--corruption-severities "${CORRUPTION_SEVERITY_ARRAY[@]}")
fi

if [[ -n "${STORAGE_DEVICE:-}" ]]; then
  GENERATOR_ARGS+=(--storage-device "$STORAGE_DEVICE")
fi

if [[ "${AMP:-0}" == "1" ]]; then
  GENERATOR_ARGS+=(--amp)
fi

if [[ "${PIN_MEMORY:-1}" == "1" ]]; then
  GENERATOR_ARGS+=(--pin-memory)
fi

if [[ "${LOG_WANDB:-0}" == "1" ]]; then
  GENERATOR_ARGS+=(--log-wandb)
fi

"${GENERATOR_PYTHON:-python3}" tools/hpc/generate_pathmnist_hpc_commands.py "${GENERATOR_ARGS[@]}"

NUM_TASKS="$(wc -l < "$CMD_FILE")"
if [[ "$NUM_TASKS" -lt 1 ]]; then
  echo "No commands generated for PHASE=${PHASE}."
  exit 4
fi

echo "Submitting ${NUM_TASKS} PathMNIST ${PHASE} tasks from ${CMD_FILE}"
SBATCH_ARGS=(
  --job-name "pathmnist-${PHASE}"
  --array "1-${NUM_TASKS}"
  --gres "$SLURM_GRES"
  --time "$SLURM_TIME"
  --export "ALL,CMD_FILE=${CMD_FILE},WORK_DIR=${REPO_ROOT}"
)

if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_ARGS+=(--partition "$SLURM_PARTITION")
fi

"$SBATCH_BIN" \
  "${SBATCH_ARGS[@]}" \
  tools/hpc/untangle_job_array.sbatch
