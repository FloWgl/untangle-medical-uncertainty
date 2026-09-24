#!/usr/bin/env bash
#SBATCH --job-name=untangle
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

if [[ -z "${CMD_FILE:-}" ]]; then
  echo "CMD_FILE must point to a text file with one command per line."
  exit 2
fi

if [[ -n "${WORK_DIR:-}" ]]; then
  cd "$WORK_DIR"
fi

mkdir -p logs/slurm checkpoints results

if [[ -n "${ENV_SETUP:-}" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_SETUP"
fi

if [[ -n "${ENV_COMMAND:-}" ]]; then
  eval "$ENV_COMMAND"
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  TASK_ID="$SLURM_ARRAY_TASK_ID"
else
  TASK_ID=1
fi

CMD="$(sed -n "${TASK_ID}p" "$CMD_FILE")"
if [[ -z "$CMD" ]]; then
  echo "No command found at line ${TASK_ID} in ${CMD_FILE}."
  exit 3
fi

echo "Host: $(hostname)"
echo "Date: $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"
echo "Command file: ${CMD_FILE}"
echo "Task id: ${TASK_ID}"
echo "Command: ${CMD}"

# Site-specific module/conda setup belongs in the submitted environment or above this line.
eval "$CMD"
