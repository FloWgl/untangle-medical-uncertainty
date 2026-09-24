#!/usr/bin/env bash
# Submit paper-style PathMNIST post-hoc methods from a CE/SNGP env manifest.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="${1:?usage: $0 configs/pathmnist_posthoc_weight_paths.env}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-pathmnist-posthoc-paperstyle}"
SBATCH_BIN="${SBATCH_BIN:-sbatch.tinygpu}"
SLURM_PARTITION="${SLURM_PARTITION:-a100}"
SLURM_GRES="${SLURM_GRES:-gpu:a100:1}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
ENV_SETUP="${ENV_SETUP:-$REPO_ROOT/hpc_env.sh}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the dataset root}"
PYTHON_BIN="${PYTHON_BIN:-python}"

COMMAND_DIR="$REPO_ROOT/logs/pathmnist_hpc_${RUN_ID}"
CMD_FILE="$COMMAND_DIR/commands.txt"
STATUS_FILE="$COMMAND_DIR/status.log"
mkdir -p "$COMMAND_DIR" "$REPO_ROOT/logs/slurm"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "$STATUS_FILE"
}

# shellcheck disable=SC1090
source "$MANIFEST"

: "${PATHMNIST_BEST_WEIGHT_PATHS:?missing PATHMNIST_BEST_WEIGHT_PATHS in manifest}"
: "${PATHMNIST_LAST_WEIGHT_PATHS:?missing PATHMNIST_LAST_WEIGHT_PATHS in manifest}"
: "${PATHMNIST_SNGP_BEST_WEIGHT_PATH:?missing PATHMNIST_SNGP_BEST_WEIGHT_PATH in manifest}"

common_args() {
  printf '%q ' \
    "$PYTHON_BIN" train.py \
    --dataset hard/pathmnist \
    --dataset-id hard/pathmnist \
    --data-dir "$DATA_ROOT/pathmnist_cache/pathmnist" \
    --data-dir-id "$DATA_ROOT/pathmnist_cache/pathmnist" \
    --dataset-download \
    --model-name timm/resnet_50 \
    --num-classes 9 \
    --opt adamw \
    --weight-decay 2e-5 \
    --lr-base 0.001 \
    --batch-size 128 \
    --accumulation-steps 16 \
    --epochs 0 \
    --seed 42 \
    --img-size 224 \
    --mean 0.485,0.456,0.406 \
    --std 0.229,0.224,0.225 \
    --train-subset 1.0 \
    --evaluate-on-test-sets \
    --storage-device cpu \
    --num-workers 8 \
    --num-eval-workers 8 \
    --loss cross-entropy \
    --discard-ood-test-sets \
    --pin-memory
}

write_eval_command() {
  local method="$1"
  local weights="$2"
  shift 2
  {
    common_args
    printf '%q ' --method-name "$method" --weight-paths "$weights" "$@"
    printf '\n'
  } >> "$CMD_FILE"
}

write_swag_command() {
  local weight="$1"
  {
    printf '%q ' \
      "$PYTHON_BIN" train.py \
      --dataset hard/pathmnist \
      --dataset-id hard/pathmnist \
      --data-dir "$DATA_ROOT/pathmnist_cache/pathmnist" \
      --data-dir-id "$DATA_ROOT/pathmnist_cache/pathmnist" \
      --dataset-download \
      --model-name timm/resnet_50 \
      --num-classes 9 \
      --opt momentum \
      --momentum 0.9 \
      --weight-decay 0.0001 \
      --lr 0.01 \
      --sched-kwargs sched=none \
      --batch-size 128 \
      --accumulation-steps 1 \
      --epochs 40 \
      --seed 42 \
      --img-size 224 \
      --mean 0.485,0.456,0.406 \
      --std 0.229,0.224,0.225 \
      --train-subset 1.0 \
      --evaluate-on-test-sets \
      --storage-device cpu \
      --num-workers 8 \
      --num-eval-workers 8 \
      --loss cross-entropy \
      --method-name swag \
      --use-low-rank-cov \
      --max-rank 20 \
      --num-checkpoints-per-epoch 1 \
      --num-mc-samples 30 \
      --weight-paths "$weight" \
      --discard-ood-test-sets \
      --pin-memory
    printf '\n'
  } >> "$CMD_FILE"
}

: > "$CMD_FILE"

write_eval_command deep-ensemble "$PATHMNIST_BEST_WEIGHT_PATHS" \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original

IFS=',' read -r -a best_array <<< "$PATHMNIST_BEST_WEIGHT_PATHS"
IFS=',' read -r -a last_array <<< "$PATHMNIST_LAST_WEIGHT_PATHS"

for weight in "${best_array[@]}"; do
  write_eval_command temperature-scaling "$weight" \
    --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
    --use-temperature-scaling
done

for weight in "${last_array[@]}"; do
  write_eval_command mahalanobis "$weight" \
    --eval-metric id_eval_mahalanobis_values_auroc_hard_bma_correctness_original \
    --magnitude 0.001
done

write_eval_command ddu "$PATHMNIST_SNGP_BEST_WEIGHT_PATH" \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
  --use-spectral-normalization \
  --spectral-normalization-iteration 1 \
  --spectral-normalization-bound 6 \
  --use-spectral-normalized-batch-norm

for weight in "${best_array[@]}"; do
  write_eval_command laplace "$weight" \
    --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
    --hessian-structure kron \
    --num-mc-samples 1000 \
    --num-mc-samples-cv 1000 \
    --pred-type glm
done

for weight in "${last_array[@]}"; do
  write_swag_command "$weight"
done

num_tasks="$(wc -l < "$CMD_FILE")"
log "Wrote ${num_tasks} PathMNIST post-hoc commands to ${CMD_FILE}"
log "Submitting PathMNIST post-hoc array."
"$SBATCH_BIN" \
  --job-name pathmnist-posthoc \
  --array "1-${num_tasks}" \
  --gres "$SLURM_GRES" \
  --time "$SLURM_TIME" \
  --partition "$SLURM_PARTITION" \
  --export "ALL,CMD_FILE=${CMD_FILE},WORK_DIR=${REPO_ROOT},ENV_SETUP=${ENV_SETUP}" \
  tools/hpc/untangle_job_array.sbatch | tee -a "$STATUS_FILE"

log "PathMNIST post-hoc submission finished."
