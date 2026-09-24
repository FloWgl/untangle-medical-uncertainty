#!/usr/bin/env bash
# Resume the CIFAR-10 paper-style post-hoc queue after Deep Ensemble and
# Temperature Scaling have already finished.
set -u

MANIFEST="${1:-configs/cifar10_posthoc_weight_paths_20260528-132834-ce-members-posthoc.env}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST"
  exit 2
fi

set -a
source "$MANIFEST"
set +a

if [[ -z "${CIFAR10_BEST_WEIGHT_PATHS:-}" ]]; then
  echo "CIFAR10_BEST_WEIGHT_PATHS is empty in $MANIFEST"
  exit 3
fi

if [[ -z "${CIFAR10_LAST_WEIGHT_PATHS:-}" ]]; then
  echo "CIFAR10_LAST_WEIGHT_PATHS is empty in $MANIFEST"
  exit 4
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-cifar10-posthoc-remaining}"
LOG_DIR="${LOG_DIR:-logs/cifar10_posthoc_remaining_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status.tsv"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf "run_id\tstage\tstatus\texit_code\tstarted_at\tfinished_at\tlog\n" > "$STATUS_FILE"
fi

COMMON_ARGS=(
  --accumulation-steps 1
  --batch-size 128
  --crop-pct 1
  --data-dir ./data
  --dataset hard/cifar10
  --dataset-id soft/cifar10
  --epochs 0
  --evaluate-on-test-sets
  --hflip 0.5
  --img-size 32
  --loss cross-entropy
  --lr 0
  --mean 0.4914,0.4822,0.4465
  --model-name untangle/wide_resnet_c_preact_26_10
  --momentum 0.9
  --num-classes 10
  --opt nesterov
  --padding 2
  --pin-memory
  --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0 warmup_epochs=0'
  --seed 42
  --std 0.2023,0.1994,0.2010
  --storage-device cuda
  --weight-decay 0.004022874547456138
)

record_status() {
  local stage="$1"
  local status="$2"
  local exit_code="$3"
  local started="$4"
  local finished="$5"
  local log="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$RUN_ID" "$stage" "$status" "$exit_code" "$started" "$finished" "$log" >> "$STATUS_FILE"
}

run_stage() {
  local stage="$1"
  shift
  local started finished exit_code log_path
  started="$(date --iso-8601=seconds)"
  log_path="$LOG_DIR/${stage}.log"
  echo "[$started] START $stage"
  "$@" > "$log_path" 2>&1
  exit_code=$?
  finished="$(date --iso-8601=seconds)"
  if [[ "$exit_code" -eq 0 ]]; then
    record_status "$stage" done "$exit_code" "$started" "$finished" "$log_path"
    echo "[$finished] DONE $stage"
  else
    record_status "$stage" failed "$exit_code" "$started" "$finished" "$log_path"
    echo "[$finished] FAILED $stage (exit $exit_code); see $log_path"
  fi
  return 0
}

run_stage mahalanobis \
  "$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" \
    --method-name mahalanobis \
    --eval-metric id_eval_mahalanobis_values_auroc_hard_bma_correctness_original \
    --magnitude 0.001 \
    --module-name-regex '^(layer[1-3].0.act1|act)$' \
    --weight-paths "$CIFAR10_LAST_WEIGHT_PATHS" "$@"

run_stage laplace \
  "$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" \
    --method-name laplace \
    --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
    --hessian-structure kron \
    --num-mc-samples 1000 \
    --num-mc-samples-cv 1000 \
    --pred-type glm \
    --weight-paths "$CIFAR10_BEST_WEIGHT_PATHS" "$@"

run_stage swag \
  "$PYTHON_BIN" train.py \
    --accumulation-steps 1 \
    --batch-size 128 \
    --crop-pct 1 \
    --data-dir ./data \
    --dataset hard/cifar10 \
    --dataset-id soft/cifar10 \
    --epochs 40 \
    --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
    --evaluate-on-test-sets \
    --hflip 0.5 \
    --img-size 32 \
    --loss cross-entropy \
    --lr 0.01 \
    --max-rank 20 \
    --mean 0.4914,0.4822,0.4465 \
    --method-name swag \
    --model-name untangle/wide_resnet_c_preact_26_10 \
    --momentum 0.9 \
    --num-checkpoints-per-epoch 1 \
    --num-classes 10 \
    --num-mc-samples 30 \
    --opt momentum \
    --padding 2 \
    --pin-memory \
    --sched-kwargs sched=none \
    --seed 42 \
    --std 0.2023,0.1994,0.2010 \
    --storage-device cuda \
    --use-low-rank-cov \
    --weight-decay 0.0001 \
    --weight-paths "$CIFAR10_LAST_WEIGHT_PATHS" "$@"

echo "[$(date --iso-8601=seconds)] CIFAR-10 remaining post-hoc resume finished."
