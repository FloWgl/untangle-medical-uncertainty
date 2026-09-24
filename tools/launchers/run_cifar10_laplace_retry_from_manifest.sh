#!/usr/bin/env bash
# Retry only CIFAR-10 Laplace from the CE-member manifest.

set -u

MANIFEST="${1:-configs/cifar10_posthoc_weight_paths.env}"
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

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-cifar10-laplace-retry}"
LOG_DIR="${LOG_DIR:-logs/cifar10_laplace_retry_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status.tsv"
printf "run_id\tstage\tstatus\texit_code\tstarted_at\tfinished_at\tlog\n" > "$STATUS_FILE"

started="$(date --iso-8601=seconds)"
log_path="$LOG_DIR/laplace.log"
echo "[$started] START laplace"
"$PYTHON_BIN" train.py \
  --accumulation-steps 1 \
  --batch-size 128 \
  --crop-pct 1 \
  --data-dir ./data \
  --dataset hard/cifar10 \
  --dataset-id soft/cifar10 \
  --epochs 0 \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
  --evaluate-on-test-sets \
  --hessian-structure kron \
  --hflip 0.5 \
  --img-size 32 \
  --loss cross-entropy \
  --lr 0 \
  --mean 0.4914,0.4822,0.4465 \
  --method-name laplace \
  --model-name untangle/wide_resnet_c_preact_26_10 \
  --momentum 0.9 \
  --num-classes 10 \
  --num-mc-samples 1000 \
  --num-mc-samples-cv 1000 \
  --opt nesterov \
  --padding 2 \
  --pin-memory \
  --pred-type glm \
  --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0 warmup_epochs=0' \
  --seed 42 \
  --std 0.2023,0.1994,0.2010 \
  --storage-device cuda \
  --weight-decay 0.004022874547456138 \
  --weight-paths "$CIFAR10_BEST_WEIGHT_PATHS" \
  "$@" > "$log_path" 2>&1
exit_code=$?
finished="$(date --iso-8601=seconds)"

if [[ "$exit_code" -eq 0 ]]; then
  status=done
  echo "[$finished] DONE laplace"
else
  status=failed
  echo "[$finished] FAILED laplace (exit $exit_code); see $log_path"
fi

printf "%s\tlaplace\t%s\t%s\t%s\t%s\t%s\n" \
  "$RUN_ID" "$status" "$exit_code" "$started" "$finished" "$log_path" >> "$STATUS_FILE"
exit "$exit_code"
