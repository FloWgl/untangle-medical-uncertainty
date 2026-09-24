#!/usr/bin/env bash
# Retry MHIST correctness prediction with the recovered CIFAR-10 W&B cdc24149 settings.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-mhist-cp-cdc24149-retry}"
LOG_DIR="$REPO_ROOT/logs/mhist_correctness_prediction_exact_${RUN_ID}"
STATUS_FILE="$LOG_DIR/status.tsv"
FAILED_FILE="$LOG_DIR/failed.tsv"
LOCK_FILE="$LOG_DIR/.queue.lock"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${MHIST_DATA_DIR:-./data/MHIST}"
SOFT_LABEL_ROOT="${MHIST_SOFT_LABEL_ROOT:-./data/MHIST}"
STORAGE_DEVICE="${STORAGE_DEVICE:-cpu}"

mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "An MHIST correctness-prediction exact retry is already running for ${RUN_ID}."
  exit 1
}

printf 'run_id\tphase\tstatus\texit_code\tstarted_at\tfinished_at\tlog\n' > "$STATUS_FILE"
: > "$FAILED_FILE"

COMMON_ARGS=(
  --accumulation-steps 1
  --batch-size 128
  --crop-pct 1
  --data-dir "$DATA_DIR"
  --soft-label-root "$SOFT_LABEL_ROOT"
  --dataset hard/mhist
  --dataset-id soft/mhist
  --epochs 200
  --eval-metric id_eval_error_probabilities_auroc_hard_bma_correctness_original
  --evaluate-on-test-sets
  --hflip 0.5
  --img-size 32
  --lambda-uncertainty-loss 0.07943308724150058
  --loss correctness-prediction
  --lr 0.12412919266597876
  --mean 0.4914,0.4822,0.4465
  --method-name correctness-prediction
  --mlp-depth 3
  --model-name untangle/wide_resnet_c_preact_26_10
  --momentum 0.9
  --num-classes 2
  --num-hidden-features 1024
  --opt nesterov
  --padding 2
  --pin-memory
  --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1'
  --seed 42
  --std 0.2023,0.1994,0.2010
  --storage-device "$STORAGE_DEVICE"
  --weight-decay 1.1415343424883092e-05
  --detach-uncertainty-target
  --discard-ood-test-sets
)

run_phase() {
  local phase="$1"
  local subset="$2"
  local log_file="$LOG_DIR/${phase}_correctness-prediction_cdc24149.log"
  local start_time end_time rc

  start_time="$(date --iso-8601=seconds)"
  echo "[$start_time] START ${phase} subset=${subset}"
  {
    echo "[$start_time] command=${PYTHON_BIN} train.py ${COMMON_ARGS[*]} --train-subset ${subset}"
    echo "Recovered CIFAR-10 W&B source: docs/recovered_wandb/missing_trainable/azcfycns_cdc24149_config.json"
  } > "$log_file"

  "$PYTHON_BIN" train.py "${COMMON_ARGS[@]}" --train-subset "$subset" >> "$log_file" 2>&1
  rc=$?

  end_time="$(date --iso-8601=seconds)"
  if [[ $rc -eq 0 ]]; then
    echo "[$end_time] DONE ${phase}"
    printf '%s\t%s\tdone\t0\t%s\t%s\t%s\n' "$RUN_ID" "$phase" "$start_time" "$end_time" "$log_file" >> "$STATUS_FILE"
  else
    echo "[$end_time] FAIL ${phase} exit=${rc}"
    printf '%s\t%s\tfailed\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$phase" "$rc" "$start_time" "$end_time" "$log_file" >> "$STATUS_FILE"
    printf '%s\t%s\n' "$phase" "$log_file" >> "$FAILED_FILE"
  fi
}

echo "[$(date --iso-8601=seconds)] MHIST correctness-prediction exact retry ${RUN_ID}"
echo "Recovered source: docs/recovered_wandb/missing_trainable/azcfycns_cdc24149_config.json"
echo "Status file: ${STATUS_FILE}"

run_phase full-data 1.0
run_phase scarce-50pct 0.5
run_phase scarce-10pct 0.1

echo "[$(date --iso-8601=seconds)] Queue finished."
