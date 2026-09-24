#!/usr/bin/env bash
# Run the MHIST methods that were not part of the first CIFAR-transfer queue.
#
# Trainable: DUQ, deep correctness prediction, deep loss prediction.
# Post-hoc: fast deep ensemble from the already trained five CE members.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-mhist-extra-methods}"
LOG_DIR="$REPO_ROOT/logs/mhist_extra_methods_${RUN_ID}"
STATUS_FILE="$LOG_DIR/status.tsv"
FAILED_FILE="$LOG_DIR/failed.tsv"
LOCK_FILE="$LOG_DIR/.queue.lock"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${MHIST_DATA_DIR:-./data/MHIST}"
SOFT_LABEL_ROOT="${MHIST_SOFT_LABEL_ROOT:-./data/MHIST}"
STORAGE_DEVICE="${STORAGE_DEVICE:-cpu}"
POSTHOC_SOURCE_RUN_ID="${POSTHOC_SOURCE_RUN_ID:?set POSTHOC_SOURCE_RUN_ID to a completed MHIST post-hoc run}"

mkdir -p "$LOG_DIR"

exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "An MHIST extra-methods queue is already running for ${RUN_ID}."
  exit 1
}

COMMON_ARGS=(
  --accumulation-steps 1
  --batch-size 128
  --crop-pct 1
  --data-dir "$DATA_DIR"
  --soft-label-root "$SOFT_LABEL_ROOT"
  --dataset hard/mhist
  --dataset-id soft/mhist
  --evaluate-on-test-sets
  --discard-ood-test-sets
  --hflip 0.5
  --img-size 32
  --mean 0.4914,0.4822,0.4465
  --model-name untangle/wide_resnet_c_preact_26_10
  --momentum 0.9
  --num-classes 2
  --opt nesterov
  --padding 2
  --pin-memory
  --seed 42
  --std 0.2023,0.1994,0.2010
  --storage-device "$STORAGE_DEVICE"
)

printf 'run_id\tphase\tmethod\tstatus\texit_code\tstarted_at\tfinished_at\tlog\n' > "$STATUS_FILE"
: > "$FAILED_FILE"

record_status() {
  local phase="$1" method="$2" status="$3" rc="$4" started="$5" finished="$6" log="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$RUN_ID" "$phase" "$method" "$status" "$rc" "$started" "$finished" "$log" \
    >> "$STATUS_FILE"
}

run_job() {
  local phase="$1"
  local subset="$2"
  local method="$3"
  shift 3

  local log_file="$LOG_DIR/${phase}_${method}.log"
  local started finished rc

  started="$(date --iso-8601=seconds)"
  echo "[$started] START ${phase} ${method}"
  {
    echo "[$started] command=${PYTHON_BIN} train.py ${COMMON_ARGS[*]} --train-subset ${subset} $*"
  } > "$log_file"

  "$PYTHON_BIN" train.py \
    "${COMMON_ARGS[@]}" \
    --train-subset "$subset" \
    "$@" >> "$log_file" 2>&1
  rc=$?

  finished="$(date --iso-8601=seconds)"
  if [[ $rc -eq 0 ]]; then
    record_status "$phase" "$method" done 0 "$started" "$finished" "$log_file"
    echo "[$finished] DONE ${phase} ${method}"
  else
    record_status "$phase" "$method" failed "$rc" "$started" "$finished" "$log_file"
    printf '%s\t%s\t%s\n' "$phase" "$method" "$log_file" >> "$FAILED_FILE"
    echo "[$finished] FAIL ${phase} ${method} exit=${rc}"
  fi
}

posthoc_manifest_for_phase() {
  local phase="$1"
  local manifest="$REPO_ROOT/configs/mhist_posthoc_weight_paths_${POSTHOC_SOURCE_RUN_ID}_${phase}.env"
  if [[ ! -f "$manifest" ]]; then
    echo "Missing post-hoc manifest: ${manifest}" >&2
    return 1
  fi
  echo "$manifest"
}

run_fast_deep_ensemble() {
  local phase="$1"
  local subset="$2"
  local manifest log_file started finished rc

  manifest="$(posthoc_manifest_for_phase "$phase")" || {
    started="$(date --iso-8601=seconds)"
    record_status "$phase" fast-deep-ensemble blocked 1 "$started" "$started" \
      "Missing CE member manifest for ${phase}."
    return 1
  }

  set -a
  source "$manifest"
  set +a

  log_file="$LOG_DIR/${phase}_fast-deep-ensemble.log"
  started="$(date --iso-8601=seconds)"
  echo "[$started] START ${phase} fast-deep-ensemble"
  {
    echo "[$started] command=${PYTHON_BIN} train.py ${COMMON_ARGS[*]} --train-subset ${subset} --epochs 0 --method-name fast-deep-ensemble --weight-paths ${MHIST_BEST_WEIGHT_PATHS}"
  } > "$log_file"

  "$PYTHON_BIN" train.py \
    "${COMMON_ARGS[@]}" \
    --train-subset "$subset" \
    --epochs 0 \
    --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
    --loss cross-entropy \
    --lr 0 \
    --method-name fast-deep-ensemble \
    --sched-kwargs 'sched=none' \
    --weight-decay 0 \
    --weight-paths "$MHIST_BEST_WEIGHT_PATHS" >> "$log_file" 2>&1
  rc=$?

  finished="$(date --iso-8601=seconds)"
  if [[ $rc -eq 0 ]]; then
    record_status "$phase" fast-deep-ensemble done 0 "$started" "$finished" "$log_file"
    echo "[$finished] DONE ${phase} fast-deep-ensemble"
  else
    record_status "$phase" fast-deep-ensemble failed "$rc" "$started" "$finished" "$log_file"
    printf '%s\tfast-deep-ensemble\t%s\n' "$phase" "$log_file" >> "$FAILED_FILE"
    echo "[$finished] FAIL ${phase} fast-deep-ensemble exit=${rc}"
  fi
}

run_phase() {
  local phase="$1"
  local subset="$2"

  echo "[$(date --iso-8601=seconds)] PHASE START ${phase} subset=${subset}"

  run_job "$phase" "$subset" duq \
    --epochs 200 \
    --eval-metric id_eval_duq_values_auroc_hard_bma_correctness_original \
    --loss duq \
    --lr 0.13540695861213595 \
    --method-name duq \
    --num-hidden-features -1 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 0.0013871238024266957

  run_job "$phase" "$subset" deep-correctness-prediction \
    --epochs 200 \
    --eval-metric id_eval_error_probabilities_auroc_hard_bma_correctness_original \
    --lambda-uncertainty-loss 0.0794330872 \
    --loss correctness-prediction \
    --lr 0.1241291927 \
    --method-name deep-correctness-prediction \
    --num-hidden-features 1024 \
    --detach-uncertainty-target \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 1.1415343e-05

  run_job "$phase" "$subset" deep-loss-prediction \
    --detach-uncertainty-target \
    --epochs 200 \
    --eval-metric id_eval_loss_values_auroc_hard_bma_correctness_original \
    --lambda-uncertainty-loss 0.01 \
    --loss loss-prediction \
    --lr 0.11482920440979424 \
    --method-name deep-loss-prediction \
    --mlp-depth 3 \
    --num-hidden-features 1024 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 6.546158927128921e-05

  run_fast_deep_ensemble "$phase" "$subset"

  echo "[$(date --iso-8601=seconds)] PHASE END ${phase}"
}

echo "[$(date --iso-8601=seconds)] MHIST extra-methods queue ${RUN_ID}"
echo "Data: ${DATA_DIR}"
echo "Storage device: ${STORAGE_DEVICE}"
echo "Status file: ${STATUS_FILE}"
echo "Post-hoc CE manifest source: ${POSTHOC_SOURCE_RUN_ID}"

run_phase full-data 1.0
run_phase scarce-50pct 0.5
run_phase scarce-10pct 0.1

echo "[$(date --iso-8601=seconds)] Queue finished."
