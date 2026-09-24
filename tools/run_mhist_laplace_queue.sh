#!/usr/bin/env bash
# Run MHIST Laplace post-hoc evaluations from the trained CE member checkpoints.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

SOURCE_RUN_ID="${SOURCE_RUN_ID:?set SOURCE_RUN_ID to a completed MHIST post-hoc run}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-mhist-laplace}"
LOG_DIR="$REPO_ROOT/logs/mhist_laplace_${RUN_ID}"
STATUS_FILE="$LOG_DIR/status.tsv"
FAILED_FILE="$LOG_DIR/failed.tsv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${MHIST_DATA_DIR:-./data/MHIST}"
SOFT_LABEL_ROOT="${MHIST_SOFT_LABEL_ROOT:-./data/MHIST}"
STORAGE_DEVICE="${STORAGE_DEVICE:-cpu}"

mkdir -p "$LOG_DIR"
printf 'run_id\tphase\tstage\tstatus\texit_code\tstarted_at\tfinished_at\tlog\n' > "$STATUS_FILE"
: > "$FAILED_FILE"

record_status() {
  local phase="$1" stage="$2" status="$3" rc="$4" started="$5" finished="$6" log="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$RUN_ID" "$phase" "$stage" "$status" "$rc" "$started" "$finished" "$log" \
    >> "$STATUS_FILE"
}

first_best_checkpoint() {
  local phase="$1"
  local manifest="$REPO_ROOT/configs/mhist_posthoc_weight_paths_${SOURCE_RUN_ID}_${phase}.env"
  set -a
  source "$manifest"
  set +a
  echo "${MHIST_BEST_WEIGHT_PATHS%%,*}"
}

run_phase() {
  local phase="$1" subset="$2"
  local weight_path log_file started finished rc
  weight_path="$(first_best_checkpoint "$phase")"
  log_file="$LOG_DIR/${phase}_laplace.log"
  started="$(date --iso-8601=seconds)"
  echo "[$started] START ${phase} laplace weight=${weight_path}"

  "$PYTHON_BIN" train.py \
    --accumulation-steps 1 \
    --batch-size 128 \
    --crop-pct 1 \
    --data-dir "$DATA_DIR" \
    --soft-label-root "$SOFT_LABEL_ROOT" \
    --dataset hard/mhist \
    --dataset-id soft/mhist \
    --discard-ood-test-sets \
    --evaluate-on-test-sets \
    --epochs 0 \
    --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
    --hessian-structure kron \
    --hflip 0.5 \
    --img-size 32 \
    --log-prior-precision-max 2 \
    --log-prior-precision-min -1 \
    --loss cross-entropy \
    --lr 0 \
    --mean 0.4914,0.4822,0.4465 \
    --method-name laplace \
    --model-name untangle/wide_resnet_c_preact_26_10 \
    --momentum 0.9 \
    --num-classes 2 \
    --num-mc-samples 1000 \
    --num-mc-samples-cv 50 \
    --opt nesterov \
    --padding 2 \
    --pin-memory \
    --pred-type glm \
    --prior-precision-grid-size 80 \
    --sched-kwargs 'sched=none' \
    --seed 42 \
    --std 0.2023,0.1994,0.2010 \
    --storage-device "$STORAGE_DEVICE" \
    --train-subset "$subset" \
    --weight-decay 0.004022874547456138 \
    --weight-paths "$weight_path" > "$log_file" 2>&1
  rc=$?
  finished="$(date --iso-8601=seconds)"

  if [[ $rc -eq 0 ]]; then
    record_status "$phase" laplace done 0 "$started" "$finished" "$log_file"
    echo "[$finished] DONE ${phase} laplace"
  else
    record_status "$phase" laplace failed "$rc" "$started" "$finished" "$log_file"
    printf '%s\tlaplace\t%s\n' "$phase" "$log_file" >> "$FAILED_FILE"
    echo "[$finished] FAIL ${phase} laplace exit=${rc}"
  fi
}

echo "[$(date --iso-8601=seconds)] MHIST Laplace queue ${RUN_ID}"
echo "Source manifests: ${SOURCE_RUN_ID}"
echo "Using one CE member per split, hessian=kron, pred-type=glm, MC=1000, CV MC=50, prior grid=80."

run_phase full-data 1.0
run_phase scarce-50pct 0.5
run_phase scarce-10pct 0.1

echo "[$(date --iso-8601=seconds)] MHIST Laplace queue finished."
