#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$REPO_ROOT/logs/cifar10_available_full_stack_${RUN_ID}"
PID_DIR="$REPO_ROOT/pids"
STATUS_FILE="$LOG_DIR/status.tsv"
LOCK_FILE="$LOG_DIR/.queue.lock"
FAILED_FILE="$LOG_DIR/failed.tsv"

mkdir -p "$LOG_DIR" "$PID_DIR"

exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "A CIFAR-10 available full-stack queue is already running for ${RUN_ID}."
  exit 1
}

SCRIPTS=(
  "archived/cleanup_20260526-1455/tools/launchers/run_1692idyk.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_5lhhpt1y.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_5w1yjrmj.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_8ugac5sn.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_9fb4ansn.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_e6rpfaue.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_jkzjb5vz.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_k6scvi1c.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_olapo0kg.sh"
  "archived/cleanup_20260526-1455/tools/launchers/run_pb3a6oru.sh"
)

OOD_TRANSFORMS="gaussian_noise,shot_noise,impulse_noise,defocus_blur,frosted_glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic,pixelate,jpeg"
if [[ "${RUN_OOD:-0}" == "1" ]]; then
  OOD_EVAL_FLAGS="--evaluate-on-test-sets --ood-transforms-eval ${OOD_TRANSFORMS} --ood-transforms-test ${OOD_TRANSFORMS} --severities 1,2,3,4,5"
else
  OOD_EVAL_FLAGS="--discard-ood-test-sets"
fi

if [[ -f data/CIFAR10H/annotations.json ]]; then
  DATASET_ID_OVERRIDE="--dataset-id soft/cifar10"
else
  DATASET_ID_OVERRIDE="--dataset-id hard/cifar10"
fi

{
  echo "run_id	phase	script	status	exit_code	started_at	finished_at	log"
} > "$STATUS_FILE"
: > "$FAILED_FILE"

latest_checkpoint_count() {
  find checkpoints -maxdepth 2 \( -name checkpoint_best.pt -o -name checkpoint_last.pt \) 2>/dev/null | wc -l
}

extract_command() {
  local script="$1"
  awk 'NF && $1 !~ /^#/ {line=$0} END {print line}' "$script"
}

run_script_with_extra_args() {
  local phase="$1"
  local script="$2"
  local extra_args="$3"
  local name log_file start_time end_time base_cmd rc

  name="$(basename "$script" .sh)"
  log_file="$LOG_DIR/${phase}_${name}.log"

  if [[ ! -f "$script" ]]; then
    start_time="$(date --iso-8601=seconds)"
    end_time="$(date --iso-8601=seconds)"
    echo -e "${RUN_ID}\t${phase}\t${script}\tmissing\t127\t${start_time}\t${end_time}\t${log_file}" >> "$STATUS_FILE"
    return 127
  fi

  base_cmd="$(extract_command "$script")"
  if [[ -z "$base_cmd" ]]; then
    start_time="$(date --iso-8601=seconds)"
    end_time="$(date --iso-8601=seconds)"
    echo -e "${RUN_ID}\t${phase}\t${script}\tempty\t126\t${start_time}\t${end_time}\t${log_file}" >> "$STATUS_FILE"
    return 126
  fi

  start_time="$(date --iso-8601=seconds)"
  echo "[$start_time] START ${phase} ${script}"
  echo "[$start_time] checkpoint_count_before=$(latest_checkpoint_count)" > "$log_file"
  echo "[$start_time] command=${base_cmd} ${extra_args}" >> "$log_file"

  eval "${base_cmd} ${extra_args}" >> "$log_file" 2>&1
  rc=$?

  end_time="$(date --iso-8601=seconds)"
  if [[ $rc -eq 0 ]]; then
    echo "[$end_time] DONE ${phase} ${script}"
    echo -e "${RUN_ID}\t${phase}\t${script}\tdone\t0\t${start_time}\t${end_time}\t${log_file}" >> "$STATUS_FILE"
  else
    echo "[$end_time] FAIL ${phase} ${script} exit=${rc}"
    echo -e "${RUN_ID}\t${phase}\t${script}\tfailed\t${rc}\t${start_time}\t${end_time}\t${log_file}" >> "$STATUS_FILE"
    echo -e "${phase}\t${script}\t${extra_args}" >> "$FAILED_FILE"
  fi

  echo "[$end_time] checkpoint_count_after=$(latest_checkpoint_count)" >> "$log_file"
  return "$rc"
}

retry_failed_once() {
  local retry_file script phase extra_args

  retry_file="$LOG_DIR/failed.retry.tsv"
  if [[ ! -s "$FAILED_FILE" ]]; then
    echo "[$(date --iso-8601=seconds)] No failed jobs to retry."
    return 0
  fi

  cp "$FAILED_FILE" "$retry_file"
  : > "$FAILED_FILE"
  echo "[$(date --iso-8601=seconds)] RETRY START $(wc -l < "$retry_file") failed job(s)"

  while IFS=$'\t' read -r phase script extra_args; do
    run_script_with_extra_args "retry-${phase}" "$script" "$extra_args"
  done < "$retry_file"

  echo "[$(date --iso-8601=seconds)] RETRY END"
}

run_phase() {
  local phase="$1"
  local flags="$2"
  local script rc

  echo "[$(date --iso-8601=seconds)] PHASE START ${phase}"
  for script in "${SCRIPTS[@]}"; do
    run_script_with_extra_args "$phase" "$script" "$flags"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "[$(date --iso-8601=seconds)] continuing after failure in ${script}"
    fi
  done
  echo "[$(date --iso-8601=seconds)] PHASE END ${phase}"
}

echo "[$(date --iso-8601=seconds)] CIFAR-10 available full-stack queue ${RUN_ID}"
echo "Repository: ${REPO_ROOT}"
echo "Dataset-id override: ${DATASET_ID_OVERRIDE}"
echo "Methods per phase: ${#SCRIPTS[@]}"
echo "Status file: ${STATUS_FILE}"
echo "OOD mode: ${RUN_OOD:-0} (${OOD_EVAL_FLAGS})"
echo "Note: CE/correctness/MC-dropout sweeps are missing from local YAMLs; final-only methods with remote checkpoint paths are excluded."

run_phase "full-data" "--train-subset 1.0 ${OOD_EVAL_FLAGS} ${DATASET_ID_OVERRIDE}"
run_phase "scarce-50pct" "--train-subset 0.5 ${OOD_EVAL_FLAGS} ${DATASET_ID_OVERRIDE}"
run_phase "scarce-10pct" "--train-subset 0.1 ${OOD_EVAL_FLAGS} ${DATASET_ID_OVERRIDE}"
retry_failed_once

echo "[$(date --iso-8601=seconds)] Queue finished."
