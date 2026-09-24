#!/usr/bin/env bash
# Sequential queue for CIFAR-10 pieces that were missing from the available queue.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="$REPO_ROOT/logs/cifar10_missing_stack_${RUN_ID}"
STATUS_FILE="$LOG_DIR/status.tsv"
mkdir -p "$LOG_DIR" pids

OOD_TRANSFORMS="gaussian_noise,shot_noise,impulse_noise,defocus_blur,frosted_glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic,pixelate,jpeg"
if [[ "${RUN_OOD:-0}" == "1" ]]; then
  OOD_FLAGS="--evaluate-on-test-sets --ood-transforms-eval ${OOD_TRANSFORMS} --ood-transforms-test ${OOD_TRANSFORMS} --severities 1,2,3,4,5"
else
  OOD_FLAGS="--discard-ood-test-sets"
fi

if [[ -f data/CIFAR10H/annotations.json ]]; then
  DATASET_ID_OVERRIDE="--dataset-id soft/cifar10"
else
  DATASET_ID_OVERRIDE="--dataset-id hard/cifar10"
fi

SCRIPTS=(
  "tools/launchers/run_cedfgnqz_reconstructed.sh"
  "tools/launchers/run_azcfycns_reconstructed.sh"
  "tools/launchers/run_k4xc00mf_reconstructed.sh"
)

echo -e "run_id\tphase\tscript\tstatus\texit_code\tstarted_at\tfinished_at\tlog" > "$STATUS_FILE"

run_one() {
  local phase="$1"
  local script="$2"
  local flags="$3"
  local started finished log_file rc

  log_file="$LOG_DIR/${phase}_$(basename "$script" .sh).log"
  started="$(date --iso-8601=seconds)"
  echo "[$started] START $phase $script"
  echo "[$started] command=bash $script $flags" > "$log_file"
  bash "$script" $flags >> "$log_file" 2>&1
  rc=$?
  finished="$(date --iso-8601=seconds)"

  if [[ $rc -eq 0 ]]; then
    echo "[$finished] DONE $phase $script"
    echo -e "${RUN_ID}\t${phase}\t${script}\tdone\t0\t${started}\t${finished}\t${log_file}" >> "$STATUS_FILE"
  else
    echo "[$finished] FAIL $phase $script exit=$rc"
    echo -e "${RUN_ID}\t${phase}\t${script}\tfailed\t${rc}\t${started}\t${finished}\t${log_file}" >> "$STATUS_FILE"
  fi
}

run_phase() {
  local phase="$1"
  local subset="$2"
  local script
  local flags="--train-subset ${subset} ${OOD_FLAGS} ${DATASET_ID_OVERRIDE}"

  for script in "${SCRIPTS[@]}"; do
    run_one "$phase" "$script" "$flags"
  done
}

echo "[$(date --iso-8601=seconds)] CIFAR-10 missing-stack queue ${RUN_ID}"
echo "OOD mode: ${RUN_OOD:-0} (${OOD_FLAGS})"
echo "Dataset-id override: ${DATASET_ID_OVERRIDE}"
echo "Status file: ${STATUS_FILE}"
echo "Note: these launchers are reconstructed because cedfgnqz/azcfycns/k4xc00mf YAMLs are absent locally."

run_phase "full-data" "1.0"
run_phase "scarce-50pct" "0.5"
run_phase "scarce-10pct" "0.1"

echo "[$(date --iso-8601=seconds)] Missing-stack queue finished."
