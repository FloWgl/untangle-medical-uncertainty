#!/usr/bin/env bash
# Retry a failed CIFAR-10 available-stack job without competing with the active queue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
WAIT_FOR_TMUX="${WAIT_FOR_TMUX:-}"
LOG_DIR="$REPO_ROOT/logs/cifar10_retry_${RUN_ID}"
STATUS_FILE="$LOG_DIR/status.tsv"
mkdir -p "$LOG_DIR" pids

SCRIPT="${SCRIPT:-archived/cleanup_20260526-1455/tools/launchers/run_1692idyk.sh}"
PHASE="${PHASE:-full-data}"
EXTRA_ARGS="${EXTRA_ARGS:---train-subset 1.0 --discard-ood-test-sets --dataset-id soft/cifar10}"

extract_command() {
  local script="$1"
  awk 'NF && $1 !~ /^#/ {line=$0} END {print line}' "$script"
}

if [[ -n "$WAIT_FOR_TMUX" ]]; then
  echo "[$(date --iso-8601=seconds)] Waiting for tmux session '$WAIT_FOR_TMUX' to finish."
  while tmux has-session -t "$WAIT_FOR_TMUX" 2>/dev/null; do
    sleep 300
  done
fi

echo -e "run_id\tphase\tscript\tstatus\texit_code\tstarted_at\tfinished_at\tlog" > "$STATUS_FILE"

BASE_CMD="$(extract_command "$SCRIPT")"
if [[ -z "$BASE_CMD" ]]; then
  echo "Could not extract command from $SCRIPT"
  exit 2
fi

LOG_FILE="$LOG_DIR/${PHASE}_$(basename "$SCRIPT" .sh).log"
STARTED="$(date --iso-8601=seconds)"
echo "[$STARTED] RETRY START $PHASE $SCRIPT"
echo "[$STARTED] command=${BASE_CMD} ${EXTRA_ARGS}" > "$LOG_FILE"

set +e
eval "${BASE_CMD} ${EXTRA_ARGS}" >> "$LOG_FILE" 2>&1
RC=$?
set -e

FINISHED="$(date --iso-8601=seconds)"
if [[ $RC -eq 0 ]]; then
  echo "[$FINISHED] RETRY DONE $PHASE $SCRIPT"
  echo -e "${RUN_ID}\t${PHASE}\t${SCRIPT}\tdone\t0\t${STARTED}\t${FINISHED}\t${LOG_FILE}" >> "$STATUS_FILE"
else
  echo "[$FINISHED] RETRY FAIL $PHASE $SCRIPT exit=$RC"
  echo -e "${RUN_ID}\t${PHASE}\t${SCRIPT}\tfailed\t${RC}\t${STARTED}\t${FINISHED}\t${LOG_FILE}" >> "$STATUS_FILE"
fi

exit "$RC"
