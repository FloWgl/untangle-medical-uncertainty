#!/usr/bin/env bash
# Wait for the current non-OOD CIFAR-10 queue/retry sessions, then run the
# available CIFAR-10 stack with full CIFAR-C/OOD evaluation enabled.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAIT_SESSIONS="${WAIT_SESSIONS:-cifar10_available_20260526-150332 cifar10_retry_20260526-retry-1692idyk-after-available}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-ood}"
OUT_FILE="$REPO_ROOT/logs/cifar10_available_full_stack_${RUN_ID}.out"

mkdir -p "$REPO_ROOT/logs"

echo "[$(date --iso-8601=seconds)] OOD watcher started."
echo "Waiting for tmux sessions: ${WAIT_SESSIONS}"
echo "Poll interval: ${POLL_SECONDS}s"
echo "Planned OOD RUN_ID: ${RUN_ID}"

while true; do
  active=()
  for session in ${WAIT_SESSIONS}; do
    if tmux has-session -t "$session" 2>/dev/null; then
      active+=("$session")
    fi
  done

  if [[ ${#active[@]} -eq 0 ]]; then
    break
  fi

  echo "[$(date --iso-8601=seconds)] Still waiting for: ${active[*]}"
  sleep "$POLL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] Dependencies finished; starting RUN_OOD=1 CIFAR-10 available stack."
echo "Output: ${OUT_FILE}"

RUN_ID="$RUN_ID" RUN_OOD=1 bash tools/run_cifar10_available_full_stack_queue.sh > "$OUT_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] OOD available-stack queue finished."
