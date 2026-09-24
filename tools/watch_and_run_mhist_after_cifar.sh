#!/usr/bin/env bash
# Wait for the active CIFAR queues, then launch the MHIST queue.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAIT_SESSIONS="${WAIT_SESSIONS:-cifar10_ood_after_current_20260527-011454-ood-after-current cifar10_missing_scarcity_after_ood_20260528-113417-missing-scarcity-after-ood cifar10_ce_members_posthoc_20260528-132834-ce-members-posthoc cifar10_duq_recovered_20260528-133840-duq-recovered}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-mhist-after-cifar}"
OUT_FILE="$REPO_ROOT/logs/mhist_cifar_params_${RUN_ID}.out"

mkdir -p "$REPO_ROOT/logs"

echo "[$(date --iso-8601=seconds)] MHIST watcher started."
echo "Waiting for tmux sessions: ${WAIT_SESSIONS}"
echo "Poll interval: ${POLL_SECONDS}s"
echo "Planned MHIST RUN_ID: ${RUN_ID}"

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

echo "[$(date --iso-8601=seconds)] CIFAR queues finished; starting MHIST CIFAR-parameter queue."
echo "Output: ${OUT_FILE}"

RUN_ID="$RUN_ID" bash tools/run_mhist_cifar_params_queue.sh > "$OUT_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] MHIST queue finished."
