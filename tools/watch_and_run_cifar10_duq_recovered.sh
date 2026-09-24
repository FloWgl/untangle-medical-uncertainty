#!/usr/bin/env bash
# Wait for the CE-member/post-hoc stack, then run recovered CIFAR-10 DUQ jobs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAIT_SESSION="${WAIT_SESSION:-cifar10_ce_members_posthoc_20260528-132834-ce-members-posthoc}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-duq-recovered}"
LOG_DIR="$REPO_ROOT/logs/cifar10_duq_recovered_${RUN_ID}"
STATUS_FILE="$LOG_DIR/status.tsv"
OOD_TRANSFORMS="gaussian_noise,shot_noise,impulse_noise,defocus_blur,frosted_glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic,pixelate,jpeg"
OOD_FLAGS="--evaluate-on-test-sets --ood-transforms-eval ${OOD_TRANSFORMS} --ood-transforms-test ${OOD_TRANSFORMS} --severities 1,2,3,4,5"

mkdir -p "$LOG_DIR"
printf 'run_id	phase	seed	status	exit_code	started_at	finished_at	log
' > "$STATUS_FILE"

echo "[$(date --iso-8601=seconds)] DUQ watcher started. Waiting for $WAIT_SESSION"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] Still waiting for $WAIT_SESSION"
  sleep "$POLL_SECONDS"
done

run_one() {
  local phase="$1" subset="$2" seed="$3"
  local log_file="$LOG_DIR/${phase}_seed${seed}.log"
  local started finished rc
  started="$(date --iso-8601=seconds)"
  echo "[$started] START DUQ ${phase} seed=${seed} subset=${subset}"
  echo "[$started] command=SEED=${seed} bash tools/launchers/run_mfsz17vy_duq_recovered.sh --train-subset ${subset} ${OOD_FLAGS}" > "$log_file"
  SEED="$seed" bash tools/launchers/run_mfsz17vy_duq_recovered.sh --train-subset "$subset" ${OOD_FLAGS} >> "$log_file" 2>&1
  rc=$?
  finished="$(date --iso-8601=seconds)"
  if [[ $rc -eq 0 ]]; then
    echo "[$finished] DONE DUQ ${phase} seed=${seed}"
    printf '%s	%s	%s	done	0	%s	%s	%s
' "$RUN_ID" "$phase" "$seed" "$started" "$finished" "$log_file" >> "$STATUS_FILE"
  else
    echo "[$finished] FAIL DUQ ${phase} seed=${seed} exit=${rc}"
    printf '%s	%s	%s	failed	%s	%s	%s	%s
' "$RUN_ID" "$phase" "$seed" "$rc" "$started" "$finished" "$log_file" >> "$STATUS_FILE"
  fi
  return "$rc"
}

for phase_subset in "full-data:1.0" "scarce-50pct:0.5" "scarce-10pct:0.1"; do
  phase="${phase_subset%%:*}"
  subset="${phase_subset##*:}"
  for seed in 0 1 2 3 4; do
    run_one "$phase" "$subset" "$seed" || true
  done
done

echo "[$(date --iso-8601=seconds)] DUQ recovered queue finished."
