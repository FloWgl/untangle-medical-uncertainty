#!/usr/bin/env bash
# Wait for previous CIFAR queues, train the five recovered CE members, generate
# the local post-hoc checkpoint manifest, then run CIFAR-10 post-hoc methods.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WAIT_SESSION="${WAIT_SESSION:-cifar10_missing_scarcity_after_ood_20260528-113417-missing-scarcity-after-ood}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-ce-members-posthoc}"
LOG_DIR="$REPO_ROOT/logs/cifar10_posthoc_stack_${RUN_ID}"
MARKER="$LOG_DIR/ce_members_start.marker"
MANIFEST="$REPO_ROOT/configs/cifar10_posthoc_weight_paths_${RUN_ID}.env"
STATUS_FILE="$LOG_DIR/status.tsv"
OOD_TRANSFORMS="gaussian_noise,shot_noise,impulse_noise,defocus_blur,frosted_glass_blur,motion_blur,zoom_blur,snow,frost,fog,brightness,contrast,elastic,pixelate,jpeg"
OOD_ARGS=(--evaluate-on-test-sets --ood-transforms-eval "$OOD_TRANSFORMS" --ood-transforms-test "$OOD_TRANSFORMS" --severities 1,2,3,4,5)

mkdir -p "$LOG_DIR" configs
printf 'run_id	stage	status	started_at	finished_at	log
' > "$STATUS_FILE"

log_status() {
  local stage="$1" status="$2" started="$3" finished="$4" log="$5"
  printf '%s	%s	%s	%s	%s	%s
' "$RUN_ID" "$stage" "$status" "$started" "$finished" "$log" >> "$STATUS_FILE"
}

echo "[$(date --iso-8601=seconds)] CE/posthoc watcher started."
echo "Waiting for tmux session: $WAIT_SESSION"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] Still waiting for $WAIT_SESSION"
  sleep "$POLL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] Previous queues finished; training CE members."
touch "$MARKER"
started="$(date --iso-8601=seconds)"
ce_log="$LOG_DIR/train_ce_members.log"
if bash tools/launchers/run_bn1hsbqz_members_reconstructed.sh --train-subset 1.0 --discard-ood-test-sets > "$ce_log" 2>&1; then
  finished="$(date --iso-8601=seconds)"
  log_status "train-ce-members" "done" "$started" "$finished" "$ce_log"
else
  finished="$(date --iso-8601=seconds)"
  log_status "train-ce-members" "failed" "$started" "$finished" "$ce_log"
  exit 1
fi

mapfile -t dirs < <(find checkpoints -maxdepth 1 -type d -newer "$MARKER"   -exec test -f '{}/checkpoint_best.pt' ';'   -exec test -f '{}/checkpoint_last.pt' ';'   -printf '%T@ %p
' | sort -n | awk '{print $2}')

if [[ ${#dirs[@]} -ne 5 ]]; then
  echo "Expected exactly five new CE checkpoint dirs after $MARKER, found ${#dirs[@]}" >&2
  printf '%s
' "${dirs[@]}" >&2
  exit 2
fi

started="$(date --iso-8601=seconds)"
manifest_log="$LOG_DIR/make_manifest.log"
if python3 tools/make_cifar10_posthoc_manifest.py --out "$MANIFEST" "${dirs[@]}" > "$manifest_log" 2>&1; then
  finished="$(date --iso-8601=seconds)"
  log_status "make-posthoc-manifest" "done" "$started" "$finished" "$manifest_log"
else
  finished="$(date --iso-8601=seconds)"
  log_status "make-posthoc-manifest" "failed" "$started" "$finished" "$manifest_log"
  exit 3
fi

cp "$MANIFEST" configs/cifar10_posthoc_weight_paths.env
printf '%s
' "${dirs[@]}" > "$LOG_DIR/ce_member_checkpoint_dirs.txt"

started="$(date --iso-8601=seconds)"
posthoc_log="$LOG_DIR/run_posthoc.log"
if bash tools/launchers/run_cifar10_posthoc_from_manifest.sh "$MANIFEST" "${OOD_ARGS[@]}" > "$posthoc_log" 2>&1; then
  finished="$(date --iso-8601=seconds)"
  log_status "run-posthoc-methods" "done" "$started" "$finished" "$posthoc_log"
else
  finished="$(date --iso-8601=seconds)"
  log_status "run-posthoc-methods" "failed" "$started" "$finished" "$posthoc_log"
  exit 4
fi

echo "[$(date --iso-8601=seconds)] CE/posthoc stack finished."
