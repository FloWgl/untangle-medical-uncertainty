#!/usr/bin/env bash
# Submit a PathMNIST corruption severity-axis evaluation.
#
# This evaluates clean PathMNIST checkpoints on on-the-fly PathMNIST-C-style
# corruptions at severities 1..5. It is intentionally separate from the fixed
# official NPZ PathMNIST-C evaluation, whose files do not encode a severity axis.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${CHECKPOINT_MANIFEST:-}" ]]; then
  echo "CHECKPOINT_MANIFEST must point to a CSV with columns method,checkpoint." >&2
  echo "Example: CHECKPOINT_MANIFEST=configs/pathmnist_checkpoint_manifest.csv $0" >&2
  exit 2
fi

export PHASE="${PHASE:-corruption-severity}"
export RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-pathmnist-corruption-severity}"
export CORRUPTION_SEVERITIES="${CORRUPTION_SEVERITIES:-1 2 3 4 5}"
export STORAGE_DEVICE="${STORAGE_DEVICE:-cpu}"
export SLURM_TIME="${SLURM_TIME:-1-00:00:00}"

exec tools/hpc/submit_pathmnist_hpc.sh
