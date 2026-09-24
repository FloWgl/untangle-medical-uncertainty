#!/usr/bin/env bash
# Train five CE members that can feed deep-ensemble, temperature, Mahalanobis,
# Laplace, and SWAG follow-up runs when the authors' remote checkpoints are absent.
# Seeds match the recovered final CE W&B sweep uo3gu133.
set -euo pipefail

SEEDS=(${SEEDS:-0 1 2 3 4})
EXTRA_ARGS_FROM_ENV=(${EXTRA_ARGS:-})
EXTRA_ARGS=("${EXTRA_ARGS_FROM_ENV[@]}" "$@")

for seed in "${SEEDS[@]}"; do
  echo "[$(date --iso-8601=seconds)] START CIFAR-10 CE ensemble member seed=${seed}"
  SEED="$seed" bash tools/launchers/run_cedfgnqz_reconstructed.sh "${EXTRA_ARGS[@]}"
  echo "[$(date --iso-8601=seconds)] DONE CIFAR-10 CE ensemble member seed=${seed}"
done
