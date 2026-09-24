#!/usr/bin/env bash
# Simple smoke-run launcher for CIFAR-10 experiments
# Adjust args as needed; this runs a short training to verify setup.
set -euo pipefail
DATA_DIR=${1:-./data}
METHOD=${2:-ce-baseline}
EPOCHS=${3:-1}

python3 train.py \
  --dataset hard/cifar10 \
  --dataset-id hard/cifar10 \
  --data-dir "$DATA_DIR" \
  --batch-size 128 \
  --epochs "$EPOCHS" \
  --method-name "$METHOD" \
  --log-wandb false \
  --evaluate-on-test-sets \
  --dataset-download false

echo "Smoke run finished"
