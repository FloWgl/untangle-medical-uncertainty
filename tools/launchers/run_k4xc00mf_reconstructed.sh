#!/usr/bin/env bash
# Reconstructed CIFAR-10 MC-dropout launcher for missing sweep k4xc00mf.
# Fetch docs/wandb_sweeps/k4xc00mf.yaml with a W&B API key if exact sweep metadata is required.
set -euo pipefail

python3 train.py \
  --accumulation-steps 1 \
  --batch-size 128 \
  --crop-pct 1 \
  --data-dir ./data \
  --dataset hard/cifar10 \
  --dataset-id soft/cifar10 \
  --dropout-probability 0.1 \
  --epochs 200 \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
  --hflip 0.5 \
  --img-size 32 \
  --loss cross-entropy \
  --lr-base 0.2 \
  --mean 0.4914,0.4822,0.4465 \
  --method-name mc-dropout \
  --model-name untangle/wide_resnet_c_preact_26_10 \
  --momentum 0.9 \
  --num-classes 10 \
  --num-mc-samples 30 \
  --opt nesterov \
  --padding 2 \
  --pin-memory \
  --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
  --seed "${SEED:-42}" \
  --std 0.2023,0.1994,0.2010 \
  --storage-device cuda \
  --weight-decay 0.0005 \
  "$@"
