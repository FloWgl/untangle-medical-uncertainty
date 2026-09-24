#!/usr/bin/env bash
# Recovered CIFAR-10 DUQ launcher from W&B sweeps:
# mfsz17vy (search), ii1o54ln (100%), 55lfn3px (50%), br8w7ajf (10%).
set -euo pipefail

python3 train.py \
  --accumulation-steps 1 \
  --batch-size 128 \
  --crop-pct 1 \
  --data-dir ./data \
  --dataset hard/cifar10 \
  --dataset-id soft/cifar10 \
  --epochs 200 \
  --eval-metric id_eval_duq_values_auroc_hard_bma_correctness_original \
  --hflip 0.5 \
  --img-size 32 \
  --loss duq \
  --lr 0.13540695861213595 \
  --mean 0.4914,0.4822,0.4465 \
  --method-name duq \
  --model-name untangle/wide_resnet_c_preact_26_10 \
  --momentum 0.9 \
  --num-classes 10 \
  --num-hidden-features -1 \
  --opt nesterov \
  --padding 2 \
  --pin-memory \
  --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
  --seed "${SEED:-0}" \
  --std 0.2023,0.1994,0.2010 \
  --storage-device cuda \
  --weight-decay 0.0013871238024266957 \
  "$@"
