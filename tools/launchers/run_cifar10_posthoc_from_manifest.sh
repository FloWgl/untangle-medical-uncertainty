#!/usr/bin/env bash
# Run CIFAR-10 final/posthoc methods from local checkpoint paths.
set -euo pipefail

MANIFEST="${1:-configs/cifar10_posthoc_weight_paths.env}"
if [[ $# -gt 0 ]]; then
  shift
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST"
  echo "Create it from configs/cifar10_posthoc_weight_paths.env.example."
  exit 2
fi

set -a
source "$MANIFEST"
set +a

if [[ -z "${CIFAR10_BEST_WEIGHT_PATHS:-}" ]]; then
  echo "CIFAR10_BEST_WEIGHT_PATHS is empty in $MANIFEST"
  exit 3
fi

if [[ -z "${CIFAR10_LAST_WEIGHT_PATHS:-}" ]]; then
  echo "CIFAR10_LAST_WEIGHT_PATHS is empty in $MANIFEST"
  exit 4
fi

COMMON_ARGS=(
  --accumulation-steps 1
  --batch-size 128
  --crop-pct 1
  --data-dir ./data
  --dataset hard/cifar10
  --dataset-id soft/cifar10
  --epochs 0
  --evaluate-on-test-sets
  --hflip 0.5
  --img-size 32
  --loss cross-entropy
  --lr 0
  --mean 0.4914,0.4822,0.4465
  --model-name untangle/wide_resnet_c_preact_26_10
  --momentum 0.9
  --num-classes 10
  --opt nesterov
  --padding 2
  --pin-memory
  --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0 warmup_epochs=0'
  --seed 42
  --std 0.2023,0.1994,0.2010
  --storage-device cuda
  --weight-decay 0.004022874547456138
)

echo "[$(date --iso-8601=seconds)] START deep-ensemble from best checkpoints"
python3 train.py "${COMMON_ARGS[@]}" \
  --method-name deep-ensemble \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
  --weight-paths "$CIFAR10_BEST_WEIGHT_PATHS" "$@"

echo "[$(date --iso-8601=seconds)] START temperature-scaling from best checkpoints"
python3 train.py "${COMMON_ARGS[@]}" \
  --method-name temperature-scaling \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
  --use-temperature-scaling \
  --weight-paths "$CIFAR10_BEST_WEIGHT_PATHS" "$@"

echo "[$(date --iso-8601=seconds)] START mahalanobis from last checkpoints"
python3 train.py "${COMMON_ARGS[@]}" \
  --method-name mahalanobis \
  --eval-metric id_eval_mahalanobis_values_auroc_hard_bma_correctness_original \
  --magnitude 0.001 \
  --module-name-regex '^(layer[1-3].0.act1|act)$' \
  --weight-paths "$CIFAR10_LAST_WEIGHT_PATHS" "$@"

echo "[$(date --iso-8601=seconds)] START laplace from best checkpoints"
python3 train.py "${COMMON_ARGS[@]}" \
  --method-name laplace \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
  --hessian-structure kron \
  --num-mc-samples 1000 \
  --num-mc-samples-cv 1000 \
  --pred-type glm \
  --weight-paths "$CIFAR10_BEST_WEIGHT_PATHS" "$@"

echo "[$(date --iso-8601=seconds)] START swag from last checkpoints"
python3 train.py \
  --accumulation-steps 1 \
  --batch-size 128 \
  --crop-pct 1 \
  --data-dir ./data \
  --dataset hard/cifar10 \
  --dataset-id soft/cifar10 \
  --epochs 40 \
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original \
  --evaluate-on-test-sets \
  --hflip 0.5 \
  --img-size 32 \
  --loss cross-entropy \
  --lr 0.01 \
  --max-rank 20 \
  --mean 0.4914,0.4822,0.4465 \
  --method-name swag \
  --model-name untangle/wide_resnet_c_preact_26_10 \
  --momentum 0.9 \
  --num-checkpoints-per-epoch 1 \
  --num-classes 10 \
  --num-mc-samples 30 \
  --opt momentum \
  --padding 2 \
  --pin-memory \
  --sched-kwargs sched=none \
  --seed 42 \
  --std 0.2023,0.1994,0.2010 \
  --storage-device cuda \
  --use-low-rank-cov \
  --weight-decay 0.0001 \
  --weight-paths "$CIFAR10_LAST_WEIGHT_PATHS" "$@"

echo "[$(date --iso-8601=seconds)] CIFAR-10 posthoc queue finished."
