#!/usr/bin/env bash
# Run MHIST with the recovered CIFAR-10 training parameters.
#
# This queue is intentionally conservative for the university server: it keeps
# the CIFAR WRN-26-10 / 32px setup and uses MHIST hard labels for training plus
# MHIST soft rater-vote labels for evaluation.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)-mhist-cifar-params}"
LOG_DIR="$REPO_ROOT/logs/mhist_cifar_params_${RUN_ID}"
PID_DIR="$REPO_ROOT/pids"
STATUS_FILE="$LOG_DIR/status.tsv"
LOCK_FILE="$LOG_DIR/.queue.lock"
FAILED_FILE="$LOG_DIR/failed.tsv"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DATA_DIR="${MHIST_DATA_DIR:-./data/MHIST}"
SOFT_LABEL_ROOT="${MHIST_SOFT_LABEL_ROOT:-./data/MHIST}"
STORAGE_DEVICE="${STORAGE_DEVICE:-cpu}"

mkdir -p "$LOG_DIR" "$PID_DIR"

exec 9>"$LOCK_FILE"
flock -n 9 || {
  echo "An MHIST CIFAR-parameter queue is already running for ${RUN_ID}."
  exit 1
}

COMMON_ARGS=(
  --accumulation-steps 1
  --batch-size 128
  --crop-pct 1
  --data-dir "$DATA_DIR"
  --soft-label-root "$SOFT_LABEL_ROOT"
  --dataset hard/mhist
  --dataset-id soft/mhist
  --eval-metric id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original
  --evaluate-on-test-sets
  --hflip 0.5
  --img-size 32
  --mean 0.4914,0.4822,0.4465
  --model-name untangle/wide_resnet_c_preact_26_10
  --momentum 0.9
  --num-classes 2
  --opt nesterov
  --padding 2
  --pin-memory
  --seed 42
  --std 0.2023,0.1994,0.2010
  --storage-device "$STORAGE_DEVICE"
  --discard-ood-test-sets
)

printf 'run_id\tphase\tmethod\tstatus\texit_code\tstarted_at\tfinished_at\tlog\n' > "$STATUS_FILE"
: > "$FAILED_FILE"

run_job() {
  local phase="$1"
  local subset="$2"
  local method="$3"
  shift 3

  local log_file="$LOG_DIR/${phase}_${method}.log"
  local start_time end_time rc

  start_time="$(date --iso-8601=seconds)"
  echo "[$start_time] START ${phase} ${method}"
  {
    echo "[$start_time] command=${PYTHON_BIN} train.py ${COMMON_ARGS[*]} --train-subset ${subset} $*"
  } > "$log_file"

  "$PYTHON_BIN" train.py \
    "${COMMON_ARGS[@]}" \
    --train-subset "$subset" \
    "$@" >> "$log_file" 2>&1
  rc=$?

  end_time="$(date --iso-8601=seconds)"
  if [[ $rc -eq 0 ]]; then
    echo "[$end_time] DONE ${phase} ${method}"
    printf '%s\t%s\t%s\tdone\t0\t%s\t%s\t%s\n' "$RUN_ID" "$phase" "$method" "$start_time" "$end_time" "$log_file" >> "$STATUS_FILE"
  else
    echo "[$end_time] FAIL ${phase} ${method} exit=${rc}"
    printf '%s\t%s\t%s\tfailed\t%s\t%s\t%s\t%s\n' "$RUN_ID" "$phase" "$method" "$rc" "$start_time" "$end_time" "$log_file" >> "$STATUS_FILE"
    printf '%s\t%s\t%s\n' "$phase" "$method" "$*" >> "$FAILED_FILE"
  fi

  return "$rc"
}

run_phase() {
  local phase="$1"
  local subset="$2"

  echo "[$(date --iso-8601=seconds)] PHASE START ${phase} subset=${subset}"

  run_job "$phase" "$subset" ce-baseline \
    --epochs 200 \
    --loss cross-entropy \
    --lr 0.1395872780316071 \
    --method-name ce-baseline \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 0.004022874547456138

  run_job "$phase" "$subset" correctness-prediction \
    --epochs 200 \
    --eval-metric id_eval_error_probabilities_auroc_hard_bma_correctness_original \
    --lambda-uncertainty-loss 0.01 \
    --loss correctness-prediction \
    --lr-base 0.2 \
    --method-name correctness-prediction \
    --num-hidden-features 1024 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 0.0005

  run_job "$phase" "$subset" mc-dropout \
    --dropout-probability 0.1 \
    --epochs 200 \
    --loss cross-entropy \
    --lr-base 0.2 \
    --method-name mc-dropout \
    --num-mc-samples 30 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 0.0005

  run_job "$phase" "$subset" sngp-1692idyk \
    --epochs 250 \
    --gp-cov-momentum -1 \
    --gp-cov-ridge-penalty 1 \
    --gp-input-dim -1 \
    --gp-kernel-scale 1 \
    --gp-output-bias 0 \
    --gp-random-feature-type orf \
    --likelihood gaussian \
    --loss cross-entropy \
    --lr 0.06477318711511448 \
    --method-name sngp \
    --num-mc-samples 1000 \
    --num-random-features 1024 \
    --sched-kwargs 'sched=multistep decay_milestones=75,150,200 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --spectral-normalization-bound 6 \
    --spectral-normalization-iteration 1 \
    --weight-decay 0.001405684338069915

  run_job "$phase" "$subset" edl \
    --edl-activation softplus \
    --edl-scaler 1 \
    --edl-start-epoch 0 \
    --epochs 200 \
    --loss edl \
    --lr 0.023784145260620084 \
    --method-name edl \
    --num-mc-samples 1000 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 2.3932585920741396e-05

  run_job "$phase" "$subset" ddu \
    --epochs 250 \
    --loss cross-entropy \
    --max-num-id-train-samples 100000 \
    --lr 0.05571770292668626 \
    --method-name ddu \
    --sched-kwargs 'sched=multistep decay_milestones=75,150,200 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --spectral-normalization-bound 3 \
    --spectral-normalization-iteration 1 \
    --use-spectral-normalization \
    --use-temperature-scaling \
    --use-tight-norm-for-pointwise-convs \
    --weight-decay 0.0015213600092979422

  run_job "$phase" "$subset" postnet \
    --epochs 200 \
    --latent-dim 6 \
    --loss uce \
    --lr 0.1022121272066948 \
    --method-name postnet \
    --num-density-components 6 \
    --num-hidden-features 64 \
    --num-mc-samples 1000 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --uce-regularization-factor 1e-05 \
    --weight-decay 0.0021021422197140325

  run_job "$phase" "$subset" loss-prediction \
    --detach-uncertainty-target \
    --epochs 200 \
    --eval-metric id_eval_loss_values_auroc_hard_bma_correctness_original \
    --loss loss-prediction \
    --lr 0.11482920440979424 \
    --method-name loss-prediction \
    --mlp-depth 3 \
    --num-hidden-features 1024 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 6.546158927128921e-05

  run_job "$phase" "$subset" het-xl-e6rpfaue \
    --epochs 200 \
    --loss bma-cross-entropy \
    --matrix-rank 6 \
    --lr 0.0875579735357911 \
    --method-name het-xl \
    --num-mc-samples 1000 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --temperature 1.3 \
    --use-het \
    --weight-decay 0.00027823460119237977

  run_job "$phase" "$subset" sngp-jkzjb5vz \
    --epochs 250 \
    --gp-cov-momentum -1 \
    --gp-cov-ridge-penalty 1 \
    --gp-input-dim -1 \
    --gp-kernel-scale 1 \
    --gp-output-bias 0 \
    --gp-random-feature-type orf \
    --likelihood gaussian \
    --loss cross-entropy \
    --lr 0.1053200118343418 \
    --method-name sngp \
    --num-mc-samples 1000 \
    --num-random-features 1024 \
    --sched-kwargs 'sched=multistep decay_milestones=75,150,200 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --spectral-normalization-bound 6 \
    --spectral-normalization-iteration 1 \
    --use-spectral-normalization \
    --weight-decay 0.0012217236419884226

  run_job "$phase" "$subset" shallow-ensemble \
    --epochs 200 \
    --loss bma-cross-entropy \
    --lr 0.08753198042323167 \
    --method-name shallow-ensemble \
    --num-heads 10 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --weight-decay 0.0008792941852492912

  run_job "$phase" "$subset" het-xl-olapo0kg \
    --epochs 200 \
    --loss bma-cross-entropy \
    --matrix-rank 6 \
    --lr 0.09522291439603134 \
    --method-name het-xl \
    --num-mc-samples 1000 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --temperature 1.3 \
    --weight-decay 0.0031745702829380448

  run_job "$phase" "$subset" hetclassnn \
    --dropout-probability 0.1 \
    --epochs 200 \
    --loss bma-cross-entropy \
    --lr 0.06291698819769312 \
    --method-name hetclassnn \
    --num-mc-samples 30 \
    --num-mc-samples-integral 100 \
    --sched-kwargs 'sched=multistep decay_milestones=60,120,160 decay_rate=0.2 warmup_lr=0.005 warmup_epochs=1' \
    --use-filterwise-dropout \
    --weight-decay 0.0010584500388027963

  echo "[$(date --iso-8601=seconds)] PHASE END ${phase}"
}

echo "[$(date --iso-8601=seconds)] MHIST CIFAR-parameter queue ${RUN_ID}"
echo "Repository: ${REPO_ROOT}"
echo "Data: ${DATA_DIR}"
echo "Status file: ${STATUS_FILE}"
echo "OOD mode: disabled for MHIST until MHIST-C style corruptions are implemented."

run_phase "full-data" "1.0"
run_phase "scarce-50pct" "0.5"
run_phase "scarce-10pct" "0.1"

echo "[$(date --iso-8601=seconds)] Queue finished."
