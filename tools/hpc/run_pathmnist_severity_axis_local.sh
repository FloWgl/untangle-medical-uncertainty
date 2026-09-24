#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"

manifest=${CHECKPOINT_MANIFEST:-configs/pathmnist_severity_axis_local_manifest.csv}
pathmnist_dir=${PATHMNIST_DIR:?set PATHMNIST_DIR to the PathMNIST directory}
python_bin=${PYTHON_BIN:-python3}
methods=${METHODS:-loss-prediction,postnet,edl,deep-correctness-prediction,deep-loss-prediction,sngp}
batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-2}
num_eval_workers=${NUM_EVAL_WORKERS:-2}
run_id=${RUN_ID:-pathmnist_severity_axis_local_$(date +%Y%m%d-%H%M%S)}
run_dir=${RUN_DIR:-logs/${run_id}}

mkdir -p "${run_dir}"

"${python_bin}" tools/hpc/generate_pathmnist_hpc_commands.py \
    --phase corruption-severity \
    --checkpoint-manifest "${manifest}" \
    --output "${run_dir}/commands.txt" \
    --python "${python_bin}" \
    --pathmnist-dir "${pathmnist_dir}" \
    --methods "${methods}" \
    --batch-size "${batch_size}" \
    --accumulation-steps 1 \
    --num-workers "${num_workers}" \
    --num-eval-workers "${num_eval_workers}" \
    --storage-device cpu

printf "%s\n" "${run_dir}" > logs/pathmnist_severity_axis_latest.txt

setsid bash -lc \
    "cd '${repo_root}' && CUDA_VISIBLE_DEVICES='' bash tools/run_command_manifest_local.sh '${run_dir}/commands.txt' '${run_dir}'" \
    > "${run_dir}/driver.log" 2>&1 < /dev/null &
printf "%s\n" "$!" > "${run_dir}/pid.txt"

echo "Started PathMNIST severity-axis run in ${run_dir}"
echo "PID: $(cat "${run_dir}/pid.txt")"
