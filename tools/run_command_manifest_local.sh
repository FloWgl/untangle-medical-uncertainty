#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 COMMANDS_FILE RUN_DIR [START_INDEX]" >&2
    exit 2
fi

commands_file=$1
run_dir=$2
start_index=${3:-1}
status_file="${run_dir}/status.tsv"

mkdir -p "${run_dir}"

if [[ ! -f "${status_file}" ]]; then
    printf "task_id\tstatus\texit_code\tstarted_at\tfinished_at\tlog_file\n" > "${status_file}"
fi

task_id=0
while IFS= read -r command || [[ -n "${command}" ]]; do
    [[ -z "${command}" ]] && continue
    task_id=$((task_id + 1))
    if (( task_id < start_index )); then
        continue
    fi

    log_file="${run_dir}/task_${task_id}.log"
    started_at=$(date -Iseconds)
    printf "%s\trunning\t\t%s\t\t%s\n" "${task_id}" "${started_at}" "${log_file}" >> "${status_file}"

    bash -lc "${command}" > "${log_file}" 2>&1
    exit_code=$?
    finished_at=$(date -Iseconds)

    if [[ ${exit_code} -eq 0 ]]; then
        status="finished"
    else
        status="failed"
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${task_id}" "${status}" "${exit_code}" "${started_at}" "${finished_at}" "${log_file}" >> "${status_file}"
done < "${commands_file}"
