#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROMPT_FILE="${SCRIPT_DIR}/prompt.txt"
RUN_DIR="/tmp/kuttajoukowski-codex"
LOG_FILE="${RUN_DIR}/run.log"
FINAL_FILE="${RUN_DIR}/final.txt"
PID_FILE="${RUN_DIR}/pid"
UV_CACHE_DIR="${RUN_DIR}/uv-cache"

if ! command -v codex >/dev/null 2>&1; then
    echo "Error: codex is not installed or is not on PATH." >&2
    exit 1
fi

if [[ ! -s "${PROMPT_FILE}" ]]; then
    echo "Error: ${PROMPT_FILE} is missing or empty." >&2
    exit 1
fi

umask 077
mkdir -p "${RUN_DIR}" "${UV_CACHE_DIR}"

if [[ ! -w "${UV_CACHE_DIR}" ]]; then
    echo "Error: UV cache is not writable: ${UV_CACHE_DIR}" >&2
    exit 1
fi

export UV_CACHE_DIR

if [[ -f "${PID_FILE}" ]]; then
    existing_pid="$(<"${PID_FILE}")"
    if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
        echo "Codex already appears to be running (PID ${existing_pid})."
        echo "Follow the log with: tail -f ${LOG_FILE}"
        exit 1
    fi
fi

codex_command=(
    codex
    --approve-for-me
    -C "${SCRIPT_DIR}"
    --add-dir "${UV_CACHE_DIR}"
    exec
    --color never
    -o "${FINAL_FILE}"
    -
)

if command -v caffeinate >/dev/null 2>&1; then
    launch_command=(caffeinate -i "${codex_command[@]}")
else
    launch_command=("${codex_command[@]}")
    echo "Warning: caffeinate is unavailable; this script cannot prevent system sleep." >&2
fi

nohup "${launch_command[@]}" \
    < "${PROMPT_FILE}" \
    > "${LOG_FILE}" 2>&1 &

launcher_pid=$!
printf '%s\n' "${launcher_pid}" > "${PID_FILE}"

echo "Started Codex in the background."
echo "PID:       ${launcher_pid}"
echo "Log:       ${LOG_FILE}"
echo "Final:     ${FINAL_FILE}"
echo "UV cache:  ${UV_CACHE_DIR}"
echo "Monitor:   tail -f ${LOG_FILE}"
echo
echo "You may close this terminal, but do not close the MacBook lid."
