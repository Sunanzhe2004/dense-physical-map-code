#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_normal_singleview_4dirs.sh"
NOHUP_LOG="${SCRIPT_DIR}/nohup_4dirs.out"
PID_FILE="${SCRIPT_DIR}/normal_4dirs.pid"

nohup bash "${RUNNER}" > "${NOHUP_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started in background: PID=$(cat "${PID_FILE}")"
echo "Log file: ${NOHUP_LOG}"
echo "PID file: ${PID_FILE}"
