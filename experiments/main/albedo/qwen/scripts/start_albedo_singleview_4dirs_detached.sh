#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_RUNNER="${SCRIPT_DIR}/run_albedo_singleview_4dirs.sh"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/nohup_4dirs.out}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/albedo_4dirs.pid}"
ENV_NAME="${ENV_NAME:-albedo}"

if [[ ! -x /usr/bin/setsid ]]; then
  echo "setsid is unavailable; cannot use this detached launcher." >&2
  exit 1
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is not set." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

START_CMD=$(cat <<'EOF'
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"
exec bash "${INNER_RUNNER}"
EOF
)

setsid env \
  ENV_NAME="${ENV_NAME}" \
  INNER_RUNNER="${INNER_RUNNER}" \
  DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY}" \
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/albedo_qwen}" \
  FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}" \
  GENERATION_MODE="${GENERATION_MODE:-edit}" \
  BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/api/v1}" \
  ALBEDO_MODEL="${ALBEDO_MODEL:-wan2.7-image}" \
  OVERWRITE="${OVERWRITE:-0}" \
  MAX_GENERATE="${MAX_GENERATE:-}" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  bash -lc "${START_CMD}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

DETACHED_PID=$!
echo "${DETACHED_PID}" > "${PID_FILE}"

echo "Started in background."
echo "PID: ${DETACHED_PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
