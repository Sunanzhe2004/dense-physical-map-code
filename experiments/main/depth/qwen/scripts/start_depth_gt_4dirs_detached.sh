#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_RUNNER="${SCRIPT_DIR}/run_depth_gt_4dirs.sh"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/nohup_depth_4dirs.out}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/depth_4dirs.pid}"
ENV_NAME="${ENV_NAME:-albedo}"

if [[ ! -x /usr/bin/setsid ]]; then
  echo "setsid is unavailable; cannot use this detached launcher." >&2
  exit 1
fi

has_key=0
if [[ -n "${DASHSCOPE_API_KEY:-}" ]]; then
  has_key=1
fi
for idx in 0 1 2 3; do
  var_name="DASHSCOPE_API_KEY_${idx}"
  if [[ -n "${!var_name:-}" ]]; then
    has_key=1
  fi
done
if (( has_key == 0 )); then
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
  DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}" \
  DASHSCOPE_API_KEY_0="${DASHSCOPE_API_KEY_0:-}" \
  DASHSCOPE_API_KEY_1="${DASHSCOPE_API_KEY_1:-}" \
  DASHSCOPE_API_KEY_2="${DASHSCOPE_API_KEY_2:-}" \
  DASHSCOPE_API_KEY_3="${DASHSCOPE_API_KEY_3:-}" \
  DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/api/v1}" \
  IMAGE_SIZE="${IMAGE_SIZE:-1280*960}" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  INPUT_ROOT="${INPUT_ROOT:-${GT_ROOT:-/path/to/benchmark_data/GT}}" \
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/depth_qwen}" \
  WORKERS="${WORKERS:-4}" \
  bash -lc "${START_CMD}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

DETACHED_PID=$!
echo "${DETACHED_PID}" > "${PID_FILE}"

echo "Started in background."
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Log file: ${LOG_FILE}"
