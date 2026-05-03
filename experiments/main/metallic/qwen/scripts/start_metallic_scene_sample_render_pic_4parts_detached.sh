#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INNER_RUNNER="${INNER_RUNNER:-${SCRIPT_DIR}/run_metallic_scene_sample_render_pic_4parts.sh}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/nohup_metallic.out}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/metallic.pid}"
ENV_NAME="${ENV_NAME:-albedo}"
NUM_WORKERS="${NUM_WORKERS:-4}"

if [[ ! -x /usr/bin/setsid ]]; then
  echo "setsid is unavailable; cannot use this detached launcher." >&2
  exit 1
fi

for ((idx = 0; idx < NUM_WORKERS; idx++)); do
  var_name="DASHSCOPE_API_KEY_${idx}"
  if [[ -z "${!var_name:-}" ]]; then
    echo "${var_name} is not set." >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${LOG_FILE}")"

if [[ "${SKIP_CONDA:-0}" == "1" ]]; then
  START_CMD='exec bash "${INNER_RUNNER}"'
else
  START_CMD=$(cat <<'EOF'
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"
exec bash "${INNER_RUNNER}"
EOF
)
fi

env_args=(
  ENV_NAME="${ENV_NAME}"
  INNER_RUNNER="${INNER_RUNNER}"
  PYTHON_BIN="${PYTHON_BIN:-python3}"
  RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_ROOT}/metallic_generation_wan_filltight_final.py}"
  INPUT_DIR="${INPUT_DIR:-/path/to/benchmark_data/metallic_scene}"
  OUTPUT_DIR="${OUTPUT_DIR:-/path/to/benchmark_outputs/metallic_qwen}"
  LOG_DIR="${LOG_DIR:-${OUTPUT_DIR:-/path/to/benchmark_outputs/metallic_qwen}/logs}"
  FILENAME_GLOB="${FILENAME_GLOB:-Image*.png}"
  NUM_WORKERS="${NUM_WORKERS}"
  MAX_GENERATE="${MAX_GENERATE:-}"
  SKIP_CONDA="${SKIP_CONDA:-0}"
)

for ((idx = 0; idx < NUM_WORKERS; idx++)); do
  var_name="DASHSCOPE_API_KEY_${idx}"
  env_args+=("${var_name}=${!var_name}")
done

setsid env \
  "${env_args[@]}" \
  bash -lc "${START_CMD}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

DETACHED_PID=$!
echo "${DETACHED_PID}" > "${PID_FILE}"

echo "Started in background."
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Log file: ${LOG_FILE}"
