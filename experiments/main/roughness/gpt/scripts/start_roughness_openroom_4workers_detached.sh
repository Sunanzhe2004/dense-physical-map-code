#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_RUNNER="${SCRIPT_DIR}/run_roughness_openroom_4workers.sh"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/nohup_roughness_openroom_4workers.out}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/roughness_openroom_4workers.pid}"
ENV_NAME="${ENV_NAME:-albedo}"
API_KEY="${AZURE_ROUGHNESS_OPENAI_API_KEY:-${AZURE_GPT_IMAGE_15_API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-${GPT_IMAGE_API_KEY:-}}}}}"

if [[ ! -x /usr/bin/setsid ]]; then
  echo "setsid is unavailable; cannot use this detached launcher." >&2
  exit 1
fi

if [[ -z "${API_KEY}" ]]; then
  echo "No GPT image API key is set. Set AZURE_ROUGHNESS_OPENAI_API_KEY, AZURE_OPENAI_API_KEY, OPENAI_API_KEY, or GPT_IMAGE_API_KEY." >&2
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
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  GT_ROOT="${GT_ROOT:-/path/to/benchmark_data/GT}" \
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/roughness_gpt}" \
  FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}" \
  INPUT_MODE="${INPUT_MODE:-rgb_only}" \
  LOG_DIR="${LOG_DIR:-}" \
  PARTS_PER_DATASET="${PARTS_PER_DATASET:-2}" \
  OVERWRITE="${OVERWRITE:-0}" \
  MAX_GENERATE="${MAX_GENERATE:-}" \
  SIZE="${SIZE:-}" \
  SLEEP="${SLEEP:-}" \
  IMAGE_MODEL="${IMAGE_MODEL:-}" \
  IMAGE_QUALITY="${IMAGE_QUALITY:-}" \
  BASE_URL="${BASE_URL:-}" \
  API_VERSION="${API_VERSION:-}" \
  SAVE_DEBUG_INTERMEDIATES="${SAVE_DEBUG_INTERMEDIATES:-0}" \
  EXAMPLE_RGB="${EXAMPLE_RGB:-}" \
  EXAMPLE_ROUGHNESS="${EXAMPLE_ROUGHNESS:-}" \
  AZURE_ROUGHNESS_OPENAI_API_KEY="${API_KEY}" \
  AZURE_GPT_IMAGE_15_API_KEY="${AZURE_GPT_IMAGE_15_API_KEY:-}" \
  AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-}" \
  OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-}" \
  AZURE_ROUGHNESS_OPENAI_ENDPOINT="${AZURE_ROUGHNESS_OPENAI_ENDPOINT:-}" \
  AZURE_GPT_IMAGE_15_ENDPOINT="${AZURE_GPT_IMAGE_15_ENDPOINT:-}" \
  AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}" \
  AZURE_ROUGHNESS_OPENAI_API_VERSION="${AZURE_ROUGHNESS_OPENAI_API_VERSION:-}" \
  AZURE_GPT_IMAGE_15_API_VERSION="${AZURE_GPT_IMAGE_15_API_VERSION:-}" \
  AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-}" \
  bash -lc "${START_CMD}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

DETACHED_PID=$!
echo "${DETACHED_PID}" > "${PID_FILE}"

echo "Started in background."
echo "PID: ${DETACHED_PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
