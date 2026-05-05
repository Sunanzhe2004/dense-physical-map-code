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

if [[ -z "${ALBEDO_API_KEY:-}" && -z "${AZURE_ALBEDO_OPENAI_API_KEY:-}" && -z "${AZURE_GPT_IMAGE_15_API_KEY:-}" && -z "${AZURE_GPT_IMAGE_2_API_KEY:-}" ]]; then
  echo "ALBEDO_API_KEY is not set (or AZURE_ALBEDO_OPENAI_API_KEY / AZURE_GPT_IMAGE_15_API_KEY / AZURE_GPT_IMAGE_2_API_KEY)." >&2
  exit 1
fi

if [[ -z "${ALBEDO_ENDPOINT:-}" && -z "${AZURE_ALBEDO_OPENAI_ENDPOINT:-}" && -z "${AZURE_GPT_IMAGE_15_ENDPOINT:-}" && -z "${AZURE_GPT_IMAGE_2_ENDPOINT:-}" ]]; then
  echo "ALBEDO_ENDPOINT is not set (or AZURE_ALBEDO_OPENAI_ENDPOINT / AZURE_GPT_IMAGE_15_ENDPOINT / AZURE_GPT_IMAGE_2_ENDPOINT)." >&2
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
  ALBEDO_API_KEY="${ALBEDO_API_KEY:-}" \
  AZURE_ALBEDO_OPENAI_API_KEY="${AZURE_ALBEDO_OPENAI_API_KEY:-}" \
  AZURE_GPT_IMAGE_15_API_KEY="${AZURE_GPT_IMAGE_15_API_KEY:-}" \
  AZURE_GPT_IMAGE_2_API_KEY="${AZURE_GPT_IMAGE_2_API_KEY:-}" \
  ALBEDO_ENDPOINT="${ALBEDO_ENDPOINT:-${AZURE_ALBEDO_OPENAI_ENDPOINT:-${AZURE_GPT_IMAGE_2_ENDPOINT:-${AZURE_GPT_IMAGE_15_ENDPOINT:-}}}}" \
  ALBEDO_API_VERSION="${ALBEDO_API_VERSION:-${AZURE_ALBEDO_OPENAI_API_VERSION:-${AZURE_GPT_IMAGE_2_API_VERSION:-${AZURE_GPT_IMAGE_15_API_VERSION:-2025-04-01-preview}}}}" \
  ALBEDO_MODEL="${ALBEDO_MODEL:-gpt-image-1.5}" \
  ALBEDO_DEPLOYMENT="${ALBEDO_DEPLOYMENT:-gpt-image-1.5}" \
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/albedo_gpt}" \
  FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}" \
  GENERATION_MODE="${GENERATION_MODE:-edit}" \
  ALBEDO_SIZE="${ALBEDO_SIZE:-1536x1024}" \
  ALBEDO_QUALITY="${ALBEDO_QUALITY:-medium}" \
  TIMEOUT="${TIMEOUT:-600}" \
  REQUEST_RETRIES="${REQUEST_RETRIES:-5}" \
  RETRY_BACKOFF="${RETRY_BACKOFF:-5}" \
  RETRY_MAX_BACKOFF="${RETRY_MAX_BACKOFF:-60}" \
  MAX_GENERATE="${MAX_GENERATE:-}" \
  OVERWRITE="${OVERWRITE:-0}" \
  GENERATE_REQUIRES_IMAGE="${GENERATE_REQUIRES_IMAGE:-0}" \
  bash -lc "${START_CMD}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

DETACHED_PID=$!
echo "${DETACHED_PID}" > "${PID_FILE}"

echo "Started in background."
echo "PID: ${DETACHED_PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
