#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INNER_RUNNER="${INNER_RUNNER:-${SCRIPT_DIR}/run_metallic_scene_sample_render_pic_4parts.sh}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/nohup_metallic.out}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/metallic.pid}"
ENV_NAME="${ENV_NAME:-albedo}"

if [[ ! -x /usr/bin/setsid ]]; then
  echo "setsid is unavailable; cannot use this detached launcher." >&2
  exit 1
fi

API_KEY="${AZURE_METALLIC_OPENAI_API_KEY:-${AZURE_GPT_IMAGE_2_API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}}"
if [[ -z "${API_KEY}" ]]; then
  echo "GPT Image API key is not set. Set one of AZURE_METALLIC_OPENAI_API_KEY / AZURE_GPT_IMAGE_2_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY." >&2
  exit 1
fi

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

setsid env \
  ENV_NAME="${ENV_NAME}" \
  INNER_RUNNER="${INNER_RUNNER}" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_ROOT}/metallic_generation_gpt2.py}" \
  INPUT_DIR="${INPUT_DIR:-/path/to/benchmark_data/metallic_scene}" \
  OUTPUT_DIR="${OUTPUT_DIR:-/path/to/benchmark_outputs/metallic_gpt2}" \
  LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}" \
  STAGING_DIR="${STAGING_DIR:-${OUTPUT_DIR}/staging_prompts}" \
  PROMPT_SOURCE_DIR="${PROMPT_SOURCE_DIR:-}" \
  FILENAME_GLOB="${FILENAME_GLOB:-Image*.png}" \
  NUM_WORKERS="${NUM_WORKERS:-6}" \
  INPUT_MODE="${INPUT_MODE:-rgb_plus_prompt}" \
  IMAGE_MODEL="${IMAGE_MODEL:-gpt-image-2}" \
  IMAGE_DEPLOYMENT="${IMAGE_DEPLOYMENT:-}" \
  GENERATION_MODE="${GENERATION_MODE:-edit}" \
  GENERATE_REQUIRES_IMAGE="${GENERATE_REQUIRES_IMAGE:-0}" \
  IMAGE_SIZE="${IMAGE_SIZE:-1536x1024}" \
  IMAGE_QUALITY="${IMAGE_QUALITY:-medium}" \
  IMAGE_SEED="${IMAGE_SEED:-123}" \
  DEFAULT_PROMPT="${DEFAULT_PROMPT:-Use the RGB image as primary evidence. Generate a sparse binary metallic map: black for dielectric and non-metal materials, white only for clearly exposed metal surfaces. Do not copy lighting, shadows, textures, object boundaries, highlights, or reflections into the metallic map.}" \
  MAX_GENERATE="${MAX_GENERATE:-}" \
  AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-}" \
  AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-}" \
  AZURE_METALLIC_OPENAI_API_KEY="${API_KEY}" \
  SKIP_CONDA="${SKIP_CONDA:-0}" \
  bash -lc "${START_CMD}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

DETACHED_PID=$!
echo "${DETACHED_PID}" > "${PID_FILE}"

echo "Started in background."
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
echo "Log file: ${LOG_FILE}"
