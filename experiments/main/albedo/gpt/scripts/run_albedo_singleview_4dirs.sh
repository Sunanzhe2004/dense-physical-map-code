#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/run_albedo_multiview_gpt.py"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/albedo_gpt}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"

ALBEDO_ENDPOINT="${ALBEDO_ENDPOINT:-${AZURE_ALBEDO_OPENAI_ENDPOINT:-${AZURE_GPT_IMAGE_15_ENDPOINT:-}}}"
ALBEDO_API_VERSION="${ALBEDO_API_VERSION:-${AZURE_ALBEDO_OPENAI_API_VERSION:-${AZURE_GPT_IMAGE_15_API_VERSION:-2025-04-01-preview}}}"
ALBEDO_MODEL="${ALBEDO_MODEL:-gpt-image-1.5}"
ALBEDO_DEPLOYMENT="${ALBEDO_DEPLOYMENT:-gpt-image-1.5}"
ALBEDO_API_KEY="${ALBEDO_API_KEY:-${AZURE_ALBEDO_OPENAI_API_KEY:-${AZURE_GPT_IMAGE_15_API_KEY:-}}}"

INPUT_DIRS=(
  "${GT_ROOT:-/path/to/benchmark_data/GT}/interiorverse_mainaxis"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/openroomff_stresstest"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/interiorverse_stresstest"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/openroomff_mainaxis"
)

OUTPUT_DIRS=(
  "${BASE_OUTPUT_DIR}/interiorverse_mainaxis"
  "${BASE_OUTPUT_DIR}/openroomff_stresstest"
  "${BASE_OUTPUT_DIR}/interiorverse_stresstest"
  "${BASE_OUTPUT_DIR}/openroomff_mainaxis"
)

COMMON_ARGS=(
  --filename_suffix "${FILENAME_SUFFIX}"
  --recursive
  --preserve_relative_dirs
  --albedo_model "${ALBEDO_MODEL}"
  --albedo_deployment "${ALBEDO_DEPLOYMENT}"
  --albedo_endpoint "${ALBEDO_ENDPOINT}"
  --albedo_api_version "${ALBEDO_API_VERSION}"
  --generation_mode "${GENERATION_MODE:-edit}"
  --albedo_size "${ALBEDO_SIZE:-1536x1024}"
  --albedo_quality "${ALBEDO_QUALITY:-medium}"
  --timeout "${TIMEOUT:-600}"
  --request_retries "${REQUEST_RETRIES:-5}"
  --retry_backoff "${RETRY_BACKOFF:-5}"
  --retry_max_backoff "${RETRY_MAX_BACKOFF:-60}"
)

if [[ -n "${MAX_GENERATE:-}" ]]; then
  COMMON_ARGS+=(--max_generate "${MAX_GENERATE}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ "${GENERATE_REQUIRES_IMAGE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--generate_requires_image)
fi

PIDS=()

mkdir -p "${LOG_DIR}"

if [[ -z "${ALBEDO_ENDPOINT}" ]]; then
  echo "ALBEDO_ENDPOINT is not set (or AZURE_ALBEDO_OPENAI_ENDPOINT / AZURE_GPT_IMAGE_15_ENDPOINT)." >&2
  exit 1
fi

if [[ -z "${ALBEDO_API_KEY}" ]]; then
  echo "ALBEDO_API_KEY is not set (or AZURE_ALBEDO_OPENAI_API_KEY / AZURE_GPT_IMAGE_15_API_KEY)." >&2
  exit 1
fi

for idx in 0 1 2 3; do
  input_dir="${INPUT_DIRS[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"

  mkdir -p "${output_dir}"
  log_path="${LOG_DIR}/$(basename "${output_dir}").log"
  echo "Starting job ${idx}: ${input_dir} -> ${output_dir}"
  echo "Log file: ${log_path}"
  "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --albedo_api_key "${ALBEDO_API_KEY}" \
    "${COMMON_ARGS[@]}" \
    > "${log_path}" 2>&1 &
  PIDS+=("$!")
done

echo "All 4 directory jobs started. Waiting for completion..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "All directory runs completed."
