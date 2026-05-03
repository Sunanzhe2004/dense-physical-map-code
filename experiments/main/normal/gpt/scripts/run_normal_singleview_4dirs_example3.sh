#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/run_normal_multiview_oneshot_gpt_edit.py"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/normal_gpt_example3}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"
EXAMPLE_RGB="${EXAMPLE_RGB:-${SCRIPT_ROOT}/../examples/image3.png}"
EXAMPLE_NORMAL="${EXAMPLE_NORMAL:-${SCRIPT_ROOT}/../examples/normal3.png}"

API_KEYS=(
  "${AZURE_NORMAL_OPENAI_API_KEY_0:-${AZURE_NORMAL_OPENAI_API_KEY:-}}"
  "${AZURE_NORMAL_OPENAI_API_KEY_1:-${AZURE_NORMAL_OPENAI_API_KEY:-}}"
  "${AZURE_NORMAL_OPENAI_API_KEY_2:-${AZURE_NORMAL_OPENAI_API_KEY:-}}"
  "${AZURE_NORMAL_OPENAI_API_KEY_3:-${AZURE_NORMAL_OPENAI_API_KEY:-}}"
)

ENDPOINTS=(
  "${AZURE_NORMAL_OPENAI_ENDPOINT_0:-${AZURE_NORMAL_OPENAI_ENDPOINT:-}}"
  "${AZURE_NORMAL_OPENAI_ENDPOINT_1:-${AZURE_NORMAL_OPENAI_ENDPOINT:-}}"
  "${AZURE_NORMAL_OPENAI_ENDPOINT_2:-${AZURE_NORMAL_OPENAI_ENDPOINT:-}}"
  "${AZURE_NORMAL_OPENAI_ENDPOINT_3:-${AZURE_NORMAL_OPENAI_ENDPOINT:-}}"
)

API_VERSIONS=(
  "${AZURE_NORMAL_OPENAI_API_VERSION_0:-${AZURE_NORMAL_OPENAI_API_VERSION:-2025-04-01-preview}}"
  "${AZURE_NORMAL_OPENAI_API_VERSION_1:-${AZURE_NORMAL_OPENAI_API_VERSION:-2025-04-01-preview}}"
  "${AZURE_NORMAL_OPENAI_API_VERSION_2:-${AZURE_NORMAL_OPENAI_API_VERSION:-2025-04-01-preview}}"
  "${AZURE_NORMAL_OPENAI_API_VERSION_3:-${AZURE_NORMAL_OPENAI_API_VERSION:-2025-04-01-preview}}"
)

INPUT_DIRS=(
  "${GT_ROOT:-/path/to/benchmark_data/GT}/interiorverse_mainaxis"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/interiorverse_stresstest"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/openroomff_mainaxis"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/openroomff_stresstest"
)

OUTPUT_DIRS=(
  "${BASE_OUTPUT_DIR}/interiorverse_mainaxis"
  "${BASE_OUTPUT_DIR}/interiorverse_stresstest"
  "${BASE_OUTPUT_DIR}/openroomff_mainaxis"
  "${BASE_OUTPUT_DIR}/openroomff_stresstest"
)

COMMON_ARGS=(
  --example_rgb "${EXAMPLE_RGB}"
  --example_normal "${EXAMPLE_NORMAL}"
  --filename_suffix "${FILENAME_SUFFIX}"
  --recursive
  --preserve_relative_dirs
)

if [[ -n "${MAX_GENERATE:-}" ]]; then
  COMMON_ARGS+=(--max_generate "${MAX_GENERATE}")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ "${SAVE_DEBUG_INTERMEDIATES:-0}" == "1" ]]; then
  COMMON_ARGS+=(--save_debug_intermediates)
fi

if [[ -n "${NORMAL_SIZE:-}" ]]; then
  COMMON_ARGS+=(--normal_size "${NORMAL_SIZE}")
fi

if [[ -n "${NORMAL_QUALITY:-}" ]]; then
  COMMON_ARGS+=(--normal_quality "${NORMAL_QUALITY}")
fi

if [[ ! -f "${EXAMPLE_RGB}" ]]; then
  echo "example_rgb does not exist: ${EXAMPLE_RGB}" >&2
  exit 1
fi

if [[ ! -f "${EXAMPLE_NORMAL}" ]]; then
  echo "example_normal does not exist: ${EXAMPLE_NORMAL}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
PIDS=()

for idx in 0 1 2 3; do
  input_dir="${INPUT_DIRS[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"
  api_key="${API_KEYS[$idx]}"
  endpoint="${ENDPOINTS[$idx]}"
  api_version="${API_VERSIONS[$idx]}"

  if [[ -z "${api_key}" ]]; then
    echo "API key ${idx} is not set. Check AZURE_NORMAL_OPENAI_API_KEY_${idx} or AZURE_NORMAL_OPENAI_API_KEY." >&2
    exit 1
  fi

  if [[ -z "${endpoint}" ]]; then
    echo "Endpoint ${idx} is not set. Check AZURE_NORMAL_OPENAI_ENDPOINT_${idx} or AZURE_NORMAL_OPENAI_ENDPOINT." >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  log_path="${LOG_DIR}/$(basename "${output_dir}").log"
  echo "Starting job ${idx}: ${input_dir} -> ${output_dir}"
  echo "Log file: ${log_path}"

  "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    --normal_api_key "${api_key}" \
    --normal_endpoint "${endpoint}" \
    --normal_api_version "${api_version}" \
    "${COMMON_ARGS[@]}" \
    > "${log_path}" 2>&1 &
  PIDS+=("$!")
done

echo "All 4 directory jobs started. Waiting for completion..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "All directory runs completed."
