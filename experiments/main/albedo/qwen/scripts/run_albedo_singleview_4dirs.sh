#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/run_albedo_multiview_qwen.py"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/albedo_qwen}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"
GENERATION_MODE="${GENERATION_MODE:-edit}"
BASE_URL="${BASE_URL:-https://dashscope.aliyuncs.com/api/v1}"
ALBEDO_MODEL="${ALBEDO_MODEL:-wan2.7-image}"

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
SHARED_API_KEY="${DASHSCOPE_API_KEY:-}"

COMMON_ARGS=(
  --generation_mode "${GENERATION_MODE}"
  --base_url "${BASE_URL}"
  --albedo_model "${ALBEDO_MODEL}"
  --filename_suffix "${FILENAME_SUFFIX}"
  --recursive
  --preserve_relative_dirs
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ -n "${MAX_GENERATE:-}" ]]; then
  COMMON_ARGS+=(--max_generate "${MAX_GENERATE}")
fi

PIDS=()

mkdir -p "${LOG_DIR}"

for idx in 0 1 2 3; do
  input_dir="${INPUT_DIRS[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"

  if [[ -z "${SHARED_API_KEY}" ]]; then
    echo "DASHSCOPE_API_KEY is not set." >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  log_path="${LOG_DIR}/$(basename "${output_dir}").log"
  echo "Starting job ${idx}: ${input_dir} -> ${output_dir}"
  echo "Log file: ${log_path}"
  DASHSCOPE_API_KEY="${SHARED_API_KEY}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    "${COMMON_ARGS[@]}" \
    > "${log_path}" 2>&1 &
  PIDS+=("$!")
done

echo "All 4 directory jobs started. Waiting for completion..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "All directory runs completed."
