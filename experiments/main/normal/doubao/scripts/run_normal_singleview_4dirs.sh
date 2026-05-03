#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-micromamba}"
ENV_NAME="${ENV_NAME:-albedo}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/run_normal_multiview_oneshot_minimal_v3k_colorsem_texturepatch_v4.py"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/normal_doubao}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"
EXAMPLE_RGB="${EXAMPLE_RGB:-${SCRIPT_DIR}/image.png}"
EXAMPLE_NORMAL="${EXAMPLE_NORMAL:-${SCRIPT_DIR}/normal.png}"

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

API_KEYS=(
  "${ARK_API_KEY_0:-your_key_0}"
  "${ARK_API_KEY_1:-your_key_1}"
  "${ARK_API_KEY_2:-your_key_2}"
  "${ARK_API_KEY_3:-your_key_3}"
)

COMMON_ARGS=(
  --example_rgb "${EXAMPLE_RGB}"
  --example_normal "${EXAMPLE_NORMAL}"
  --independent_images
  --filename_suffix "${FILENAME_SUFFIX}"
  --recursive
  --preserve_relative_dirs
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ ! -f "${EXAMPLE_RGB}" ]]; then
  echo "example_rgb does not exist: ${EXAMPLE_RGB}" >&2
  exit 1
fi

if [[ ! -f "${EXAMPLE_NORMAL}" ]]; then
  echo "example_normal does not exist: ${EXAMPLE_NORMAL}" >&2
  exit 1
fi

if ! command -v "${MICROMAMBA_BIN}" >/dev/null 2>&1; then
  echo "micromamba was not found: ${MICROMAMBA_BIN}" >&2
  exit 1
fi

PIDS=()

mkdir -p "${LOG_DIR}"

for idx in 0 1 2 3; do
  input_dir="${INPUT_DIRS[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"
  api_key="${API_KEYS[$idx]}"

  if [[ -z "${api_key}" || "${api_key}" == your_key_* ]]; then
    echo "API key ${idx} is not set correctly. Check ARK_API_KEY_${idx}." >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  log_path="${LOG_DIR}/$(basename "${output_dir}").log"
  echo "Starting job ${idx}: ${input_dir} -> ${output_dir}"
  echo "Log file: ${log_path}"
  ARK_API_KEY="${api_key}" "${MICROMAMBA_BIN}" run -n "${ENV_NAME}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
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
