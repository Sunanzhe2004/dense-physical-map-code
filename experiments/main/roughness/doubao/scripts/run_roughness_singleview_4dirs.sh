#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/roughness_generation_doubao_last.py"
GT_ROOT="${GT_ROOT:-/path/to/benchmark_data/GT}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/roughness_doubao}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}"
INPUT_MODE="${INPUT_MODE:-rgb_only}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"

INPUT_DIRS=(
  "${GT_ROOT}/interiorverse_mainaxis"
  "${GT_ROOT}/interiorverse_stresstest"
  "${GT_ROOT}/openroomff_mainaxis"
  "${GT_ROOT}/openroomff_stresstest"
)

SEG_DIRS=(
  "${GT_ROOT}/interiorverse_mainaxis"
  "${GT_ROOT}/interiorverse_stresstest"
  "${GT_ROOT}/openroomff_mainaxis"
  "${GT_ROOT}/openroomff_stresstest"
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
  --input_mode "${INPUT_MODE}"
  --filename_suffix "${FILENAME_SUFFIX}"
  --recursive
  --preserve_relative_dirs
  --skip_existing
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

if [[ -n "${MAX_GENERATE:-}" ]]; then
  COMMON_ARGS+=(--max_generate "${MAX_GENERATE}")
fi

if [[ -n "${SIZE:-}" ]]; then
  COMMON_ARGS+=(--size "${SIZE}")
fi

if [[ -n "${SLEEP:-}" ]]; then
  COMMON_ARGS+=(--sleep "${SLEEP}")
fi

if [[ "${INPUT_MODE}" == "rgb_plus_example" ]]; then
  if [[ -z "${EXAMPLE_RGB:-}" || -z "${EXAMPLE_ROUGHNESS:-}" ]]; then
    echo "INPUT_MODE=rgb_plus_example requires EXAMPLE_RGB and EXAMPLE_ROUGHNESS." >&2
    exit 1
  fi
  COMMON_ARGS+=(--example_rgb "${EXAMPLE_RGB}" --example_roughness "${EXAMPLE_ROUGHNESS}")
fi

PIDS=()

mkdir -p "${LOG_DIR}"

for idx in 0 1 2 3; do
  input_dir="${INPUT_DIRS[$idx]}"
  seg_dir="${SEG_DIRS[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"
  api_key="${API_KEYS[$idx]}"

  if [[ -z "${api_key}" || "${api_key}" == your_key_* ]]; then
    echo "API key ${idx} is not set. Please set ARK_API_KEY_${idx}." >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  log_path="${LOG_DIR}/$(basename "${output_dir}").log"
  echo "Start task ${idx}: ${input_dir} -> ${output_dir}"
  echo "Log file: ${log_path}"

  EXTRA_ARGS=()
  if [[ "${INPUT_MODE}" == "rgb_plus_seg" ]]; then
    EXTRA_ARGS+=(--seg_dir "${seg_dir}")
  fi

  ARK_API_KEY="${api_key}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    --input_dir "${input_dir}" \
    --output_dir "${output_dir}" \
    "${COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    > "${log_path}" 2>&1 &
  PIDS+=("$!")
done

echo "4 directory tasks started; waiting for all of them..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "All directory tasks finished."
