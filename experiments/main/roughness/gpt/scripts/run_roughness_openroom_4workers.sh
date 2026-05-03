#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/roughness_generation_gpt.py"
GT_ROOT="${GT_ROOT:-/path/to/benchmark_data/GT}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/roughness_gpt}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-_im.png}"
INPUT_MODE="${INPUT_MODE:-rgb_only}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"
PARTS_PER_DATASET="${PARTS_PER_DATASET:-2}"

DATASET_NAMES=(
  "openroomff_mainaxis"
  "openroomff_stresstest"
)

INPUT_DIRS=(
  "${GT_ROOT}/openroomff_mainaxis"
  "${GT_ROOT}/openroomff_stresstest"
)

SEG_DIRS=(
  "${GT_ROOT}/openroomff_mainaxis"
  "${GT_ROOT}/openroomff_stresstest"
)

OUTPUT_DIRS=(
  "${BASE_OUTPUT_DIR}/openroomff_mainaxis"
  "${BASE_OUTPUT_DIR}/openroomff_stresstest"
)

API_KEY="${AZURE_ROUGHNESS_OPENAI_API_KEY:-${AZURE_GPT_IMAGE_15_API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-${GPT_IMAGE_API_KEY:-}}}}}"

if [[ -z "${API_KEY}" ]]; then
  echo "No GPT image API key is set. Set AZURE_ROUGHNESS_OPENAI_API_KEY, AZURE_OPENAI_API_KEY, OPENAI_API_KEY, or GPT_IMAGE_API_KEY." >&2
  exit 1
fi

if (( PARTS_PER_DATASET < 1 )); then
  echo "PARTS_PER_DATASET must be >= 1." >&2
  exit 1
fi

COMMON_ARGS=(
  --input_mode "${INPUT_MODE}"
  --filename_suffix "${FILENAME_SUFFIX}"
  --recursive
  --preserve_relative_dirs
  --num_parts "${PARTS_PER_DATASET}"
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

if [[ -n "${IMAGE_MODEL:-}" ]]; then
  COMMON_ARGS+=(--image_model "${IMAGE_MODEL}")
fi

if [[ -n "${IMAGE_QUALITY:-}" ]]; then
  COMMON_ARGS+=(--image_quality "${IMAGE_QUALITY}")
fi

if [[ -n "${BASE_URL:-}" ]]; then
  COMMON_ARGS+=(--base_url "${BASE_URL}")
fi

if [[ -n "${API_VERSION:-}" ]]; then
  COMMON_ARGS+=(--api_version "${API_VERSION}")
fi

if [[ "${SAVE_DEBUG_INTERMEDIATES:-0}" == "1" ]]; then
  COMMON_ARGS+=(--save_debug_intermediates)
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

for dataset_idx in 0 1; do
  dataset_name="${DATASET_NAMES[$dataset_idx]}"
  input_dir="${INPUT_DIRS[$dataset_idx]}"
  seg_dir="${SEG_DIRS[$dataset_idx]}"
  output_dir="${OUTPUT_DIRS[$dataset_idx]}"

  mkdir -p "${output_dir}"

  for ((part_index = 0; part_index < PARTS_PER_DATASET; part_index++)); do
    log_path="${LOG_DIR}/${dataset_name}.part$((part_index + 1))of${PARTS_PER_DATASET}.log"
    echo "Start ${dataset_name} shard $((part_index + 1))/${PARTS_PER_DATASET}: ${input_dir} -> ${output_dir}"
    echo "Log file: ${log_path}"

    EXTRA_ARGS=()
    if [[ "${INPUT_MODE}" == "rgb_plus_seg" ]]; then
      EXTRA_ARGS+=(--seg_dir "${seg_dir}")
    fi

    AZURE_ROUGHNESS_OPENAI_API_KEY="${API_KEY}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
      --input_dir "${input_dir}" \
      --output_dir "${output_dir}" \
      "${COMMON_ARGS[@]}" \
      --part_index "${part_index}" \
      "${EXTRA_ARGS[@]}" \
      > "${log_path}" 2>&1 &
    PIDS+=("$!")
  done
done

echo "$((2 * PARTS_PER_DATASET)) openroom workers started; waiting for all of them..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "All openroom workers finished."
