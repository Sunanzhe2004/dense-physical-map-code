#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/run_albedo_multiview_seed2.py"

INPUT_DIR="${GT_ROOT:-/path/to/benchmark_data/GT}/interiorverse_mainaxis"
OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/albedo_doubao}/interiorverse_mainaxis"

API_KEYS=(
  "${ARK_API_KEY_0:-your_key_0}"
  "${ARK_API_KEY_1:-your_key_1}"
  "${ARK_API_KEY_2:-your_key_2}"
  "${ARK_API_KEY_3:-your_key_3}"
)

COMMON_ARGS=(
  --input_dir "${INPUT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --max_views 1
  --independent_images
  --filename_suffix _im.png
  --recursive
  --preserve_relative_dirs
  --num_parts 4
)

PIDS=()

for part_index in 0 1 2 3; do
  api_key="${API_KEYS[$part_index]}"
  if [[ -z "${api_key}" || "${api_key}" == your_key_* ]]; then
    echo "API key ${part_index} is not set correctly. Check ARK_API_KEY_${part_index}." >&2
    exit 1
  fi

  echo "Starting interiorverse_mainaxis shard ${part_index}/4"
  ARK_API_KEY="${api_key}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --part_index "${part_index}" &
  PIDS+=("$!")
done

echo "All 4 shards started. Waiting for completion..."

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "interiorverse_mainaxis 4-shard run completed."
