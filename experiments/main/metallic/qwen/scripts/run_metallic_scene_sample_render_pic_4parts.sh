#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_ROOT}/metallic_generation_wan_filltight_final.py}"

INPUT_DIR="${INPUT_DIR:-/path/to/benchmark_data/metallic_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/benchmark_outputs/metallic_qwen}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
FILENAME_GLOB="${FILENAME_GLOB:-Image*.png}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

mapfile -t RGB_DIRS < <(find "${INPUT_DIR}" -type f -iname "${FILENAME_GLOB}" -printf '%h\n' | sort -u)
if (( ${#RGB_DIRS[@]} == 0 )); then
  echo "No files matching ${FILENAME_GLOB} were found under ${INPUT_DIR}." >&2
  exit 1
fi

KEY_VARS=()
for ((worker_idx = 0; worker_idx < NUM_WORKERS; worker_idx++)); do
  key_var="DASHSCOPE_API_KEY_${worker_idx}"
  if [[ -z "${!key_var:-}" ]]; then
    echo "${key_var} is not set." >&2
    exit 1
  fi
  KEY_VARS+=("${key_var}")
done

WORKER_LIST_DIR="${OUTPUT_DIR}/worker_lists"
rm -rf "${WORKER_LIST_DIR}"
mkdir -p "${WORKER_LIST_DIR}"

for ((worker_idx = 0; worker_idx < NUM_WORKERS; worker_idx++)); do
  : > "${WORKER_LIST_DIR}/worker_${worker_idx}.txt"
done

for dir_index in "${!RGB_DIRS[@]}"; do
  worker_idx=$((dir_index % NUM_WORKERS))
  printf "%s\n" "${RGB_DIRS[$dir_index]}" >> "${WORKER_LIST_DIR}/worker_${worker_idx}.txt"
done

run_worker() {
  local worker_idx="$1"
  local key_var="${KEY_VARS[$worker_idx]}"
  local key_value="${!key_var}"
  local list_path="${WORKER_LIST_DIR}/worker_${worker_idx}.txt"
  local rgb_dir rel_dir output_subdir log_path

  while IFS= read -r rgb_dir; do
    [[ -z "${rgb_dir}" ]] && continue
    rel_dir="${rgb_dir#${INPUT_DIR}/}"
    output_subdir="${OUTPUT_DIR}/${rel_dir}"
    log_path="${LOG_DIR}/${rel_dir//\//__}.log"

    echo "worker ${worker_idx}: ${rgb_dir} -> ${output_subdir}"
    echo "Log file: ${log_path}"

    ARGS=(
      --input_dir "${rgb_dir}"
      --output_dir "${output_subdir}"
      --skip_existing
    )
    if [[ -n "${MAX_GENERATE:-}" ]]; then
      ARGS+=(--max_generate "${MAX_GENERATE}")
    fi

    DASHSCOPE_API_KEY="${key_value}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
      "${ARGS[@]}" \
      > "${log_path}" 2>&1
  done < "${list_path}"
}

PIDS=()
for ((worker_idx = 0; worker_idx < NUM_WORKERS; worker_idx++)); do
  run_worker "${worker_idx}" &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "All metallic directories completed; processed ${#RGB_DIRS[@]} RGB subdirectories."
