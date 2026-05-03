#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_ROOT}/metallic_generation_doubao_final.py}"

INPUT_DIR="${INPUT_DIR:-/path/to/benchmark_data/metallic_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/benchmark_outputs/metallic_doubao}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
STAGING_DIR="${STAGING_DIR:-${OUTPUT_DIR}/staging_prompts}"
PROMPT_SOURCE_DIR="${PROMPT_SOURCE_DIR:-}"

FILENAME_GLOB="${FILENAME_GLOB:-Image*.png}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Use the RGB image as primary evidence. Generate a sparse binary metallic map: black for dielectric and non-metal materials, white only for clearly exposed metal surfaces. Do not copy lighting, shadows, textures, object boundaries, highlights, or reflections into the metallic map.}"

mkdir -p "${LOG_DIR}" "${STAGING_DIR}" "${OUTPUT_DIR}"

mapfile -t RGB_DIRS < <(find "${INPUT_DIR}" -type f -iname "${FILENAME_GLOB}" -printf '%h\n' | sort -u)
if (( ${#RGB_DIRS[@]} == 0 )); then
  echo "No files matching ${FILENAME_GLOB} were found under ${INPUT_DIR}." >&2
  exit 1
fi

KEY_VARS=()
for ((worker_idx = 0; worker_idx < NUM_WORKERS; worker_idx++)); do
  key_var="ARK_API_KEY_${worker_idx}"
  if [[ -z "${!key_var:-}" ]]; then
    echo "${key_var} is not set." >&2
    exit 1
  fi
  KEY_VARS+=("${key_var}")
done

prepare_prompt_dir() {
  local rgb_dir="$1"
  local prompt_dir="$2"
  local image_path rel_path rel_no_ext source_prompt alt_prompt prompt_path

  rm -rf "${prompt_dir}"
  mkdir -p "${prompt_dir}"

  while IFS= read -r image_path; do
    rel_path="${image_path#${INPUT_DIR}/}"
    rel_no_ext="${rel_path%.*}"
    prompt_path="${prompt_dir}/$(basename "${image_path%.*}")_prompt.txt"
    source_prompt=""
    alt_prompt=""

    if [[ -n "${PROMPT_SOURCE_DIR}" ]]; then
      source_prompt="${PROMPT_SOURCE_DIR}/${rel_no_ext}_prompt.txt"
      alt_prompt="${PROMPT_SOURCE_DIR}/$(basename "${image_path%.*}")_prompt.txt"
    fi

    if [[ -n "${source_prompt}" && -f "${source_prompt}" ]]; then
      ln -sf "${source_prompt}" "${prompt_path}"
    elif [[ -n "${alt_prompt}" && -f "${alt_prompt}" ]]; then
      ln -sf "${alt_prompt}" "${prompt_path}"
    else
      printf "%s\n" "${DEFAULT_PROMPT}" > "${prompt_path}"
    fi
  done < <(find "${rgb_dir}" -maxdepth 1 -type f -iname "${FILENAME_GLOB}" | sort)
}

WORKER_LIST_DIR="${STAGING_DIR}/worker_lists"
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
  local rgb_dir rel_dir output_subdir prompt_subdir log_path

  while IFS= read -r rgb_dir; do
    [[ -z "${rgb_dir}" ]] && continue
    rel_dir="${rgb_dir#${INPUT_DIR}/}"
    output_subdir="${OUTPUT_DIR}/${rel_dir}"
    prompt_subdir="${STAGING_DIR}/${rel_dir}"
    log_path="${LOG_DIR}/${rel_dir//\//__}.log"

    prepare_prompt_dir "${rgb_dir}" "${prompt_subdir}"

    echo "worker ${worker_idx}: ${rgb_dir} -> ${output_subdir}"
    echo "Log file: ${log_path}"

    ARGS=(
      --input_dir "${rgb_dir}"
      --output_dir "${output_subdir}"
      --prompt_dir "${prompt_subdir}"
      --skip_existing
    )
    if [[ -n "${MAX_GENERATE:-}" ]]; then
      ARGS+=(--max_generate "${MAX_GENERATE}")
    fi

    ARK_API_KEY="${key_value}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
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
