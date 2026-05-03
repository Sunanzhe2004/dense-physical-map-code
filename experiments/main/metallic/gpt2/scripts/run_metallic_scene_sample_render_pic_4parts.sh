#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_ROOT}/metallic_generation_gpt2.py}"

INPUT_DIR="${INPUT_DIR:-/path/to/benchmark_data/metallic_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/benchmark_outputs/metallic_gpt2}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
STAGING_DIR="${STAGING_DIR:-${OUTPUT_DIR}/staging_prompts}"
PROMPT_SOURCE_DIR="${PROMPT_SOURCE_DIR:-}"

FILENAME_GLOB="${FILENAME_GLOB:-Image*.png}"
NUM_WORKERS="${NUM_WORKERS:-6}"
INPUT_MODE="${INPUT_MODE:-rgb_plus_prompt}"
IMAGE_MODEL="${IMAGE_MODEL:-gpt-image-2}"
GENERATION_MODE="${GENERATION_MODE:-edit}"
IMAGE_SIZE="${IMAGE_SIZE:-1536x1024}"
IMAGE_QUALITY="${IMAGE_QUALITY:-medium}"
IMAGE_SEED="${IMAGE_SEED:-123}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Use the RGB image as primary evidence. Generate a sparse binary metallic map: black for dielectric and non-metal materials, white only for clearly exposed metal surfaces. Do not copy lighting, shadows, textures, object boundaries, highlights, or reflections into the metallic map.}"

mkdir -p "${LOG_DIR}" "${STAGING_DIR}" "${OUTPUT_DIR}"

mapfile -t RGB_DIRS < <(
  find "${INPUT_DIR}" \
    -type f \
    -path '*/V*_P*_L*/Image/*/Image*.png' \
    -printf '%h\n' | sort -u
)
if (( ${#RGB_DIRS[@]} == 0 )); then
  echo "No image directories matching */V*_P*_L*/Image/*/Image*.png were found under ${INPUT_DIR}." >&2
  exit 1
fi

API_KEY="${AZURE_METALLIC_OPENAI_API_KEY:-${AZURE_GPT_IMAGE_2_API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}}"
API_ENDPOINT="${AZURE_METALLIC_OPENAI_ENDPOINT:-${AZURE_GPT_IMAGE_2_ENDPOINT:-${AZURE_OPENAI_ENDPOINT:-}}}"
API_VERSION="${AZURE_METALLIC_OPENAI_API_VERSION:-${AZURE_GPT_IMAGE_2_API_VERSION:-${AZURE_OPENAI_API_VERSION:-2025-04-01-preview}}}"
if [[ -z "${API_KEY}" ]]; then
  echo "GPT Image API key is not set. Set one of AZURE_METALLIC_OPENAI_API_KEY / AZURE_GPT_IMAGE_2_API_KEY / AZURE_OPENAI_API_KEY / OPENAI_API_KEY." >&2
  exit 1
fi
if [[ -z "${API_ENDPOINT}" ]]; then
  echo "GPT Image endpoint is not set. Set one of AZURE_METALLIC_OPENAI_ENDPOINT / AZURE_GPT_IMAGE_2_ENDPOINT / AZURE_OPENAI_ENDPOINT." >&2
  exit 1
fi

normalize_path() {
  local path="$1"
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$path"
}

INPUT_DIR_ABS="$(normalize_path "${INPUT_DIR}")"

relative_to_input_dir() {
  local path="$1"
  local abs_path
  abs_path="$(normalize_path "${path}")"
  python3 -c '
import os, sys
base = os.path.abspath(sys.argv[1])
path = os.path.abspath(sys.argv[2])
try:
    rel = os.path.relpath(path, base)
except Exception:
    rel = path
print("." if rel == "." else rel)
' "${INPUT_DIR_ABS}" "${abs_path}"
}

prepare_prompt_dir() {
  local rgb_dir="$1"
  local prompt_dir="$2"
  local image_path rel_path rel_no_ext source_prompt alt_prompt prompt_path

  rm -rf "${prompt_dir}"
  mkdir -p "${prompt_dir}"

  while IFS= read -r image_path; do
    rel_path="$(relative_to_input_dir "${image_path}")"
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
  local list_path="${WORKER_LIST_DIR}/worker_${worker_idx}.txt"
  local rgb_dir rel_dir output_subdir prompt_subdir log_path

  while IFS= read -r rgb_dir; do
    [[ -z "${rgb_dir}" ]] && continue
    rel_dir="$(relative_to_input_dir "${rgb_dir}")"
    if [[ "${rel_dir}" == "." ]]; then
      output_subdir="${OUTPUT_DIR}"
      prompt_subdir="${STAGING_DIR}/root"
      log_path="${LOG_DIR}/root.log"
    else
      output_subdir="${OUTPUT_DIR}/${rel_dir}"
      prompt_subdir="${STAGING_DIR}/${rel_dir}"
      log_path="${LOG_DIR}/${rel_dir//\//__}.log"
    fi

    prepare_prompt_dir "${rgb_dir}" "${prompt_subdir}"

    echo "worker ${worker_idx}: ${rgb_dir} -> ${output_subdir}"
    echo "Log file: ${log_path}"

    ARGS=(
      --input_dir "${rgb_dir}"
      --output_dir "${output_subdir}"
      --input_mode "${INPUT_MODE}"
      --image_model "${IMAGE_MODEL}"
      --generation_mode "${GENERATION_MODE}"
      --size "${IMAGE_SIZE}"
      --quality "${IMAGE_QUALITY}"
      --seed "${IMAGE_SEED}"
      --skip_existing
    )

    if [[ "${INPUT_MODE}" == "rgb_plus_prompt" ]]; then
      ARGS+=(--prompt_dir "${prompt_subdir}")
    fi

    ARGS+=(--azure_endpoint "${API_ENDPOINT}")
    ARGS+=(--api_version "${API_VERSION}")
    if [[ -n "${IMAGE_DEPLOYMENT:-}" ]]; then
      ARGS+=(--image_deployment "${IMAGE_DEPLOYMENT}")
    fi
    if [[ "${GENERATE_REQUIRES_IMAGE:-0}" == "1" ]]; then
      ARGS+=(--generate_requires_image)
    fi
    if [[ -n "${MAX_GENERATE:-}" ]]; then
      ARGS+=(--max_generate "${MAX_GENERATE}")
    fi
    if [[ "${SAVE_DEBUG_INTERMEDIATES:-0}" == "1" ]]; then
      ARGS+=(--save_debug_intermediates)
    fi

    AZURE_METALLIC_OPENAI_API_KEY="${API_KEY}" "${PYTHON_BIN}" "${RUN_SCRIPT}" \
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
