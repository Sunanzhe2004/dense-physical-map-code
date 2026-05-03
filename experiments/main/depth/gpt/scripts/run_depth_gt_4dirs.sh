#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_SCRIPT="${SCRIPT_ROOT}/depth_generation_gpt.py"
INPUT_ROOT="${INPUT_ROOT:-${GT_ROOT:-/path/to/benchmark_data/GT}}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/depth_gpt}"
LOG_DIR="${LOG_DIR:-${BASE_OUTPUT_DIR}/logs}"
WORKERS="${WORKERS:-4}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-*_im.png}"

COMMON_ARGS=(
  --input_mode rgb_only
  --depth_polarity near_white
  --image_model "${IMAGE_MODEL:-gpt-image-1.5}"
  --size "${IMAGE_SIZE:-1536x1024}"
  --skip_existing
)

if [[ "${MAX_GENERATE:-0}" != "0" ]]; then
  COMMON_ARGS+=(--max_generate "${MAX_GENERATE}")
fi

if [[ "${SLEEP:-}" != "" ]]; then
  COMMON_ARGS+=(--sleep "${SLEEP}")
fi

mkdir -p "${LOG_DIR}"

mapfile -t SCENE_DIRS < <(find "${INPUT_ROOT}" -type f -name "${FILENAME_SUFFIX}" -printf '%h\n' | sort -u)

if (( ${#SCENE_DIRS[@]} == 0 )); then
  echo "No files matching ${FILENAME_SUFFIX} were found under ${INPUT_ROOT}." >&2
  exit 1
fi

echo "Found ${#SCENE_DIRS[@]} directories containing ${FILENAME_SUFFIX}; using ${WORKERS} parallel workers."
echo "Output root: ${BASE_OUTPUT_DIR}"

PIDS=()

for ((worker_id = 0; worker_id < WORKERS; worker_id++)); do
  key_var="AZURE_GPT_IMAGE_15_API_KEY_${worker_id}"
  worker_api_key="${!key_var:-${AZURE_GPT_IMAGE_15_API_KEY:-${AZURE_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}}}"
  if [[ -z "${worker_api_key}" ]]; then
    echo "${key_var} or AZURE_GPT_IMAGE_15_API_KEY/AZURE_OPENAI_API_KEY/OPENAI_API_KEY is not set." >&2
    exit 1
  fi

  log_path="${LOG_DIR}/worker_${worker_id}.log"
  echo "Log file: ${log_path}"

  (
    set +e
    export AZURE_GPT_IMAGE_15_API_KEY="${worker_api_key}"
    status=0
    for ((i = worker_id; i < ${#SCENE_DIRS[@]}; i += WORKERS)); do
      input_dir="${SCENE_DIRS[$i]}"
      rel_dir="${input_dir#${INPUT_ROOT}/}"
      output_dir="${BASE_OUTPUT_DIR}/${rel_dir}"
      mkdir -p "${output_dir}"

      echo "[$(date '+%F %T')] worker=${worker_id} scene=$((i + 1))/${#SCENE_DIRS[@]} ${input_dir} -> ${output_dir}"
      "${PYTHON_BIN}" "${RUN_SCRIPT}" \
        --input_dir "${input_dir}" \
        --output_dir "${output_dir}" \
        "${COMMON_ARGS[@]}"
      rc=$?
      if (( rc != 0 )); then
        echo "[$(date '+%F %T')] ERROR rc=${rc}: ${input_dir}" >&2
        status=1
      fi
    done
    exit "${status}"
  ) > "${log_path}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status == 0 )); then
  echo "All depth gpt jobs completed."
else
  echo "Some depth gpt jobs failed. Check ${LOG_DIR}/worker_*.log." >&2
fi
exit "${status}"
