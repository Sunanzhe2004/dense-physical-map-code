#!/usr/bin/env bash

set -euo pipefail

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/normal_gpt_example3}"
FILENAME_SUFFIX="${FILENAME_SUFFIX:-*_im.png}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-*_im_normal.png}"

INPUT_DIRS=(
  "${GT_ROOT:-/path/to/benchmark_data/GT}/interiorverse_mainaxis"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/interiorverse_stresstest"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/openroomff_mainaxis"
  "${GT_ROOT:-/path/to/benchmark_data/GT}/openroomff_stresstest"
)

OUTPUT_DIRS=(
  "${BASE_OUTPUT_DIR}/interiorverse_mainaxis/normal"
  "${BASE_OUTPUT_DIR}/interiorverse_stresstest/normal"
  "${BASE_OUTPUT_DIR}/openroomff_mainaxis/normal"
  "${BASE_OUTPUT_DIR}/openroomff_stresstest/normal"
)

SKIP_FILES=(
  "${BASE_OUTPUT_DIR}/interiorverse_mainaxis/meta/skipped_images.json"
  "${BASE_OUTPUT_DIR}/interiorverse_stresstest/meta/skipped_images.json"
  "${BASE_OUTPUT_DIR}/openroomff_mainaxis/meta/skipped_images.json"
  "${BASE_OUTPUT_DIR}/openroomff_stresstest/meta/skipped_images.json"
)

DATASET_NAMES=(
  "interiorverse_mainaxis"
  "interiorverse_stresstest"
  "openroomff_mainaxis"
  "openroomff_stresstest"
)

count_json_items() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo 0
    return
  fi
  python3 - "$path" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(len(data) if isinstance(data, list) else 0)
except Exception:
    print(0)
PY
}

total_input=0
total_output=0
total_skipped=0

printf "%-28s %10s %10s %10s %10s %10s\n" "dataset" "input" "done" "skipped" "remain" "percent"
printf "%-28s %10s %10s %10s %10s %10s\n" "----------------------------" "----------" "----------" "----------" "----------" "----------"

for idx in 0 1 2 3; do
  dataset_name="${DATASET_NAMES[$idx]}"
  input_dir="${INPUT_DIRS[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"
  skip_file="${SKIP_FILES[$idx]}"

  input_count=$(find "${input_dir}" -type f -name "${FILENAME_SUFFIX}" | wc -l | tr -d ' ')
  if [[ -d "${output_dir}" ]]; then
    output_count=$(find "${output_dir}" -type f -name "${OUTPUT_SUFFIX}" | wc -l | tr -d ' ')
  else
    output_count=0
  fi
  skipped_count=$(count_json_items "${skip_file}")

  remain_count=$((input_count - output_count - skipped_count))
  if (( remain_count < 0 )); then
    remain_count=0
  fi

  if (( input_count > 0 )); then
    percent=$(awk -v a="$((output_count + skipped_count))" -v b="${input_count}" 'BEGIN { printf "%.2f%%", (a / b) * 100 }')
  else
    percent="0.00%"
  fi

  printf "%-28s %10d %10d %10d %10d %10s\n" \
    "${dataset_name}" "${input_count}" "${output_count}" "${skipped_count}" "${remain_count}" "${percent}"

  total_input=$((total_input + input_count))
  total_output=$((total_output + output_count))
  total_skipped=$((total_skipped + skipped_count))
done

total_remain=$((total_input - total_output - total_skipped))
if (( total_remain < 0 )); then
  total_remain=0
fi

if (( total_input > 0 )); then
  total_percent=$(awk -v a="$((total_output + total_skipped))" -v b="${total_input}" 'BEGIN { printf "%.2f%%", (a / b) * 100 }')
else
  total_percent="0.00%"
fi

printf "%-28s %10s %10s %10s %10s %10s\n" "----------------------------" "----------" "----------" "----------" "----------" "----------"
printf "%-28s %10d %10d %10d %10d %10s\n" "TOTAL" "${total_input}" "${total_output}" "${total_skipped}" "${total_remain}" "${total_percent}"
