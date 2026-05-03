#!/usr/bin/env bash

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-/path/to/benchmark_data/metallic_scene}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/benchmark_outputs/metallic_qwen}"
FILENAME_GLOB="${FILENAME_GLOB:-Image*.png}"

input_total=$(find "${INPUT_DIR}" -type f -iname "${FILENAME_GLOB}" | wc -l | tr -d ' ')
if [[ -d "${OUTPUT_DIR}" ]]; then
  done_total=$(find "${OUTPUT_DIR}" -type f -name '*_metallic.png' | wc -l | tr -d ' ')
else
  done_total=0
fi

remain_total=$((input_total - done_total))
if (( remain_total < 0 )); then
  remain_total=0
fi

if (( input_total > 0 )); then
  percent_total=$(awk -v a="${done_total}" -v b="${input_total}" 'BEGIN { printf "%.2f%%", (a / b) * 100 }')
else
  percent_total="0.00%"
fi

printf "%-28s %10s %10s %10s %10s\n" "dataset" "input" "done" "remain" "percent"
printf "%-28s %10s %10s %10s %10s\n" "----------------------------" "----------" "----------" "----------" "----------"
printf "%-28s %10d %10d %10d %10s\n" "metallic_qwen_total" "${input_total}" "${done_total}" "${remain_total}" "${percent_total}"
