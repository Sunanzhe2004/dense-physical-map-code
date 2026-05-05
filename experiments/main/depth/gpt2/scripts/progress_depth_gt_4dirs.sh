#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_RUNNER="${SCRIPT_DIR}/../../gpt/scripts/progress_depth_gt_4dirs.sh"

export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/depth_gpt2}"

exec bash "${INNER_RUNNER}"
