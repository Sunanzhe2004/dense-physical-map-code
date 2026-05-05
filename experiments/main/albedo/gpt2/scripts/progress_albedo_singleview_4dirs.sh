#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_RUNNER="${SCRIPT_DIR}/../../gpt/scripts/progress_albedo_singleview_4dirs.sh"

export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/albedo_gpt2}"

exec bash "${INNER_RUNNER}"
