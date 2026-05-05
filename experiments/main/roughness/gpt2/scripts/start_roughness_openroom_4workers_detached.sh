#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_RUNNER="${SCRIPT_DIR}/../../gpt/scripts/start_roughness_openroom_4workers_detached.sh"

export BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/path/to/benchmark_outputs/roughness_gpt2}"
export IMAGE_MODEL="${IMAGE_MODEL:-gpt-image-2}"
export AZURE_GPT_IMAGE_15_API_KEY="${AZURE_GPT_IMAGE_15_API_KEY:-${AZURE_GPT_IMAGE_2_API_KEY:-}}"
export AZURE_GPT_IMAGE_15_ENDPOINT="${AZURE_GPT_IMAGE_15_ENDPOINT:-${AZURE_GPT_IMAGE_2_ENDPOINT:-}}"
export AZURE_GPT_IMAGE_15_API_VERSION="${AZURE_GPT_IMAGE_15_API_VERSION:-${AZURE_GPT_IMAGE_2_API_VERSION:-}}"

exec bash "${INNER_RUNNER}"
