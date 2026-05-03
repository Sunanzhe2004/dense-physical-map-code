#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_RUNNER="${SCRIPT_DIR}/run_albedo_interiorverse_mainaxis_4parts.sh"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/nohup_4dirs.out}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/albedo_4dirs.pid}"
ENV_NAME="${ENV_NAME:-albedo}"

if [[ ! -x /usr/bin/setsid ]]; then
  echo "setsid is unavailable; cannot use this detached launcher." >&2
  exit 1
fi

for idx in 0 1 2 3; do
  var_name="ARK_API_KEY_${idx}"
  if [[ -z "${!var_name:-}" ]]; then
    echo "${var_name} is not set." >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${LOG_FILE}")"

START_CMD=$(cat <<'EOF'
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"
exec bash "${INNER_RUNNER}"
EOF
)

setsid env \
  ENV_NAME="${ENV_NAME}" \
  INNER_RUNNER="${INNER_RUNNER}" \
  ARK_API_KEY_0="${ARK_API_KEY_0}" \
  ARK_API_KEY_1="${ARK_API_KEY_1}" \
  ARK_API_KEY_2="${ARK_API_KEY_2}" \
  ARK_API_KEY_3="${ARK_API_KEY_3}" \
  bash -lc "${START_CMD}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

DETACHED_PID=$!
echo "${DETACHED_PID}" > "${PID_FILE}"

echo "Started in background."
echo "PID: ${DETACHED_PID}"
echo "PID file: ${PID_FILE}"
echo "Log file: ${LOG_FILE}"
