#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOLMO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STARVLA_ROOT="$(cd "${MOLMO_ROOT}/../../.." && pwd)"

ASSETS_DIR="${ASSETS_DIR:-/mnt/project/world_model/molmospaces_assets}"
CACHE_DIR="${CACHE_DIR:-/mnt/project/world_model/molmospaces_cache}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7777}"
NUM_WORKERS="${NUM_WORKERS:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MOLMO_ROOT}/eval_output/smartworld_leaderboard_$(date +%Y%m%d_%H%M%S)}"
START_SERVER="${START_SERVER:-0}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_ROOT}/smartworld_server.log}"

mkdir -p "${OUTPUT_ROOT}"

export MLSPACES_ASSETS_DIR="${ASSETS_DIR}"
export MLSPACES_CACHE_DIR="${CACHE_DIR}"
export SMARTWORLD_SERVER_HOST="${HOST}"
export SMARTWORLD_SERVER_PORT="${PORT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    echo "Stopping SmartWorld server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  python - "$HOST" "$PORT" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 1800
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        try:
            sock.connect((host, port))
        except OSError:
            time.sleep(5)
        else:
            print(f"SmartWorld server is reachable at {host}:{port}", flush=True)
            raise SystemExit(0)
raise SystemExit(f"Timed out waiting for SmartWorld server at {host}:{port}")
PY
}

cd "${STARVLA_ROOT}"
if [[ "${START_SERVER}" == "1" ]]; then
  echo "Starting SmartWorld server with scripts/server_launch.sh"
  echo "Server log: ${SERVER_LOG}"
  bash scripts/server_launch.sh >"${SERVER_LOG}" 2>&1 &
  SERVER_PID="$!"
  wait_for_server
else
  echo "Using existing SmartWorld server at ${HOST}:${PORT}"
fi

cd "${MOLMO_ROOT}"
"${MOLMO_ROOT}/.venv/bin/python" scripts/benchmarks/run_smartworld_eval.py \
  --all \
  --assets-dir "${ASSETS_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --num-workers "${NUM_WORKERS}" \
  --no-wandb

echo
echo "Leaderboard eval complete."
echo "Summary: ${OUTPUT_ROOT}/summary.csv"
