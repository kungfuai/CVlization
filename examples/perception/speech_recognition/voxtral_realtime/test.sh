#!/usr/bin/env bash
set -euo pipefail
# Full cycle test: start server, wait for readiness, run streaming client, stop server.
# Ensures no server is left running after the test completes (success or failure).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${VOXTRAL_CONTAINER_NAME:-cvl-voxtral-realtime-server}"
PORT="${PORT:-8000}"
HOST="${VOXTRAL_HOST:-localhost}"

cleanup() {
  echo ""
  echo "=== Cleanup: stopping server ==="
  bash "${SCRIPT_DIR}/stop.sh"
}
trap cleanup EXIT

echo "=== Step 1/4: Build image ==="
bash "${SCRIPT_DIR}/build.sh"

echo ""
echo "=== Step 2/4: Start server (detached) ==="
VOXTRAL_DETACH=1 bash "${SCRIPT_DIR}/serve.sh"

echo ""
echo "=== Step 3/4: Wait for server readiness ==="
TIMEOUT=600
ELAPSED=0
echo "Waiting for http://${HOST}:${PORT}/v1/models (timeout: ${TIMEOUT}s) ..."
until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "ERROR: Server did not become ready within ${TIMEOUT}s"
    echo "Last logs:"
    docker logs --tail 30 "${CONTAINER_NAME}" 2>&1 || true
    exit 1
  fi
  sleep 5
  ELAPSED=$((ELAPSED + 5))
  if [ $((ELAPSED % 30)) -eq 0 ]; then
    echo "  ... still waiting (${ELAPSED}s elapsed)"
  fi
done
echo "Server ready after ${ELAPSED}s."

echo ""
echo "=== Step 4/4: Run streaming transcription ==="
# Run client directly on the host to avoid nested Docker complexity during test
python3 "${SCRIPT_DIR}/predict.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --output "voxtral_realtime_transcript.json" \
  --verbose \
  "$@"

echo ""
echo "=== Test PASSED ==="
echo "Transcript saved. Server will be stopped by cleanup trap."
