#!/usr/bin/env bash
# Smoke test: exercise CVL CLI and dry-run inference.
# Requires: cvl CLI installed (`pip install -e .` from repo root).
# Optional: GUICLAW_BASE_URL pointing to a vLLM server with a VLM for
#           full inference (otherwise dry-run produces APIConnectionError).
set -euo pipefail

EXAMPLE_NAME="agentic-knowact-guiclaw"
SCRATCH="$(mktemp -d -t cvl-guiclaw-test-XXXXXX)"
cleanup() { docker run --rm -v "${SCRATCH}:${SCRATCH}" alpine rm -rf "${SCRATCH}" 2>/dev/null || rm -rf "${SCRATCH}" 2>/dev/null || true; }
trap cleanup EXIT

pass=0
fail=0

report() {
    local status="$1"; shift
    if [ "$status" = "PASS" ]; then
        echo "  [PASS] $*"
        pass=$((pass + 1))
    else
        echo "  [FAIL] $*"
        fail=$((fail + 1))
    fi
}

echo "=== KnowAct-GUIClaw Smoke Test ==="
echo "scratch: ${SCRATCH}"
echo ""

# ---- 1. cvl info ----
echo "--- cvl info ---"
if cvl info "${EXAMPLE_NAME}" >/dev/null 2>&1; then
    report PASS "cvl info ${EXAMPLE_NAME}"
else
    report FAIL "cvl info ${EXAMPLE_NAME}"
fi

# ---- 2. cvl run build ----
echo "--- cvl run build ---"
if cvl run "${EXAMPLE_NAME}" build >"${SCRATCH}/build.log" 2>&1; then
    report PASS "cvl run ${EXAMPLE_NAME} build"
else
    report FAIL "cvl run ${EXAMPLE_NAME} build (see ${SCRATCH}/build.log)"
fi

# ---- 3. cvl run predict (dry-run) ----
echo "--- cvl run predict (dry-run from external cwd) ---"
mkdir -p "${SCRATCH}/workdir"
pushd "${SCRATCH}/workdir" >/dev/null

# Dry-run exits non-zero without a VLM endpoint; ignore exit code
GUICLAW_BACKEND=dry-run \
GUICLAW_MAX_STEPS=3 \
cvl run "${EXAMPLE_NAME}" predict -- \
    --task "Describe the current screen" \
    --backend dry-run \
    >"${SCRATCH}/predict.log" 2>&1 || true

popd >/dev/null

if [ -f "${SCRATCH}/workdir/knowact_guiclaw_output/result.json" ]; then
    report PASS "output result.json created on host"
else
    report FAIL "output result.json not found on host"
fi

if [ -f "${SCRATCH}/workdir/knowact_guiclaw_output/metrics.json" ]; then
    report PASS "output metrics.json created on host"
else
    report FAIL "output metrics.json not found on host"
fi

# Check trajectory directory
if [ -d "${SCRATCH}/workdir/knowact_guiclaw_output/latest_run" ]; then
    report PASS "trajectory latest_run/ directory created"
else
    report FAIL "trajectory latest_run/ directory not found"
fi

# Check memory snapshot
if [ -d "${SCRATCH}/workdir/knowact_guiclaw_output/memory_snapshot" ]; then
    report PASS "memory_snapshot/ directory created"
else
    report FAIL "memory_snapshot/ directory not found"
fi

# Check result.json has expected fields (task, backend, result)
if python3 -c "
import json, sys
with open('${SCRATCH}/workdir/knowact_guiclaw_output/result.json') as f:
    d = json.load(f)
assert 'task' in d, 'missing task'
assert 'backend' in d, 'missing backend'
assert 'result' in d, 'missing result'
assert 'artifacts' in d, 'missing artifacts'
" 2>/dev/null; then
    report PASS "result.json has expected fields (task, backend, result, artifacts)"
else
    report FAIL "result.json missing expected fields"
fi

echo ""
echo "=== Summary ==="
total=$((pass + fail))
echo "  ${pass}/${total} passed"
echo "  scratch dir: ${SCRATCH} (auto-cleaned on exit)"

if [ "$fail" -gt 0 ]; then
    echo "  FAIL"
    exit 1
fi
echo "  PASS"
