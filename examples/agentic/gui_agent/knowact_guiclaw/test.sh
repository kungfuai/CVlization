#!/usr/bin/env bash
# Smoke test: exercise CVL CLI and dry-run inference.
#
# Two tiers:
#   Tier 1 (always): cvl info, cvl run build, wiring test (no VLM).
#   Tier 2 (if GUICLAW_BASE_URL is set): VLM semantic test requiring
#     exit_code==0, tool calls in trajectory, and new memory entries.
#
# Requires: cvl CLI installed (`pip install -e .` from repo root).
set -euo pipefail

EXAMPLE_NAME="agentic-knowact-guiclaw"
SCRATCH="$(mktemp -d -t cvl-guiclaw-test-XXXXXX)"
cleanup() { docker run --rm -v "${SCRATCH}:${SCRATCH}" alpine rm -rf "${SCRATCH}" 2>/dev/null || rm -rf "${SCRATCH}" 2>/dev/null || true; }
trap cleanup EXIT

pass=0
fail=0
skip=0

report() {
    local status="$1"; shift
    if [ "$status" = "PASS" ]; then
        echo "  [PASS] $*"
        pass=$((pass + 1))
    elif [ "$status" = "SKIP" ]; then
        echo "  [SKIP] $*"
        skip=$((skip + 1))
    else
        echo "  [FAIL] $*"
        fail=$((fail + 1))
    fi
}

echo "=== KnowAct-GUIClaw Smoke Test ==="
echo "scratch: ${SCRATCH}"
echo ""

# ================================================================
# TIER 1: Wiring checks (no VLM needed)
# ================================================================
echo "--- Tier 1: wiring checks ---"

# 1a. cvl info
if cvl info "${EXAMPLE_NAME}" >/dev/null 2>&1; then
    report PASS "cvl info ${EXAMPLE_NAME}"
else
    report FAIL "cvl info ${EXAMPLE_NAME}"
fi

# 1b. cvl run build
if cvl run "${EXAMPLE_NAME}" build >"${SCRATCH}/build.log" 2>&1; then
    report PASS "cvl run ${EXAMPLE_NAME} build"
else
    report FAIL "cvl run ${EXAMPLE_NAME} build (see ${SCRATCH}/build.log)"
fi

# 1c. Wiring test: dry-run without VLM (should fail with APIConnectionError)
mkdir -p "${SCRATCH}/wiring"
pushd "${SCRATCH}/wiring" >/dev/null

GUICLAW_BACKEND=dry-run \
GUICLAW_MAX_STEPS=2 \
GUICLAW_BASE_URL=http://127.0.0.1:1 \
cvl run "${EXAMPLE_NAME}" predict -- \
    --task "Wiring test" \
    --backend dry-run \
    >"${SCRATCH}/wiring.log" 2>&1 || true

popd >/dev/null

# Verify result.json exists and has expected structure
if python3 -c "
import json
with open('${SCRATCH}/wiring/knowact_guiclaw_output/result.json') as f:
    d = json.load(f)
assert d.get('task') == 'Wiring test', 'wrong task'
assert d.get('backend') == 'dry-run', 'wrong backend'
assert 'result' in d, 'missing result'
assert 'artifacts' in d, 'missing artifacts'
# Wiring test: expect non-zero exit (no VLM)
assert d['result'].get('exit_code', 0) != 0, 'expected non-zero exit without VLM'
" 2>/dev/null; then
    report PASS "wiring: result.json has correct structure and expected error"
else
    report FAIL "wiring: result.json missing or wrong structure"
fi

# Check trajectory was produced (even with error)
if [ -d "${SCRATCH}/wiring/knowact_guiclaw_output/latest_run" ]; then
    report PASS "wiring: trajectory directory created"
else
    report FAIL "wiring: trajectory directory not found"
fi

# ================================================================
# TIER 2: VLM semantic test (requires GUICLAW_BASE_URL)
# ================================================================
echo ""
echo "--- Tier 2: VLM semantic test ---"

if [ -z "${GUICLAW_BASE_URL:-}" ]; then
    report SKIP "VLM semantic test (set GUICLAW_BASE_URL to enable)"
else
    # Verify VLM endpoint is reachable
    if ! curl -sf "${GUICLAW_BASE_URL}/models" >/dev/null 2>&1; then
        report FAIL "VLM endpoint not reachable at ${GUICLAW_BASE_URL}"
    else
        mkdir -p "${SCRATCH}/vlm"
        pushd "${SCRATCH}/vlm" >/dev/null

        GUICLAW_BACKEND=dry-run \
        GUICLAW_MAX_STEPS=3 \
        cvl run "${EXAMPLE_NAME}" predict -- \
            --task "Open the Settings app" \
            --backend dry-run \
            >"${SCRATCH}/vlm.log" 2>&1 || true

        popd >/dev/null

        RESULT="${SCRATCH}/vlm/knowact_guiclaw_output/result.json"
        TRAJ="${SCRATCH}/vlm/knowact_guiclaw_output/latest_run/traj.json"

        # Check result has steps_taken > 0 (VLM actually responded)
        if python3 -c "
import json
with open('${RESULT}') as f:
    d = json.load(f)
parsed = d.get('result', {}).get('parsed', {})
steps = parsed.get('steps_taken', 0)
assert steps > 0, f'expected steps_taken > 0, got {steps}'
" 2>/dev/null; then
            report PASS "VLM: steps_taken > 0"
        else
            report FAIL "VLM: steps_taken == 0 (VLM did not respond)"
        fi

        # Check trajectory has tool_calls with computer_use
        if python3 -c "
import json
with open('${TRAJ}') as f:
    traj = json.load(f)
steps = traj.get('steps', [])
assert len(steps) > 0, 'no steps in trajectory'
tc = steps[0].get('model_output', {}).get('tool_calls', [])
assert len(tc) > 0, 'no tool_calls in step 1'
assert tc[0].get('name') == 'computer_use', f'expected computer_use, got {tc[0].get(\"name\")}'
args = tc[0].get('arguments', {})
assert 'action_type' in args, 'missing action_type'
assert 'intent' in args, 'missing intent'
" 2>/dev/null; then
            report PASS "VLM: trajectory has computer_use tool calls with action_type+intent"
        else
            report FAIL "VLM: trajectory missing expected tool calls"
        fi

        # Check memory was produced
        if [ -d "${SCRATCH}/vlm/knowact_guiclaw_output/memory_snapshot" ]; then
            report PASS "VLM: memory_snapshot directory created"
        else
            report FAIL "VLM: memory_snapshot directory not found"
        fi
    fi
fi

echo ""
echo "=== Summary ==="
total=$((pass + fail + skip))
echo "  ${pass} passed, ${fail} failed, ${skip} skipped (${total} total)"
echo "  scratch dir: ${SCRATCH} (auto-cleaned on exit)"

if [ "$fail" -gt 0 ]; then
    echo "  FAIL"
    exit 1
fi
echo "  PASS"
