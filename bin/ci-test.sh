#!/usr/bin/env bash
#
# Run unit tests.
#
# -s                Show all output, do not capture
# -v                Verbose
# -q                Less verbose
# -x                Stop after first failure
# -l                Show local variables in tracebacks
# --lf              Only run tests that failed last run (or all if none failed)
# -k "expression"   Only run tests that match expession
# -r chars          Show extra test summary info as specified by chars:
#                   (f)failed,
#                   (E)error
#                   (s)skipped
#                   (x)failed
#                   (X)passed
#                   (w)pytest-warnings
#                   (p)passed
#                   (P)passed with output
#                   (a)all except (p) and (P)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${CVL_DOCKER_IMAGE:-cvlization-test}"

# Build the CI image only if it doesn't already exist (e.g., prebuilt in CI)
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    docker build -t "${IMAGE_NAME}" -f "${REPO_ROOT}/Dockerfile" "${REPO_ROOT}"
fi

# Execute pytest suite inside the container
docker run --rm \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    python -m pytest -p no:warnings tests/ "$@"
