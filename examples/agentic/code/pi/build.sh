#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${CVL_IMAGE:-pi}"
PI_BASE_IMG="${PI_BASE_IMG:-oh-my-pi/pi:dev}"
PI_REPO_URL="${PI_REPO_URL:-https://github.com/can1357/oh-my-pi.git}"
# Pinned commit of can1357/oh-my-pi at the time of last verification.
# Override via PI_REPO_REF=main (or any ref) to track upstream.
PI_REPO_REF="${PI_REPO_REF:-3b072a10b93a5e6ec8d024acab928b715edd6803}"
PI_SRC_DIR="${PI_SRC_DIR:-${HOME}/.cache/cvlization/oh-my-pi}"

# Step 1: ensure the upstream pi base image exists. Upstream publishes
# no prebuilt image (no CI tag, no Docker Hub) so we build from source.
if ! docker image inspect "${PI_BASE_IMG}" >/dev/null 2>&1; then
  echo "Base image ${PI_BASE_IMG} not found; cloning + building from ${PI_REPO_URL} @ ${PI_REPO_REF}"
  if [ ! -d "${PI_SRC_DIR}/.git" ]; then
    git clone "${PI_REPO_URL}" "${PI_SRC_DIR}"
  fi
  ( cd "${PI_SRC_DIR}"
    git fetch --depth 1 origin "${PI_REPO_REF}" 2>/dev/null || git fetch
    git checkout "${PI_REPO_REF}"
    docker build -t "${PI_BASE_IMG}" .
  )
fi

# Step 2: build our derived image (adds Python + TS language servers).
echo "Building ${IMG} on top of ${PI_BASE_IMG} ..."
docker build --build-arg "PI_BASE=${PI_BASE_IMG}" -t "${IMG}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
echo "Done: ${IMG}"
