#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   WAN2GP_SRC=/path/to/Wan2GP ./sync_vendor.sh
WAN2GP_SRC="${WAN2GP_SRC:-}"
if [[ -z "$WAN2GP_SRC" ]]; then
  echo "WAN2GP_SRC is not set. Example: WAN2GP_SRC=/tmp/Wan2GP ./sync_vendor.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${SCRIPT_DIR}/vendor/wan2gp"
PATCHES_DIR="${SCRIPT_DIR}/patches"

mkdir -p "$VENDOR_DIR"

rsync -a --delete \
  --exclude='**/__pycache__' \
  --include='models/ltx2/***' \
  --include='models/wan/***' \
  --include='models/longcat/***' \
  --include='models/qwen/convert_diffusers_qwen_vae.py' \
  --include='models/qwen/__init__.py' \
  --include='shared/***' \
  --include='defaults/ltx2_19B.json' \
  --include='defaults/ltx2_distilled.json' \
  --include='defaults/longcat_avatar.json' \
  --include='defaults/longcat_video.json' \
  --include='defaults/t2v.json' \
  --include='defaults/t2v_1.3B.json' \
  --include='defaults/i2v.json' \
  --exclude='*' \
  "${WAN2GP_SRC}/" \
  "$VENDOR_DIR/"

echo "Synced Wan2GP vendor files into ${VENDOR_DIR}"

# Apply patches
if [[ -d "$PATCHES_DIR" ]]; then
  echo "Applying patches..."
  for patchfile in "$PATCHES_DIR"/*.patch; do
    if [[ -f "$patchfile" ]]; then
      echo "  Applying $(basename "$patchfile")..."
      patch -d "$VENDOR_DIR" -p1 < "$patchfile"
    fi
  done
  echo "Patches applied successfully."
fi
