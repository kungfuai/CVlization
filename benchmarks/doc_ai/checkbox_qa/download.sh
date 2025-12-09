#!/bin/bash
# Download CheckboxQA dataset to cache
#
# Usage:
#   ./download.sh              # Download to default cache
#   ./download.sh --no-pdfs    # Download annotations only (skip PDFs)
#   ./download.sh --force      # Force re-download
#
# Cache location: ~/.cache/cvlization/data/checkbox_qa/
# Override with: CHECKBOX_QA_CACHE_DIR=/custom/path ./download.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Python dependencies
if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing HuggingFace datasets library..."
    pip install -q datasets
fi

# Run the download
python3 -m checkbox_qa.dataset --download-only "$@"

echo ""
echo "Download complete!"
echo "Cache location: ${CHECKBOX_QA_CACHE_DIR:-~/.cache/cvlization/data/checkbox_qa/}"
