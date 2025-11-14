#!/bin/bash
# Run FastAvatar with full guidance (test-time optimization)
# This demonstrates high-quality novel view synthesis using multi-view supervision

set -e

# Default parameters
SAMPLE_ID=${1:-422}
MAX_EPOCHS=${2:-101}  # Quick test: 101 epochs, Full quality: 401-801
DATA_ROOT="data"
CACHE_DIR="$HOME/.cache/fastavatar"
SAVE_PATH="results/sample_${SAMPLE_ID}_full_guidance"

echo "========================================="
echo "FastAvatar - Full Guidance Mode"
echo "========================================="
echo "Sample ID: $SAMPLE_ID"
echo "Max epochs: $MAX_EPOCHS"
echo "Data root: $DATA_ROOT"
echo "Save path: $SAVE_PATH"
echo "========================================="

# Check if dataset exists, download if needed
if [ ! -d "${DATA_ROOT}/${SAMPLE_ID}" ] && [ ! -L "${DATA_ROOT}/${SAMPLE_ID}" ]; then
    echo ""
    echo "Dataset '${SAMPLE_ID}' not found in ${DATA_ROOT}/"
    echo "Attempting to download and cache dataset..."
    echo ""

    # Download dataset using the download script
    python scripts/download_data.py ${SAMPLE_ID} --cache-dir ${CACHE_DIR}/datasets || {
        echo ""
        echo "ERROR: Failed to download dataset '${SAMPLE_ID}'"
        echo "Please manually download from:"
        echo "  https://github.com/hliang2/FastAvatar/tree/main/data/${SAMPLE_ID}"
        echo "Extract to: ${DATA_ROOT}/${SAMPLE_ID}/"
        exit 1
    }

    # Create symlink from cache to data directory
    DATASET_CACHE="${CACHE_DIR}/datasets/${SAMPLE_ID}"
    if [ -d "${DATASET_CACHE}" ]; then
        mkdir -p "${DATA_ROOT}"
        echo "Creating symlink: ${DATA_ROOT}/${SAMPLE_ID} -> ${DATASET_CACHE}"
        ln -sf "$(cd "${DATASET_CACHE}" && pwd)" "${DATA_ROOT}/${SAMPLE_ID}"
    fi
    echo ""
fi

echo "Using dataset: ${DATA_ROOT}/${SAMPLE_ID}"
echo ""

# Run inside Docker container
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v ${CACHE_DIR}:${CACHE_DIR} \
  fastavatar:latest \
  python scripts/inference_feedforward_full_guidance.py \
    --data_root ${DATA_ROOT} \
    --sample_id ${SAMPLE_ID} \
    --max_epochs ${MAX_EPOCHS} \
    --save_path ${SAVE_PATH} \
    --encoder_load_path ${CACHE_DIR}/pretrained_weights/encoder_neutral_flame.pth \
    --decoder_load_path ${CACHE_DIR}/pretrained_weights/decoder_neutral_flame.pth \
    --ply_file_path ${CACHE_DIR}/pretrained_weights/averaged_model.ply \
    --mlp_lr 1e-4 \
    --w_lr 1e-4 \
    --l1_weight 0.6 \
    --ssim_weight 0.3 \
    --lpips_weight 0.1

echo ""
echo "========================================="
echo "Full guidance optimization complete!"
echo "Results saved to: $SAVE_PATH"
echo "========================================="
echo ""
echo "Compare with feedforward-only results:"
echo "  Feedforward: results/sample_422_novel_views_corrected/"
echo "  Full guidance: ${SAVE_PATH}/images/"
echo ""
