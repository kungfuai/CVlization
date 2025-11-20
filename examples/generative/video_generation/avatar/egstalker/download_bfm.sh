#!/bin/bash
set -euo pipefail

echo "Downloading BFM (Basel Face Model) files from HuggingFace..."
echo "================================================================"

# Create BFM directory
BFM_DIR="data_utils/face_tracking/3DMM"
mkdir -p "$BFM_DIR"

# Base URL for HuggingFace repository
BASE_URL="https://huggingface.co/wsj1995/sadTalker/resolve/main"

# Files to download (based on files found in working EGSTalker)
# Note: .npy files (3DMM_info.npy, exp_info.npy, etc.) are generated from .mat files
declare -A FILES=(
    ["01_MorphableModel.mat"]="$BASE_URL/BFM/01_MorphableModel.mat"           # 230MB - Main BFM model
    ["BFM09_model_info.mat"]="$BASE_URL/BFM/BFM09_model_info.mat"             # 122MB - BFM 2009 model
    ["BFM_exp_idx.mat"]="$BASE_URL/BFM/BFM_exp_idx.mat"                       # 90KB - Expression indices
    ["BFM_front_idx.mat"]="$BASE_URL/BFM/BFM_front_idx.mat"                   # 44KB - Front face indices
    ["facemodel_info.mat"]="$BASE_URL/BFM/facemodel_info.mat"                 # 722KB - Face model info
    ["select_vertex_id.mat"]="$BASE_URL/BFM/select_vertex_id.mat"             # 61KB - Selected vertices
    ["similarity_Lm3D_all.mat"]="$BASE_URL/BFM/similarity_Lm3D_all.mat"       # 1KB - Similarity transform
)

# Download each file
for filename in "${!FILES[@]}"; do
    url="${FILES[$filename]}"
    output_path="$BFM_DIR/$filename"

    if [ -f "$output_path" ]; then
        echo "✓ $filename already exists, skipping..."
        continue
    fi

    echo "Downloading $filename..."
    if wget -q --show-progress "$url" -O "$output_path"; then
        echo "✓ Downloaded $filename"
    else
        echo "✗ Failed to download $filename"
        rm -f "$output_path"  # Clean up partial download
    fi
done

echo ""
echo "BFM Download Summary:"
echo "================================================================"
ls -lh "$BFM_DIR"/*.mat 2>/dev/null || echo "No .mat files found"

echo ""
echo "Download complete! BFM files are in: $BFM_DIR"
echo "Total size: ~353MB (.mat files only)"
echo ""
echo "Note: Additional .npy files (3DMM_info.npy, exp_info.npy, keys_info.npy, topology_info.npy)"
echo "will be generated automatically when you first run BFM preprocessing."
echo ""
echo "These files enable BFM-based face tracking."
echo "To use BFM tracking: ./preprocess.sh video.mp4 --tracker bfm"
