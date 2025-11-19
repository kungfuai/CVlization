#!/bin/bash
# Download ComfyUI tokenizers from HuggingFace
# These tokenizers are used by wan_comfy and vace_comfy examples

set -e

echo "Downloading ComfyUI tokenizers from HuggingFace..."

# Function to download tokenizers using HuggingFace CLI or git
download_tokenizer() {
    local model_id=$1
    local target_dir=$2
    local files=$3  # comma-separated list of files to download

    mkdir -p "$target_dir"

    echo "Downloading tokenizer from $model_id to $target_dir"

    # Check if huggingface-cli is available
    if command -v huggingface-cli &> /dev/null; then
        for file in $(echo $files | tr ',' ' '); do
            huggingface-cli download "$model_id" "$file" --local-dir "$target_dir" --local-dir-use-symlinks False
        done
    else
        echo "Using wget to download files..."
        for file in $(echo $files | tr ',' ' '); do
            wget -P "$target_dir" "https://huggingface.co/$model_id/resolve/main/$file"
        done
    fi
}

# Download Llama 3 tokenizer for wan_comfy
if [ ! -f "wan_comfy/comfy/text_encoders/llama_tokenizer/tokenizer.json" ]; then
    echo "Downloading Llama 3 tokenizer for wan_comfy..."
    download_tokenizer \
        "meta-llama/Meta-Llama-3-8B" \
        "wan_comfy/comfy/text_encoders/llama_tokenizer" \
        "tokenizer.json,tokenizer_config.json,special_tokens_map.json"
else
    echo "Llama tokenizer already exists in wan_comfy"
fi

# Download Llama 3 tokenizer for vace_comfy
if [ ! -f "vace_comfy/comfy/text_encoders/llama_tokenizer/tokenizer.json" ]; then
    echo "Downloading Llama 3 tokenizer for vace_comfy..."
    download_tokenizer \
        "meta-llama/Meta-Llama-3-8B" \
        "vace_comfy/comfy/text_encoders/llama_tokenizer" \
        "tokenizer.json,tokenizer_config.json,special_tokens_map.json"
else
    echo "Llama tokenizer already exists in vace_comfy"
fi

# Download T5 tokenizer for wan_comfy
if [ ! -f "wan_comfy/comfy/text_encoders/t5_tokenizer/tokenizer.json" ]; then
    echo "Downloading T5 tokenizer for wan_comfy..."
    download_tokenizer \
        "google/umt5-xxl" \
        "wan_comfy/comfy/text_encoders/t5_tokenizer" \
        "tokenizer.json,tokenizer_config.json,special_tokens_map.json,spiece.model"
else
    echo "T5 tokenizer already exists in wan_comfy"
fi

# Download T5 tokenizer for vace_comfy
if [ ! -f "vace_comfy/comfy/text_encoders/t5_tokenizer/tokenizer.json" ]; then
    echo "Downloading T5 tokenizer for vace_comfy..."
    download_tokenizer \
        "google/umt5-xxl" \
        "vace_comfy/comfy/text_encoders/t5_tokenizer" \
        "tokenizer.json,tokenizer_config.json,special_tokens_map.json,spiece.model"
else
    echo "T5 tokenizer already exists in vace_comfy"
fi

# Download SD1 CLIP tokenizer for wan_comfy
if [ ! -f "wan_comfy/comfy/sd1_tokenizer/vocab.json" ]; then
    echo "Downloading SD1 CLIP tokenizer for wan_comfy..."
    download_tokenizer \
        "openai/clip-vit-large-patch14" \
        "wan_comfy/comfy/sd1_tokenizer" \
        "vocab.json,merges.txt,tokenizer_config.json"
else
    echo "SD1 tokenizer already exists in wan_comfy"
fi

# Download SD1 CLIP tokenizer for vace_comfy
if [ ! -f "vace_comfy/comfy/sd1_tokenizer/vocab.json" ]; then
    echo "Downloading SD1 CLIP tokenizer for vace_comfy..."
    download_tokenizer \
        "openai/clip-vit-large-patch14" \
        "vace_comfy/comfy/sd1_tokenizer" \
        "vocab.json,merges.txt,tokenizer_config.json"
else
    echo "SD1 tokenizer already exists in vace_comfy"
fi

echo ""
echo "✓ All tokenizers downloaded successfully!"
echo ""
echo "Note: You can also install huggingface-cli for faster downloads:"
echo "  pip install huggingface_hub"
