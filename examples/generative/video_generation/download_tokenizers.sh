#!/bin/bash
# Download ComfyUI tokenizers from HuggingFace
# These tokenizers are used by wan_comfy and vace_comfy examples

set -e

echo "Downloading ComfyUI tokenizers from HuggingFace..."

# Check for HF_TOKEN for gated models
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  WARNING: HF_TOKEN not set. Some tokenizers (Llama) require authentication."
    echo "   To download Llama tokenizers, set HF_TOKEN environment variable:"
    echo "   export HF_TOKEN=your_token_here"
    echo ""
    echo "   You can get a token from: https://huggingface.co/settings/tokens"
    echo ""
fi

# Function to download tokenizers using HuggingFace CLI or wget
download_tokenizer() {
    local model_id=$1
    local target_dir=$2
    local files=$3  # comma-separated list of files to download
    local require_token=${4:-false}

    mkdir -p "$target_dir"

    echo "Downloading tokenizer from $model_id to $target_dir"

    # Check if huggingface-cli is available
    if command -v huggingface-cli &> /dev/null; then
        for file in $(echo $files | tr ',' ' '); do
            if [ "$require_token" = "true" ] && [ -n "$HF_TOKEN" ]; then
                huggingface-cli download "$model_id" "$file" --token "$HF_TOKEN" --local-dir "$target_dir" --local-dir-use-symlinks False
            else
                huggingface-cli download "$model_id" "$file" --local-dir "$target_dir" --local-dir-use-symlinks False
            fi
        done
    else
        echo "Using wget to download files..."
        for file in $(echo $files | tr ',' ' '); do
            if [ "$require_token" = "true" ] && [ -n "$HF_TOKEN" ]; then
                wget --header="Authorization: Bearer $HF_TOKEN" -P "$target_dir" "https://huggingface.co/$model_id/resolve/main/$file"
            else
                wget -P "$target_dir" "https://huggingface.co/$model_id/resolve/main/$file"
            fi
        done
    fi
}

# Download Llama 3 tokenizer for wan_comfy
if [ ! -f "wan_comfy/comfy/text_encoders/llama_tokenizer/tokenizer.json" ]; then
    echo "Downloading Llama 3 tokenizer for wan_comfy..."
    if download_tokenizer \
        "meta-llama/Meta-Llama-3-8B" \
        "wan_comfy/comfy/text_encoders/llama_tokenizer" \
        "tokenizer.json,tokenizer_config.json,special_tokens_map.json" \
        "true"; then
        echo "✓ Llama tokenizer downloaded for wan_comfy"
    else
        echo "⚠️  Failed to download Llama tokenizer. Skipping..."
        echo "   The tokenizers are still in the git repo, so this should not break anything."
    fi
else
    echo "Llama tokenizer already exists in wan_comfy"
fi

# Download Llama 3 tokenizer for vace_comfy
if [ ! -f "vace_comfy/comfy/text_encoders/llama_tokenizer/tokenizer.json" ]; then
    echo "Downloading Llama 3 tokenizer for vace_comfy..."
    if download_tokenizer \
        "meta-llama/Meta-Llama-3-8B" \
        "vace_comfy/comfy/text_encoders/llama_tokenizer" \
        "tokenizer.json,tokenizer_config.json,special_tokens_map.json" \
        "true"; then
        echo "✓ Llama tokenizer downloaded for vace_comfy"
    else
        echo "⚠️  Failed to download Llama tokenizer. Skipping..."
        echo "   The tokenizers are still in the git repo, so this should not break anything."
    fi
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
