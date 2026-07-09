#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: build.sh [--runtime-profile PROFILE] [-- DOCKER_BUILD_ARGS...]

Build a Docker image for this example.

Runtime profiles:
  default          Baseline Transformers/TRL image (aliases: base, transformers)
  modern           Newer Transformers image with prebuilt FlashAttention
                   (alias: transformers-modern)
  qwen35           Qwen3.5 Transformers image (alias: transformers-qwen35)
  unsloth-qwen35   Unsloth Qwen3.5 image
  unsloth-latest   General Unsloth image for Llama/Qwen/Ministral-like models
                   (alias: unsloth)
  nemotron         Unsloth Nemotron image with Mamba dependencies
                   (alias: unsloth-nemotron)
  gemma4           Gemma 4 Transformers image
  unsloth-gemma4   Unsloth Gemma 4 image

Set CVL_IMAGE to override the output image tag.
EOF
}

PROFILE="default"
DOCKER_BUILD_ARGS=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --runtime-profile)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --runtime-profile" >&2
                exit 2
            fi
            PROFILE="$2"
            shift 2
            ;;
        --runtime-profile=*)
            PROFILE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            DOCKER_BUILD_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$PROFILE" in
    default|base|transformers)
        DOCKERFILE="Dockerfile"
        DEFAULT_IMAGE="doc_extraction_sft"
        ;;
    modern|transformers-modern)
        DOCKERFILE="Dockerfile.modern"
        DEFAULT_IMAGE="doc_extraction_sft_modern"
        ;;
    qwen35|transformers-qwen35)
        DOCKERFILE="Dockerfile.qwen35"
        DEFAULT_IMAGE="doc_extraction_sft_qwen35"
        ;;
    unsloth-qwen35)
        DOCKERFILE="Dockerfile.unsloth_qwen35"
        DEFAULT_IMAGE="doc_extraction_sft_unsloth_qwen35"
        ;;
    unsloth-latest|unsloth)
        DOCKERFILE="Dockerfile.unsloth_latest"
        DEFAULT_IMAGE="doc_extraction_sft_unsloth_latest"
        ;;
    nemotron|unsloth-nemotron)
        DOCKERFILE="Dockerfile.unsloth_nemotron"
        DEFAULT_IMAGE="doc_extraction_sft_unsloth_nemotron"
        ;;
    gemma4)
        DOCKERFILE="Dockerfile.gemma4"
        DEFAULT_IMAGE="doc_extraction_sft_gemma4"
        ;;
    unsloth-gemma4)
        DOCKERFILE="Dockerfile.unsloth_gemma4"
        DEFAULT_IMAGE="doc_extraction_sft_unsloth_gemma4"
        ;;
    *)
        echo "Unknown runtime profile: $PROFILE" >&2
        usage >&2
        exit 2
        ;;
esac

IMAGE="${CVL_IMAGE:-$DEFAULT_IMAGE}"
echo "Building runtime profile '$PROFILE' as image '$IMAGE' using $DOCKERFILE"
docker build -f "$SCRIPT_DIR/$DOCKERFILE" -t "$IMAGE" "${DOCKER_BUILD_ARGS[@]}" "$SCRIPT_DIR"
