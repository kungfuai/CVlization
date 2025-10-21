# Run this script at the project root directory.

# You may need to install huggingface_hub[cli] first:
# pip install "huggingface_hub[cli]"

ROOT_DIR=data/container_cache/huggingface/hub
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir $ROOT_DIR/Wan-AI/Wan2.1-T2V-1.3B
huggingface-cli download bytedance-research/Phantom --local-dir $ROOT_DIR/bytedance-research/Phantom
