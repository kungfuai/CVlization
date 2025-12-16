# Miles GRPO Training for Qwen3-0.6B
# Uses pre-built Miles Docker image with all dependencies
# Note: Miles image includes PyTorch 2.5+ with CUDA support, SGLang, Megatron, Ray, etc.
# For torch 2.9+ compatibility, check Miles releases for updated base images.

FROM radixark/miles:latest

# Set cache directories for model downloads
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# Install additional dependencies for our wrapper
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Working directory
WORKDIR /workspace

# Default command
CMD ["python", "train.py"]
