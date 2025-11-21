FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Install system dependencies including build tools for bitsandbytes
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install unsloth - let it control all dependency versions
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install vllm and stable packages
RUN pip install --no-cache-dir vllm datasets==3.4.1 pyyaml==6.0.2 pillow==11.1.0

# Verify critical imports work (unsloth requires GPU so tested at runtime)
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')" && \
    python -c "import trl; print(f'TRL: {trl.__version__}')" && \
    python -c "import peft; print(f'PEFT: {peft.__version__}')" && \
    python -c "import vllm; print(f'vLLM: {vllm.__version__}')" && \
    echo "✓ All base imports successful (unsloth will be tested at runtime)"

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV UNSLOTH_VLLM_STANDBY=1

# Default command
CMD ["python", "train.py"]
