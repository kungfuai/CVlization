FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Unsloth controls all core dependency versions — install both together to avoid version mismatch
RUN pip install --no-cache-dir --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"

# NOTE: flash-attn 2.8.3 causes garbage output on Blackwell (SM120) with unsloth/Qwen3.5.
# PyTorch SDPA is used instead (attn_implementation="sdpa").
# RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

# Optional: causal-conv1d and flash-linear-attention for Qwen3.5 fast path
# NOTE: Only enable for training NEW models. Existing checkpoints trained without
# these libs will produce garbage if loaded with them (different computation graph).
# RUN MAX_JOBS=8 pip install --no-cache-dir causal-conv1d --no-build-isolation
# RUN MAX_JOBS=8 pip install --no-cache-dir flash-linear-attention --no-build-isolation

RUN pip install --no-cache-dir datasets==3.4.1 pyyaml==6.0.2 pillow==11.1.0 timm==1.0.12 wandb==0.19.9 qwen-vl-utils==0.0.10 addict matplotlib mergekit

RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')" && \
    python -c "import trl; print(f'TRL: {trl.__version__}')" && \
    python -c "import peft; print(f'PEFT: {peft.__version__}')" && \
    echo "All base imports successful"

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
CMD ["python", "train.py"]
