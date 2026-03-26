FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Unsloth controls all core dependency versions
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

RUN pip install --no-cache-dir datasets==3.4.1 pyyaml==6.0.2 pillow==11.1.0 timm==1.0.12 wandb==0.19.9 qwen-vl-utils==0.0.10 addict matplotlib

RUN python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')" && \
    python -c "import trl; print(f'TRL: {trl.__version__}')" && \
    python -c "import peft; print(f'PEFT: {peft.__version__}')" && \
    echo "All base imports successful"

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
CMD ["python", "train.py"]
