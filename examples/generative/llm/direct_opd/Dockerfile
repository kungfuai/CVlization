# Direct-OPD: Weak-to-Strong On-Policy Distillation
# Based on: https://github.com/BytedTsinghua-SIA/Direct-OPD
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV TORCH_HOME=/root/.cache/torch \
    HF_HOME=/root/.cache/huggingface \
    HF_DATASETS_CACHE=/root/.cache/huggingface/datasets \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
