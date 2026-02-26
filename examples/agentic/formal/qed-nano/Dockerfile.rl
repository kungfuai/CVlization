# Note: vLLM 0.8.5.post1 (pinned by PipelineRL) does not have CUDA kernels compiled
# for SM120 (Blackwell). Run this on Ampere (SM80) or Hopper (SM90) GPUs such as
# A100 / H100. Blackwell support requires a PipelineRL update to newer vLLM.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    VLLM_LOGGING_LEVEL=ERROR

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

# 1. vLLM 0.8.5.post1 — pinned version required by PipelineRL.
#    Installs alongside the pre-built torch 2.6.0 in the base image.
RUN pip install --no-cache-dir vllm==0.8.5.post1

# 2. flash-attn — build against the installed torch + CUDA toolkit.
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# 3. Clone QED-Nano and install PipelineRL with the rest of its deps.
#    pyproject.toml pins: accelerate, Tapeagents, transformers, ring-flash-attn,
#    math-verify, orjson, redis, hydra-core.
RUN git clone --depth 1 https://github.com/CMU-AIRe/QED-Nano.git /opt/qed-nano && \
    cd /opt/qed-nano/training && \
    pip install --no-cache-dir -e .

ENV PIPELINERL_ROOT=/opt/qed-nano/training

CMD ["python", "-m", "pipelinerl.launch", "--help"]
