FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxcb1 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pin transformers 5.5.0 for stable behaviour across the example's verified
# models. vLLM 0.21.0's deps want 5.9.0, but 5.5.0 has been the verified
# baseline since vLLM 0.19.0 (covers OLMo-3, LFM2, DeepSeekVL2, Pixtral, etc.).
# Install after vLLM so the explicit version overrides vLLM's transitive pull.
RUN pip install --no-cache-dir transformers==5.5.0

# Patch FlashInfer 0.6.8's misleading check_cuda_arch on SM120 (consumer
# Blackwell): TARGET_CUDA_ARCHS comes back empty, the eligibility loop
# never runs, and it raises "FlashInfer requires GPUs with sm75 or higher"
# despite SM12.0 being well above that. Force the check to no-op; vLLM 0.21
# imports FlashInfer eagerly so we can't simply uninstall it.
RUN python - <<'EOF'
import pathlib, re
p = pathlib.Path("/opt/conda/lib/python3.11/site-packages/flashinfer/jit/core.py")
text = p.read_text()
needle = "def check_cuda_arch():"
if needle in text:
    new_text = re.sub(
        r"def check_cuda_arch\(\):\n( +)[^\n]+\n(?:\1[^\n]*\n)+",
        f'def check_cuda_arch():\n    return  # CVL patch: SM120 false-positive\n',
        text,
        count=1,
    )
    p.write_text(new_text)
    print("FlashInfer check_cuda_arch patched to no-op")
else:
    print("FlashInfer check_cuda_arch not found -- assuming upstream fix")
EOF

# vLLM 0.21's MoE backend oracle prefers FLASHINFER_TRTLLM > FLASHINFER_CUTLASS
# > TRITON for CUDA. FlashInfer's BF16 kernels fail on SM120 (consumer Blackwell)
# at the "No supported CUDA architectures found for major versions [12]" check.
# Reorder the priority list so TRITON is tried first for CUDA. FlashInfer stays
# in the list as a fallback for cases where TRITON's is_supported_config returns
# False (rare; would also fail later on SM120, but matches upstream behavior).
RUN python - <<'EOF'
import pathlib
p = pathlib.Path("/opt/conda/lib/python3.11/site-packages/vllm/model_executor/layers/fused_moe/oracle/unquantized.py")
text = p.read_text()
old = (
    "    elif current_platform.is_cuda():\n"
    "        _AVAILABLE_BACKENDS = [\n"
    "            UnquantizedMoeBackend.FLASHINFER_TRTLLM,\n"
    "            UnquantizedMoeBackend.FLASHINFER_CUTLASS,\n"
    "            UnquantizedMoeBackend.TRITON,\n"
    "            UnquantizedMoeBackend.BATCHED_TRITON,\n"
    "        ]\n"
)
new = (
    "    elif current_platform.is_cuda():\n"
    "        # CVL patch: TRITON first on consumer Blackwell (SM120) where\n"
    "        # FlashInfer BF16 kernels fail at CUDA arch detection.\n"
    "        _AVAILABLE_BACKENDS = [\n"
    "            UnquantizedMoeBackend.TRITON,\n"
    "            UnquantizedMoeBackend.BATCHED_TRITON,\n"
    "            UnquantizedMoeBackend.FLASHINFER_TRTLLM,\n"
    "            UnquantizedMoeBackend.FLASHINFER_CUTLASS,\n"
    "        ]\n"
)
if old in text:
    p.write_text(text.replace(old, new, 1))
    print("vLLM MoE backend priority patched (TRITON first)")
else:
    print("vLLM MoE backend priority target not found -- assuming upstream changed it")
EOF

COPY . .

CMD ["python3", "serve.py"]
