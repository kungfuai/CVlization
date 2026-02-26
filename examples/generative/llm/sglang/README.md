# SGLang Serve + Predict (OpenAI-compatible)

Dockerized SGLang preset with sensible defaults. Supports both **text LLMs** and **vision-language models (VLMs)**. Includes:
- `build.sh`: builds the image (torch 2.9.1 + sglang 0.5.9, OpenAI SDK 2.6.1).
- `serve.sh`/`serve.py`: starts an OpenAI-compatible HTTP server with heuristics for tensor-parallel size, context length, dtype, and static memory fraction (overridable).
- `predict.sh`/`predict.py`: spins up a local server inside the container, sends a chat completion via the OpenAI client, then tears the server down. Supports VLMs via `--image` flag.

## Quick start
```bash
# From repo root
bash examples/generative/llm/sglang/build.sh
bash examples/generative/llm/sglang/predict.sh  # chat (starts server+client), writes outputs/result.txt

# To serve (OpenAI-compatible)
bash examples/generative/llm/sglang/serve.sh
```

Defaults:
- Model: `allenai/Olmo-3-7B-Instruct` (set `MODEL_ID` to change)
- Port: `30000`, Host: `0.0.0.0`
- Base: `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` (nvcc present; numactl installed)

## Auto-tuning rules (override via env or flags)
- GPU count → `--tensor-parallel-size` (capped at number of GPUs; env `SGLANG_TP_SIZE`)
- Smallest GPU memory decides `--context-length`:
  - ≥80GB: 65536
  - ≥48GB: 32768
  - ≥24GB: 16384
  - ≥16GB: 8192
  - else: 4096
- Default dtype: `bfloat16` on GPU, `float32` on CPU (`SGLANG_DTYPE`)
- `--mem-fraction-static`: 0.92 when ≥32GB, else 0.9 (`SGLANG_MEM_FRACTION_STATIC`)
- Extra args: `SGLANG_EXTRA_ARGS="--enable-prefix-caching"` (example)

## Client usage
```bash
# Chat (local server started automatically) - text-only LLM
bash examples/generative/llm/sglang/predict.sh --prompt "Summarize SGLang routing."

# Override model / context / TP
MODEL_ID=microsoft/Phi-4-mini-instruct \
SGLANG_CONTEXT_LENGTH=4096 \
SGLANG_TP_SIZE=1 \
bash examples/generative/llm/sglang/predict.sh --max-tokens 64

# Vision-Language Model (VLM) - pass an image
MODEL_ID=Qwen/Qwen2-VL-2B-Instruct \
SGLANG_CONTEXT_LENGTH=4096 \
bash examples/generative/llm/sglang/predict.sh \
  --image /path/to/image.jpg \
  --prompt "Describe this image in detail."

# VLM with URL image
MODEL_ID=Qwen/Qwen2-VL-2B-Instruct \
bash examples/generative/llm/sglang/predict.sh \
  --image "https://example.com/image.jpg" \
  --prompt "What text is in this image?"
```

## Supported VLMs
Any VLM supported by SGLang should work:
- `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `microsoft/Phi-3-vision-128k-instruct`
- See [SGLang supported models](https://sgl-project.github.io/references/supported_models.html) for full list

## Model-specific overrides

Some models require non-default settings:

```bash
# LiquidAI/LFM2-1.2B — Mamba model; triton backend is rejected on Blackwell (SM120), use torch_native
MODEL_ID=LiquidAI/LFM2-1.2B \
SGLANG_CONTEXT_LENGTH=4096 SGLANG_MEM_FRACTION_STATIC=0.88 \
SGLANG_EXTRA_ARGS="--attention-backend torch_native" \
bash examples/generative/llm/sglang/predict.sh

# allenai/OLMo-2-1124-7B-Instruct-preview — needs lower context to avoid OOM
MODEL_ID=allenai/OLMo-2-1124-7B-Instruct-preview \
SGLANG_CONTEXT_LENGTH=2048 SGLANG_MEM_FRACTION_STATIC=0.88 \
bash examples/generative/llm/sglang/predict.sh

# internlm/internlm3-8b-instruct, Qwen/Qwen3-8B — 8B models; use lower context on ≤24GB GPUs
MODEL_ID=Qwen/Qwen3-8B \
SGLANG_CONTEXT_LENGTH=2048 SGLANG_MEM_FRACTION_STATIC=0.88 \
bash examples/generative/llm/sglang/predict.sh
```

## Notes
- Mounts `~/.cache/huggingface` into the container for model pulls. Set `HF_TOKEN` if needed.
- OpenAI client base URL is `http://127.0.0.1:${PORT}/v1`; API key is ignored but required by the SDK (defaults to `sk-noauth`).
- CPU-only will work for tiny models but uses conservative defaults (`context_length=4096`, `dtype=float32`).

## Verification (RTX PRO 6000 / Blackwell SM120, Feb 2026)
sglang 0.5.9, PyTorch 2.9.1+CUDA 12.8, 2× RTX PRO 6000 Blackwell Max-Q (95GB VRAM each), SM120.
- ✅ `allenai/Olmo-3-7B-Instruct` (bf16, ctx=4096, mem_frac=0.88)
- ✅ `microsoft/Phi-4-mini-instruct` (bf16, ctx=4096, mem_frac=0.88) ~5.4GB model + ~76GB KV cache
- ✅ `allenai/Olmo-3-7B-Think` (bf16, ctx=4096, mem_frac=0.88)
- ✅ `LiquidAI/LFM2-1.2B` (bf16, ctx=4096, mem_frac=0.88) — auto-selects `torch_native` backend (Mamba/SSM model; triton rejected by SGLang)
- ✅ `allenai/OLMo-2-1124-7B-Instruct-preview` (bf16, ctx=2048, mem_frac=0.88)
- ✅ `internlm/internlm3-8b-instruct` (bf16, ctx=2048, mem_frac=0.88)
- ✅ `Qwen/Qwen3-8B` (bf16, ctx=2048, mem_frac=0.88) ~15.3GB model
- ✅ `google/gemma-3-1b-it` (bf16, ctx=4096, mem_frac=0.9) smoke-tested ~2GB VRAM
- ✅ `meta-llama/Llama-3.1-8B-Instruct` (bf16, ctx=4096, mem_frac=0.9) — requires `HF_TOKEN` (gated)
- ❌ `openai/gpt-oss-20b` (MXFP4): FlashInfer MXFP4 MoE kernel crashes (`assert K % 4 == 0`); bf16 variant `lmsys/gpt-oss-20b-bf16` also fails — Triton MoE kernel uses `.tile::gather4 .shared::cluster` TMA instruction not available on SM120a (consumer Blackwell); use vLLM instead
- ❌ `tencent/Hunyuan-A13B-Instruct-FP8` two SM120 blockers: (1) `v_head_dim=null` in config.json crashes KV-cache profiler — patched in Dockerfile; (2) FP8 MoE Triton kernel requires 147 KB shared memory, SM120 limit is 101 KB (H100 has 228 KB) — use vLLM instead

## Verification (A10, Dec 2025)
sglang 0.5.6.post2, PyTorch 2.9.1+CUDA 12.8, A10 (24GB VRAM), SM86.
- ✅ `allenai/Olmo-3-7B-Instruct` (bf16, ctx=4096, mem_frac=0.9) ~14GB VRAM
- ✅ `microsoft/Phi-4-mini-instruct` (bf16, ctx=4096, mem_frac=0.9) ~7–8GB VRAM
- ✅ `google/gemma-3-1b-it` (bf16, ctx=4096, mem_frac=0.9) ~1–2GB VRAM
- ✅ `LiquidAI/LFM2-1.2B` (bf16, ctx=4096, mem_frac=0.9) loads via Transformers backend (slow tokenizer warning)
- ✅ `allenai/Olmo-3-7B-Think` (bf16, ctx=4096, mem_frac=0.9)
- ✅ `allenai/OLMo-2-1124-7B-Instruct-preview` (bf16, ctx=2048, mem_frac=0.88)
- ✅ `internlm/internlm3-8b-instruct` (bf16, ctx=2048, mem_frac=0.88)
- ✅ `Qwen/Qwen3-8B` (bf16, ctx=2048, mem_frac=0.88)
- ❌ `tencent/Hunyuan-A13B-Instruct-FP8` OOM on A10 (24GB) — model is 75.4 GB, requires ≥80GB GPU
- ⚠️ `openai/gpt-oss-20b`, `meta-llama/Llama-3.1-8B-Instruct` not tested on A10
  - For larger 7B/8B models, reducing context (e.g., 2048) and mem_fraction_static (e.g., 0.88) helps fit on A10.
