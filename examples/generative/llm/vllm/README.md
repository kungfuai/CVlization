# vLLM Serve + Predict (auto-tuned)

Dockerized vLLM preset with sensible defaults and optional auto-tuning based on your GPU. Supports both **text LLMs** and **vision-language models (VLMs)**. Includes:
- `build.sh`: builds the image (vLLM 0.21.0 brings torch 2.11.0 + CUDA 13.0; we still pin transformers 5.5.0; OpenAI SDK 2.12.0).
- `serve.sh`/`serve.py`: starts an OpenAI-compatible server with heuristics for tensor-parallel size, max context, dtype, and GPU memory utilization (overridable).
- `predict.sh`/`predict.py`: runs a quick test. Default mode is **chat** (loads the model inside the container with vLLM, no server needed). Supports VLMs via `--image` flag. `embed` and `rerank` modes use transformers locally (not vLLM) for encoder models.

## Quick start
```bash
# From repo root
bash examples/generative/llm/vllm/build.sh
bash examples/generative/llm/vllm/predict.sh  # chat (local), writes outputs/result.txt

# To serve (OpenAI-compatible)
bash examples/generative/llm/vllm/serve.sh
```

Defaults:
- Model: `allenai/Olmo-3-7B-Instruct` (set `MODEL_ID` to change; served name mirrors the model unless `SERVED_MODEL_NAME` is set)
- Port: `8000`, Host: `0.0.0.0`
- Base: `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel`

## Auto-tuning rules (override any via env or flags)
- GPU count → `--tensor-parallel-size` (capped at number of GPUs; env `VLLM_TP_SIZE`)
- Smallest GPU memory decides `--max-model-len`:
  - ≥80GB: 131072
  - ≥48GB: 65536
  - ≥24GB: 32768
  - ≥16GB: 16384
  - else: 8192
- Default dtype: `bfloat16` on GPU, `float32` on CPU (`VLLM_DTYPE`)
- `--gpu-memory-utilization`: 0.95 when >=32GB, else 0.92 (`VLLM_GPU_MEMORY_UTILIZATION`)
- Extra args: `VLLM_EXTRA_ARGS="--enable-chunked-prefill --max-num-batched-tokens 8192"` (example)

## Common overrides
```bash
MODEL_ID=meta-llama/Llama-4-Scout-17B-16E \
VLLM_TP_SIZE=4 \
VLLM_MAX_MODEL_LEN=65536 \
VLLM_DTYPE=bfloat16 \
VLLM_EXTRA_ARGS="--trust-remote-code --gpu-memory-utilization 0.96" \
bash examples/generative/llm/vllm/serve.sh
```

## Model-specific overrides

Some models require non-default settings to fit in GPU memory:

```bash
# allenai/OLMo-2-1124-7B-Instruct-preview — needs lower context to avoid OOM on ≤24GB GPUs
MODEL_ID=allenai/OLMo-2-1124-7B-Instruct-preview \
VLLM_MAX_MODEL_LEN=1024 VLLM_GPU_MEMORY_UTILIZATION=0.88 \
bash examples/generative/llm/vllm/predict.sh

# 7B/8B models on ≤24GB GPUs — reduce context from auto-tuned default
MODEL_ID=Qwen/Qwen3-8B \
VLLM_MAX_MODEL_LEN=2048 VLLM_GPU_MEMORY_UTILIZATION=0.88 \
bash examples/generative/llm/vllm/predict.sh

# Qwen3.6 (GDN hybrid attention, VLM, reasoning) — needs ~51GB (27B) / ~66GB (35B-A3B)
# BF16 weights, so an 80GB+ GPU. VLLM_ENFORCE_EAGER=1 avoids GDN CUDA-graph issues;
# raise --max-tokens since the reasoning <think> trace counts against the budget.
MODEL_ID=Qwen/Qwen3.6-27B \
VLLM_MAX_MODEL_LEN=8192 VLLM_ENFORCE_EAGER=1 \
bash examples/generative/llm/vllm/predict.sh --max-tokens 1024
# MoE variant: MODEL_ID=Qwen/Qwen3.6-35B-A3B (~3B active params)

# GLM-4.7-Flash (Glm4MoeLiteForCausalLM, MoE ~3B active, reasoning) — ~56GB BF16
# weights. Natively supported by vLLM 0.19.0.
MODEL_ID=zai-org/GLM-4.7-Flash \
VLLM_MAX_MODEL_LEN=8192 VLLM_ENFORCE_EAGER=1 \
bash examples/generative/llm/vllm/predict.sh --max-tokens 1024

# NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 (NemotronHForCausalLM, MoE+Mamba2+Attn,
# ~3B active, reasoning) — ~59GB BF16 weights.
MODEL_ID=nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-BF16 \
VLLM_MAX_MODEL_LEN=8192 VLLM_ENFORCE_EAGER=1 \
bash examples/generative/llm/vllm/predict.sh --max-tokens 1024

# Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 (NemotronH_Nano_Omni_Reasoning_V3,
# omni-modal MoE+Mamba2+Attn, ~3B active, reasoning) — ~62GB BF16 weights. vLLM
# 0.21.0 registers the arch (maps to NemotronH_Nano_VL_V2 class). Requires the
# SM120 FlashInfer patches in the Dockerfile + gpu_utils.py (see below).
MODEL_ID=nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
VLLM_MAX_MODEL_LEN=8192 VLLM_ENFORCE_EAGER=1 \
bash examples/generative/llm/vllm/predict.sh --max-tokens 1024
```

## SM120 (consumer Blackwell) notes

vLLM 0.21's default MoE/sampler kernel choices route through FlashInfer, whose
CUDA-arch detection is broken on SM120 (RTX 5090 / RTX PRO 6000). The example
ships three workarounds applied automatically:

- `Dockerfile` patches `flashinfer.jit.core.check_cuda_arch` to a no-op.
- `Dockerfile` reorders the vLLM MoE backend priority list so TRITON is tried
  before FlashInfer on CUDA platforms.
- `gpu_utils.py` sets `VLLM_USE_FLASHINFER_SAMPLER=0` for SM120+, falling back
  to the PyTorch-native sampler.

Combined they let vLLM 0.21 run cleanly on SM120 for BF16 models.

**Known regression**: `mistralai/Ministral-3-8B-Instruct-2512` (fp8 VLM) worked on
vLLM 0.19.0 but fails on 0.21.0 because the FP8 GEMM path now routes through
FlashInfer, whose JIT compiles `compute_120f` kernels that need CUDA ≥ 12.9
nvcc — our base image ships CUDA 12.8 nvcc. Fixing requires a CUDA 13.0-devel
base image. Other FP8 models on different archs may hit the same issue.

## Reasoning models

Reasoning models (Qwen3 / Qwen3.5 / Qwen3.6, GLM-4.7-Flash, OLMo-3-Think, etc.)
emit a `<think>...</think>` trace before their answer. (Note: Qwen3.x strips the
`<think>`/`</think>` delimiters as special tokens, whereas GLM-4.7-Flash emits a
literal `</think>` tag in the text.)

- **Serving (`serve.sh`)** — pass a reasoning parser so the trace is returned in a
  separate `reasoning` field instead of being mixed into `content`:
  ```bash
  MODEL_ID=Qwen/Qwen3.6-27B \
  VLLM_EXTRA_ARGS="--reasoning-parser qwen3 --enforce-eager" \
  bash examples/generative/llm/vllm/serve.sh
  ```
  The parser name follows the model family — `qwen3` for the Qwen3.x line,
  `glm45` for GLM-4.x (incl. GLM-4.7-Flash). `--enforce-eager` guards against
  the GDN CUDA-graph issue on Qwen3.6.
- **Local predict (`predict.sh`)** — the offline `LLM` path has no reasoning
  parser, so the trace stays inline in the output text. Raise `--max-tokens` so
  the final answer isn't truncated by a long trace.

## Client usage
```bash
# Chat local (no server) - text-only LLM
bash examples/generative/llm/vllm/predict.sh --prompt "Summarize PagedAttention."

# Vision-Language Model (VLM) - pass a local image
MODEL_ID=Qwen/Qwen3.5-9B \
bash examples/generative/llm/vllm/predict.sh \
  --image /path/to/image.jpg \
  --prompt "Describe this image in detail."

# VLM with URL image
MODEL_ID=mistralai/Ministral-3-8B-Instruct-2512 \
bash examples/generative/llm/vllm/predict.sh \
  --image "https://example.com/image.jpg" \
  --prompt "What text is in this image?"

# Embeddings (encoder or decoder models) - runs locally with transformers
python examples/generative/llm/vllm/predict.py --mode embed \
  --model google/embeddinggemma-300m \
  --text-a "hello world" --normalize

# Rerank (cross-encoder) - runs locally with transformers
python examples/generative/llm/vllm/predict.py --mode rerank \
  --model mixedbread-ai/mxbai-rerank-base-v2 \
  --text-a "query text" \
  --doc "candidate 1" --doc "candidate 2" \
  --docs-file my_docs.txt  # optional, one doc per line
```

## Supported VLMs
Any VLM supported by vLLM 0.19.0 should work via `--image` flag:
- `google/gemma-4-e2b-it` (verified), `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`
- `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`
- `allenai/Molmo2-8B`, `allenai/Molmo2-4B`
- `openbmb/MiniCPM-o-4_5` (omni-modal: video+audio+speech)
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `microsoft/Phi-3-vision-128k-instruct`
- `OpenGVLab/InternVL2-8B`
- See [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/) for full list

## Notes
- Mounts `~/.cache/huggingface` into the container for model pulls.
- If you disable Docker build caching, set `VLLM_IMAGE` to reuse a prebuilt image.
- CPU-only will work for small models but uses conservative defaults (`max_model_len=4096`, `dtype=float32`).

## Verification (RTX PRO 6000 / Blackwell SM120, Apr 2026)
vLLM 0.19.0 + transformers 5.5.0, PyTorch 2.9.1+CUDA 12.8, RTX PRO 6000 Blackwell (98GB VRAM), SM120.
- ✅ `allenai/OLMo-3-7B-Instruct` (bf16, max_len=4096)
- ✅ `Qwen/Qwen3-4B` (bf16, think mode; use mem_util=0.80 if GPU partly occupied)
- ✅ `Qwen/Qwen3.5-9B` (bf16, VLM; think mode; text-only and image both verified)
- ✅ `google/gemma-4-e2b-it` (bf16, max_len=4096; use mem_util=0.80 if GPU partly occupied)
- ✅ `nvidia/NVIDIA-Nemotron-Nano-9B-v2` (bf16; Mamba-2+Attention hybrid, thinking mode)
- ✅ `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (bf16; MoE+Mamba-2+Attention hybrid, 3.5B active params)
- ✅ `mistralai/Ministral-3-8B-Instruct-2512` (fp8/Pixtral; text-only and image both verified; requires devel base image for FP8 JIT kernel on SM120)

## Verification (RTX PRO 6000 / Blackwell SM120, Feb 2026)
vLLM 0.15.1, PyTorch 2.9.1+CUDA 12.8, 2× RTX PRO 6000 Blackwell Max-Q (95GB VRAM each), SM120. FLASH_ATTN backend auto-selected.
- ✅ `allenai/Olmo-3-7B-Instruct` (bf16, max_len=2048, mem_util=0.88) ~13.6GB model
- ✅ `allenai/Olmo-3-7B-Think` (bf16, max_len=2048, mem_util=0.88) ~13.6GB model
- ✅ `microsoft/Phi-4-mini-instruct` (bf16, max_len=2048, mem_util=0.88)
- ✅ `LiquidAI/LFM2-1.2B` (bf16, max_len=2048, mem_util=0.88)
- ✅ `internlm/internlm3-8b-instruct` (bf16, max_len=2048, mem_util=0.88)
- ✅ `Qwen/Qwen3-8B` (bf16, max_len=2048, mem_util=0.88) ~15.3GB model
- ✅ `google/gemma-3-1b-it` (bf16, max_len=2048, mem_util=0.88) ~1.9GB model
- ✅ `allenai/OLMo-2-1124-7B-Instruct-preview` (bf16, max_len=1024, mem_util=0.88) ~13.6GB model
- ✅ embed: `google/embeddinggemma-300m` (~1024d), `sentence-transformers/all-roberta-large-v1` (~768d)
- ✅ rerank: `mixedbread-ai/mxbai-rerank-base-v2` (cross-encoder)
- ✅ `tencent/Hunyuan-A13B-Instruct-FP8` (bf16, max_len=4096, mem_util=0.88) — loads 75.4 GB, only ~0.82 GB KV cache remaining; use the lowest practical context length
- ✅ `openai/gpt-oss-20b` (MXFP4, max_len=4096, mem_util=0.95) — 13.72 GB loaded via Marlin kernel (no native FP4 on SM120, uses weight-only compression); 64.27 GB KV cache
- ✅ `meta-llama/Llama-3.1-8B-Instruct` (bf16, max_len=4096) — 14.99 GB; requires `HF_TOKEN` (gated)

## Verification (A10, Dec 2025)
vLLM 0.14.0, PyTorch 2.9.1+CUDA 12.8, A10 (24GB VRAM), SM86.
- ✅ `Qwen/Qwen2.5-1.5B-Instruct` (bf16, max_len=4096, mem_util=0.9)
- ✅ `allenai/Olmo-3-7B-Instruct` (bf16, max_len=2048, mem_util=0.88)
- ✅ `allenai/Olmo-3-7B-Think` (bf16, max_len=2048)
- ✅ `allenai/OLMo-2-1124-7B-Instruct-preview` (bf16, max_len=1024, mem_util=0.85)
- ✅ `microsoft/Phi-4-mini-instruct` (bf16, max_len=2048)
- ✅ `Qwen/Qwen3-8B` (bf16, max_len=2048) ~15GB VRAM
- ✅ `LiquidAI/LFM2-1.2B`
- ✅ `internlm/internlm3-8b-instruct` (bf16, max_len=2048) ~16GB VRAM
- ✅ `google/gemma-3-1b-it` (bf16, max_len=2048)
- ✅ embed: `google/embeddinggemma-300m`, `sentence-transformers/all-roberta-large-v1`
- ✅ rerank: `mixedbread-ai/mxbai-rerank-base-v2`
- ❌ `tencent/Hunyuan-A13B-Instruct-FP8` OOM on A10 (24GB) — model is 75.4 GB, requires ≥80GB GPU
