# vLLM Serve + Predict (auto-tuned)

Dockerized vLLM preset with sensible defaults and optional auto-tuning based on your GPU. Supports both **text LLMs** and **vision-language models (VLMs)**. Includes:
- `build.sh`: builds the image (torch 2.9.1 + vLLM 0.15.1, OpenAI SDK 2.12.0).
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
- Base: `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`

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
```

## Client usage
```bash
# Chat local (no server) - text-only LLM
bash examples/generative/llm/vllm/predict.sh --prompt "Summarize PagedAttention."

# Vision-Language Model (VLM) - pass an image
MODEL_ID=Qwen/Qwen2-VL-2B-Instruct \
bash examples/generative/llm/vllm/predict.sh \
  --image /path/to/image.jpg \
  --prompt "Describe this image in detail."

# VLM with URL image
MODEL_ID=Qwen/Qwen2-VL-2B-Instruct \
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
Any VLM supported by vLLM 0.15.1 with stable transformers should work:
- `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `microsoft/Phi-3-vision-128k-instruct`
- `OpenGVLab/InternVL2-8B`
- See [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/) for full list

## Notes
- Mounts `~/.cache/huggingface` into the container for model pulls.
- If you disable Docker build caching, set `VLLM_IMAGE` to reuse a prebuilt image.
- CPU-only will work for small models but uses conservative defaults (`max_model_len=4096`, `dtype=float32`).

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
- ❌ `tencent/Hunyuan-A13B-Instruct-FP8` OOM on A10 (24GB); works on ≥80GB GPUs (see Blackwell section)
- ⚠️ `meta-llama/*Llama-3*/Llama-4*` (gated), `openai/gpt-oss-20b` and others not attempted (need bigger GPUs or special access)
