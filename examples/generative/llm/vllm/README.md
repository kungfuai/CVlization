# vLLM Serve + Predict (auto-tuned)

Dockerized vLLM preset with sensible defaults and optional auto-tuning based on your GPU. Supports both **text LLMs** and **vision-language models (VLMs)**. Includes:
- `build.sh`: builds the image (torch 2.9.1 + vLLM 0.14.0, OpenAI SDK 2.12.0).
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
Any VLM supported by vLLM 0.14.0 with stable transformers should work:
- `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `microsoft/Phi-3-vision-128k-instruct`
- `OpenGVLab/InternVL2-8B`
- See [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/) for full list

## Notes
- Mounts `~/.cache/huggingface` into the container for model pulls.
- If you disable Docker build caching, set `VLLM_IMAGE` to reuse a prebuilt image.
- CPU-only will work for small models but uses conservative defaults (`max_model_len=4096`, `dtype=float32`).

## Quick verification notes (A10, Dec 2025)
- ✅ `Qwen/Qwen2.5-1.5B-Instruct` (local vLLM, bf16, max_len=4096, gpu_mem_util=0.9, enforce_eager=1) works; output in `outputs/result.txt`.
- ⚠️ Llama 3.x / Llama 4 / GPT-OSS-20B / RNJ-1 not run here: Llama3/4 are gated; GPT-OSS-20B likely exceeds single A10 without aggressive quant/multi-GPU; RNJ-1 repo unknown—needs exact HF id.
- ✅ `allenai/Olmo-3-7B-Instruct` (bf16, max_len=2048, gpu_mem_util=0.88, enforce_eager=1) works.
- ✅ `allenai/Olmo-3-7B-Think` works (bf16, 2048 context).
- ✅ `allenai/OLMo-2-1124-7B-Instruct-preview` works (bf16, 1024 context, gpu_mem_util=0.85).
- ✅ `microsoft/Phi-4-mini-instruct` works (bf16, 2048 context).
- ✅ `Qwen/Qwen3-8B` works (bf16, 2048 context); uses ~15GB VRAM.
- ✅ `LiquidAI/LFM2-1.2B` works.
- ✅ `internlm/internlm3-8b-instruct` works (bf16, 2048 context); uses ~16GB VRAM.
- ❌ `tencent/Hunyuan-A13B-Instruct-FP8` OOM on A10 during MoE init even with max_len=1024, gpu_mem_util=0.85, enforce_eager=1.
- ✅ `google/gemma-3-1b-it` works (bf16, 2048 context).
- ✅ Embedding/rerank paths (tested via dedicated modes): `google/embeddinggemma-300m` (embed mode, ~1024d), `sentence-transformers/all-roberta-large-v1` (embed mode, ~768d), `mixedbread-ai/mxbai-rerank-base-v2` (rerank mode; cross-encoder query vs document score).
- Not attempted here (likely need bigger GPUs, special access, or different runtimes): `meta-llama/*Llama-3*/Llama-4*` (gated), `openai/gpt-oss-20b`, `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1`, `microsoft/Phi-4-multimodal-instruct`, `mistralai/Ministral-3-3B-Instruct-2512`, `SmolVLM2-2.2B-Instruct`, `openai/whisper-small`, `openai/whisper-large-v3-turbo`, `mistralai/Voxtral-Mini-3B-2507`, `ibm-granite/granite-speech-3.3-8b`, `baidu/ERNIE-4.5-0.3B-PT`, `RNJ-1` (HF id unknown).
