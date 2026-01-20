# SGLang Serve + Predict (OpenAI-compatible)

Dockerized SGLang preset with sensible defaults. Supports both **text LLMs** and **vision-language models (VLMs)**. Includes:
- `build.sh`: builds the image (torch 2.9.1 + sglang 0.5.6.post2, OpenAI SDK 2.6.1).
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

## Notes
- Mounts `~/.cache/huggingface` into the container for model pulls. Set `HF_TOKEN` if needed.
- OpenAI client base URL is `http://127.0.0.1:${PORT}/v1`; API key is ignored but required by the SDK (defaults to `sk-noauth`).
- CPU-only will work for tiny models but uses conservative defaults (`context_length=4096`, `dtype=float32`).

## Verification (A10, Dec 2025)
- Build + predict verified via `cvl run sglang predict` (CUDA devel base with nvcc) for:
  - `allenai/Olmo-3-7B-Instruct` (bf16, ctx=4096, mem_fraction_static=0.9) ~14GB VRAM.
  - `microsoft/Phi-4-mini-instruct` (bf16, ctx=4096, mem_fraction_static=0.9) ~7–8GB VRAM.
  - `google/gemma-3-1b-it` (bf16, ctx=4096, mem_fraction_static=0.9) ~1–2GB VRAM.
  - `LiquidAI/LFM2-1.2B` (bf16, ctx=4096, mem_fraction_static=0.9) loads via Transformers backend (slow tokenizer warning), works.
  - `allenai/Olmo-3-7B-Think` (bf16, ctx=4096, mem_fraction_static=0.9) works.
  - `allenai/OLMo-2-1124-7B-Instruct-preview` (bf16, ctx=2048, mem_fraction_static=0.88) works.
  - `internlm/internlm3-8b-instruct` (bf16, ctx=2048, mem_fraction_static=0.88) works.
  - `Qwen/Qwen3-8B` (bf16, ctx=2048, mem_fraction_static=0.88) works.
  - `tencent/Hunyuan-A13B-Instruct-FP8` fails on A10 (OOM during MoE init) even with ctx=1024 and mem_fraction_static=0.85.
    - For larger 7B/8B models, reducing context (e.g., 2048) and mem_fraction_static (e.g., 0.88) helps fit on A10.
