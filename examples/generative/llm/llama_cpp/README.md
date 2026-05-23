# llama.cpp Serve + Predict (OpenAI-compatible)

Dockerized [llama.cpp](https://github.com/ggml-org/llama.cpp) preset wrapping
`llama-server` with sensible defaults. Useful for **GGUF-only releases**
(e.g. `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`) that vLLM / SGLang / HF
transformers can't load directly, and for low-VRAM / CPU-friendly chat.

- `build.sh`: builds the image (`ghcr.io/ggml-org/llama.cpp:full-cuda` base + Python + OpenAI SDK 2.6+, pillow, huggingface_hub).
- `serve.sh`/`serve.py`: starts `llama-server` (OpenAI-compatible) with VRAM-driven defaults for `-c` and `-ngl`.
- `predict.sh`/`predict.py`: spins up `llama-server` in-container, hits it via the OpenAI Python client, tears it down. Supports VLMs via `--image`.

## Quick start
```bash
# From repo root
bash examples/generative/llm/llama_cpp/build.sh
bash examples/generative/llm/llama_cpp/predict.sh  # chat (default Qwen3-8B Q4_K_M)

# Serve (OpenAI-compatible):
bash examples/generative/llm/llama_cpp/serve.sh
# Then point any OpenAI client at http://localhost:8080/v1
```

Defaults:
- Model: `Qwen/Qwen3-8B-GGUF:Q4_K_M` (set `MODEL_ID` to change; or `MODEL_PATH` for a local `.gguf`)
- Port: `8080`, Host: `0.0.0.0`
- Base: `ghcr.io/ggml-org/llama.cpp:full-cuda` (override with `LLAMA_CPP_TAG`, e.g. `b9297` for a pinned build)

## Auto-tuning rules (override via env or flags)
- GPU detected → `-ngl 999` (full layer offload); CPU-only → `-ngl 0` (env `LLAMA_GPU_LAYERS`)
- Smallest GPU memory decides `-c`:
  - ≥ 80GB: 65536
  - ≥ 48GB: 32768
  - ≥ 24GB: 16384
  - ≥ 16GB: 8192
  - else (incl. CPU): 4096
- `--jinja` is on by default (uses the model's HF chat template; critical for correct prompts; disable with `LLAMA_USE_JINJA=0`)
- `-fa on` (flash attention) on by default (`LLAMA_FLASH_ATTN`)
- `--reasoning-format auto` (`LLAMA_REASONING_FORMAT`) — splits `<think>` traces for reasoning models into the response's `reasoning_content` field
- Extra args: `LLAMA_EXTRA_ARGS="-np 4 --mlock"` (example)

## Model-specific overrides

```bash
# Phi-4-mini (edge / CPU-only smoke test) — ~2.5GB Q4_K_M
MODEL_ID=unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M \
LLAMA_CONTEXT_LENGTH=4096 \
bash examples/generative/llm/llama_cpp/predict.sh

# Gemma-3 4B (vision + tool-calling)
MODEL_ID=unsloth/gemma-3-4b-it-GGUF:Q4_K_M \
bash examples/generative/llm/llama_cpp/predict.sh \
  --image /path/to/image.jpg \
  --prompt "Describe this image."

# GLM-4.7-Flash GGUF (MoE reasoning) — exercises --reasoning-format
MODEL_ID=unsloth/GLM-4.7-Flash-GGUF:Q4_K_M \
LLAMA_REASONING_FORMAT=auto \
bash examples/generative/llm/llama_cpp/predict.sh --max-tokens 1024

# Llama-3.3-70B at Q2_K — fits a single 24GB GPU as a "capability ceiling" demo
MODEL_ID=bartowski/Llama-3.3-70B-Instruct-GGUF:Q2_K \
LLAMA_CONTEXT_LENGTH=8192 \
bash examples/generative/llm/llama_cpp/predict.sh

# NVIDIA-Nemotron-3-Nano-4B-GGUF — GGUF-only release; the reason this example exists
MODEL_ID=nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF \
bash examples/generative/llm/llama_cpp/predict.sh

# Local .gguf file instead of -hf download
MODEL_PATH=/path/to/model.gguf \
bash examples/generative/llm/llama_cpp/predict.sh
```

## Reasoning models

llama-server has a built-in equivalent to vLLM/SGLang's `--reasoning-parser`:
`--reasoning-format <auto|none|deepseek>`. With `auto` (default in this preset),
the server splits `<think>` traces into a separate `reasoning_content` field
on the OpenAI response, leaving `content` clean. predict.py prints both.

## Notes
- llama.cpp's `-hf` downloads to `LLAMA_CACHE` (we point it at `~/.cache/huggingface/llama_cpp` so GGUFs sit alongside the rest of the HF cache).
- For CPU-only runs drop `--gpus all` from your docker invocation or set `LLAMA_GPU_LAYERS=0`.
- OpenAI base URL inside the container: `http://127.0.0.1:8080/v1`. API key is ignored unless you set `LLAMA_API_KEY`.
