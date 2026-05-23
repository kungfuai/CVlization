# llama.cpp Diffusion LM (LLaDA / Dream)

Runs **diffusion language models** (LLaDA, Dream) via llama.cpp's dedicated
`llama-diffusion-cli` binary. Diffusion LMs don't fit llama-server's
OpenAI-compatible autoregressive flow (the chat endpoint returns 500 with
"the current context does not logits computation"), so this preset shells
out to the diffusion CLI directly and captures its final generation.

Shares the same Docker image as the sibling [`llama_cpp/`](../llama_cpp/)
preset (`cvl-llama-cpp`). Building this example just delegates to the
sibling's `build.sh` if the image isn't already present.

## Quick start

```bash
bash examples/generative/llm/llama_cpp_dllm/build.sh    # builds cvl-llama-cpp if missing
bash examples/generative/llm/llama_cpp_dllm/predict.sh  # default: LLaDA-8B-Instruct Q4_K_M
```

Defaults:
- Model: `mradermacher/LLaDA-8B-Instruct-GGUF:Q4_K_M` (set `MODEL_ID` to change, or `MODEL_PATH` for a local `.gguf`)
- `-ngl 999` on GPU (full layer offload), `0` on CPU
- `--diffusion-steps 128`, `--diffusion-algorithm 4` (confidence-based)
- LLaDA family auto-defaults: `--diffusion-block-length 32`, `--diffusion-eps 0`, `--diffusion-cfg-scale 0`
- Dream family auto-defaults: `--diffusion-eps 1e-3`, `--diffusion-block-length 0`

## Why a separate preset

`llama-diffusion-cli` is a one-shot CLI (no HTTP server, no OpenAI API). Diffusion
LMs decode in parallel masked steps, not autoregressively, so they don't have a
running KV cache that llama-server's `/v1/chat/completions` expects. The CLI
takes a prompt, runs N diffusion steps, prints the generation, and exits.

`predict.py` runs the CLI, captures stdout+stderr, and extracts the text after
the final `total time:` marker (the line just before the generation).

## Family detection

Auto-detected from the model id substring:

- contains `dream` → Dream defaults (`--diffusion-eps 1e-3`, block_length 0)
- otherwise → LLaDA defaults (`--diffusion-block-length 32`, eps 0)

Override either explicitly via `LLAMA_DIFFUSION_BLOCK_LENGTH` /
`LLAMA_DIFFUSION_EPS`. `llama-diffusion-cli` asserts that exactly one of these
is non-zero per invocation.

## Model-specific overrides

```bash
# LLaDA 8B Instruct (default; older but the most-quantized open LLaDA today)
MODEL_ID=mradermacher/LLaDA-8B-Instruct-GGUF:Q4_K_M \
LLAMA_DIFFUSION_STEPS=128 \
bash examples/generative/llm/llama_cpp_dllm/predict.sh --max-tokens 256

# LLaDA 1.5 (community successor)
MODEL_ID=mradermacher/LLaDA-1.5-GGUF:Q4_K_M \
bash examples/generative/llm/llama_cpp_dllm/predict.sh
```

LLaDA **2.0 / 2.1** (the strongest current dLLMs) don't have community GGUFs
yet on Hugging Face — only the autoregressive-friendly base checkpoints exist
at `inclusionAI/LLaDA2.{0,1}-mini` / `…-flash`. Revisit when quants land.

## Quality knobs

- `LLAMA_DIFFUSION_STEPS` — more steps generally improves quality (default 128, try 64 for speed, 256 for max).
- `LLAMA_DIFFUSION_ALGORITHM` — 0=origin, 1=entropy, 2=margin, 3=random, 4=confidence (default 4).
- `LLAMA_DIFFUSION_CFG_SCALE` — LLaDA classifier-free guidance; non-zero values can sharpen instruction following.

## Notes
- `--diffusion-visual` mode (progressive token-by-token redraw) is intentionally
  off in this preset — the raw output is cleaner. Enable with
  `LLAMA_EXTRA_ARGS="--diffusion-visual"` if you want it interactively.
- Output capture is best-effort: we strip everything up to and including the
  last `total time:` line and save the tail. The CLI doesn't have a structured
  output mode today.
