# transformers-inference

Dockerized HuggingFace Transformers inference for any causal LM. Useful for models not yet well-supported by vLLM or SGLang, such as hybrid architectures like [OLMo-Hybrid-7B](https://huggingface.co/allenai/Olmo-Hybrid-7B).

Default model: `allenai/Olmo-Hybrid-Instruct-DPO-7B`

## What to expect

- **First-run download**: ~14GB (OLMo-Hybrid-Instruct-DPO-7B in bfloat16), cached in `~/.cache/huggingface/` afterward
- **VRAM**: ~14GB for the default 7B model
- **Runtime**: ~30s to first token on RTX PRO 6000 (98GB), slower on smaller GPUs
- **Output**: text response saved to `result.txt` in your current working directory

## Quick start

```bash
# Build
cvl run transformers-inference build

# Run inference (default model)
cvl run transformers-inference predict

# Custom prompt
cvl run transformers-inference predict -- --prompt "Explain diffusion language models."

# Different model
MODEL_ID=allenai/Olmo-3-7B-Instruct cvl run transformers-inference predict
```

## Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--model` | `allenai/Olmo-Hybrid-Instruct-DPO-7B` | HuggingFace model ID |
| `--prompt` | *(fun fact prompt)* | User message |
| `--system` | `""` | System prompt |
| `--max-tokens` | `256` | Max new tokens |
| `--temperature` | `0.0` | Sampling temperature (0 = greedy) |
| `--dtype` | `bfloat16` | Model dtype |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `allenai/Olmo-Hybrid-Instruct-DPO-7B` | Model to load |
| `HF_TOKEN` | *(unset)* | HuggingFace token for gated models |
| `DTYPE` | `bfloat16` | Model dtype |
| `CUDA_VISIBLE_DEVICES` | *(all)* | GPU selection |

## OLMo-Hybrid notes

OLMo-Hybrid-7B uses a **Gated DeltaNet + attention hybrid architecture** (75% linear RNN layers, 25% attention). As of March 2026, vLLM requires workaround flags (`--disable-cascade-attn --enforce-eager`) that kill throughput, and SGLang support is unconfirmed. Transformers ≥5.3.0 is the officially recommended inference path.

```bash
# Base model
MODEL_ID=allenai/Olmo-Hybrid-7B cvl run transformers-inference predict

# Instruct (DPO) model — best for chat
MODEL_ID=allenai/Olmo-Hybrid-Instruct-DPO-7B cvl run transformers-inference predict

# Thinking/reasoning model
MODEL_ID=allenai/Olmo-Hybrid-Think-SFT-7B cvl run transformers-inference predict
```

## Serving (TGI)

`serve.sh` uses the official [HuggingFace Text Generation Inference](https://github.com/huggingface/text-generation-inference) image — no build step needed, it pulls automatically.

```bash
# Start server (OpenAI-compatible on port 8080)
cvl run transformers-inference serve

# Custom model / port
MODEL_ID=allenai/Olmo-Hybrid-Think-SFT-7B PORT=8081 cvl run transformers-inference serve

# Call the server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tgi", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Environment variables: `MODEL_ID`, `PORT` (default 8080), `HF_TOKEN`, `CUDA_VISIBLE_DEVICES`, `TGI_EXTRA_ARGS`, `TGI_IMAGE`.

## References

- [OLMo-Hybrid-7B on HuggingFace](https://huggingface.co/allenai/Olmo-Hybrid-7B)
- [Ai2 blog: Introducing OLMo Hybrid](https://allenai.org/blog/olmohybrid)
- [Gated DeltaNet (ICLR 2025)](https://github.com/NVlabs/GatedDeltaNet)
