# Gemma Scope 2 (270M IT) - SAE Token Analysis

This example runs Gemma Scope 2 sparse autoencoder analysis over Gemma 3 270M IT tokens. It can optionally export HTML and JSON/JSONL reports with top activating SAE features per token.

## Requirements

- NVIDIA GPU recommended (CPU works but is slow)
- Docker with NVIDIA runtime
- Hugging Face access for Gemma 3 base weights (set `HF_TOKEN`)

## Quick Start

```bash
# Build the image
./build.sh

# Run analysis with default prompt
./predict.sh --export html --export json
```

## Using HF_TOKEN from `.env`

If the Gemma 3 weights are gated, set `HF_TOKEN` in a repo-level `.env`:

```
HF_TOKEN=hf_your_token_here
```

The `predict.sh` script loads `HF_TOKEN` from the repo root if present.

## Usage

```bash
# Custom prompt, export HTML and JSONL
./predict.sh \
  --prompt "Explain sparse autoencoders in one paragraph." \
  --max_new_tokens 64 \
  --top_k 5 \
  --export html \
  --export jsonl

# Analyze prompt only (no generation)
./predict.sh --prompt "Define activation patching." --max_new_tokens 0

# Prompt-only analysis without chat wrappers
./predict.sh --prompt "Define activation patching." --max_new_tokens 0 --no_chat
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | `google/gemma-3-270m-it` | Base Gemma model ID |
| `--sae_release` | `google/gemma-scope-2-270m-it` | SAE repo ID |
| `--sae_category` | `resid_post` | Subfolder for SAE weights |
| `--sae_id` | `layer_12_width_16k_l0_small` | SAE identifier |
| `--layer` | `12` | Layer index for hidden states |
| `--top_k` | `5` | Top SAE features per token |
| `--max_new_tokens` | `64` | Generate tokens before analysis |
| `--export` | unset | Export `html`, `json`, or `jsonl` (repeatable) |
| `--out_dir` | `outputs` | Output directory for exports |

## Outputs

- `outputs/gemma_scope_2_report.html` - token-level HTML report (optional)
- `outputs/gemma_scope_2_report.json` - full structured output (optional)
- `outputs/gemma_scope_2_report.jsonl` - per-token rows (optional)

## Notes

- The analysis uses hidden states at the selected layer as a proxy for residual stream activations.
- Change `--sae_id`, `--sae_category`, or `--sae_release` to explore different SAE widths or sparsity targets.
