# qed-nano

Dockerized inference for [QED-Nano](https://huggingface.co/lm-provers/QED-Nano), a 4B-parameter language model trained to generate natural-language proofs for Olympiad-level mathematical competition problems.

QED-Nano is based on Qwen3-4B and was post-trained with supervised fine-tuning (SFT) on curated proof data followed by GRPO reinforcement learning using proof correctness as the reward signal. Despite its size, it approaches Gemini 3 Pro on the IMOProofBench and outperforms GPT-OSS-120B.

The model outputs a chain-of-thought reasoning trace (inside `<think>...</think>`) followed by the final proof. This wrapper extracts and presents the proof; the thinking trace can be included with `--show-thinking`.

## Requirements

- Docker with NVIDIA GPU support (`--gpus all`)
- ~10 GB VRAM (A100 40GB, H100, or similar)
- A Hugging Face token is **not required** — the model is public

## Quick start

```bash
cd examples/agentic/formal/qed-nano

# Build
bash build.sh

# Prove the default example problem
bash predict.sh

# Prove your own problem
bash predict.sh --problem "Prove that there are infinitely many prime numbers."

# Show the chain-of-thought alongside the proof
bash predict.sh --problem "Prove that sqrt(2) is irrational." --show-thinking

# Output as JSON
bash predict.sh --problem "..." --format json --output outputs/result.json

# Smoke test (short 2048-token generation)
bash test.sh
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--problem` | Simple induction example | Mathematical problem to prove |
| `--model` | `lm-provers/QED-Nano` | HuggingFace model ID |
| `--max-tokens` | `8192` | Max tokens to generate |
| `--temperature` | `0.8` | Sampling temperature (matches training config) |
| `--max-model-len` | `16384` | vLLM context window |
| `--gpu-memory-utilization` | `0.9` | Fraction of VRAM for vLLM |
| `--enforce-eager` | off | Disable torch.compile |
| `--show-thinking` | off | Include chain-of-thought in output |
| `--format` | `txt` | Output format: `txt` or `json` |
| `--output` | `outputs/proof.txt` | Output file path |
| `--verbose` | off | Enable verbose logging |

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `lm-provers/QED-Nano` | Model to load |
| `HF_TOKEN` | _(empty)_ | HuggingFace token (not needed for this public model) |
| `QED_NANO_IMAGE` | `qed-nano` | Docker image name |

## Output

By default, `outputs/proof.txt` contains:
```
=== Proof ===
<proof text>
```

With `--show-thinking`:
```
=== Chain of Thought ===
<reasoning trace>

=== Proof ===
<proof text>
```

With `--format json`, `outputs/proof.json` contains:
```json
{
  "problem": "...",
  "thinking": "...",
  "proof": "..."
}
```

## Notes

- The model targets **informal natural-language proofs**, not formal Lean proofs. For formal theorem proving with Lean, see `examples/agentic/formal/nanoproof/`.
- Proof quality scales with `--max-tokens`. Hard Olympiad problems may benefit from `--max-tokens 16384` or higher.
- The model produces variable-length reasoning traces; longer traces generally correspond to harder problems.

## References

- Model: [lm-provers/QED-Nano](https://huggingface.co/lm-provers/QED-Nano)
- Blog post: [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost)
- Training data (SFT): [lm-provers/FineProofs-SFT](https://huggingface.co/datasets/lm-provers/FineProofs-SFT)
- Training data (RL): [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL)
- Repository: [CMU-AIRe/QED-Nano](https://github.com/CMU-AIRe/QED-Nano)
