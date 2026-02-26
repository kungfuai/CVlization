# qed-nano

Dockerized inference for [QED-Nano](https://huggingface.co/lm-provers/QED-Nano), a 4B-parameter language model trained to generate natural-language proofs for Olympiad-level mathematical competition problems.

QED-Nano is based on Qwen3-4B and was post-trained with supervised fine-tuning (SFT) on curated proof data followed by GRPO reinforcement learning using proof correctness as the reward signal. Despite its size, it approaches Gemini 3 Pro on the IMOProofBench and outperforms GPT-OSS-120B.

The model outputs a chain-of-thought reasoning trace (inside `<think>...</think>`) followed by the final proof. This wrapper extracts and presents the proof; the thinking trace can be included with `--show-thinking`.

## Requirements

- Docker with NVIDIA GPU support (`--gpus all`)
- ~10 GB VRAM — works on consumer GPUs (RTX 3090/4090/5090) as well as data-centre GPUs (A100, H100). Verified on Blackwell SM120.
- A Hugging Face token is **not required** — the model is public

## Quick start

```bash
cd examples/agentic/formal/qed-nano

# Build (~5 min on first run)
bash build.sh

# Prove the default example problem (AM-GM inequality)
bash predict.sh

# Run a named problem
bash predict.sh --named-problem imo1988p6   # IMO 1988 P6 (Vieta jumping)
bash predict.sh --named-problem imo2000p2   # IMO 2000 P2 (abc=1 inequality)
bash predict.sh --named-problem amgm        # AM-GM inequality (default)

# Prove your own problem (inline or from a file)
bash predict.sh --problem "Prove that there are infinitely many prime numbers."
bash predict.sh --problem @my_problem.txt

# Show the chain-of-thought alongside the proof
bash predict.sh --problem "Prove that sqrt(2) is irrational." --show-thinking

# Output as JSON
bash predict.sh --problem "..." --format json --output outputs/result.json

# Smoke test
bash test.sh
```

## Example output

Running `bash predict.sh` on the default problem produces output like:

```
=== Proof ===
We are asked to prove that for positive real numbers a, b, c,

    (a + b)(b + c)(c + a) ≥ 8abc.

Proof.
For any two positive numbers x, y the arithmetic–geometric mean inequality gives

    (x + y) / 2 ≥ sqrt(xy)  ⟺  x + y ≥ 2·sqrt(xy).

Apply this to the three pairs (a, b), (b, c), (c, a):

    a + b ≥ 2·sqrt(ab),    b + c ≥ 2·sqrt(bc),    c + a ≥ 2·sqrt(ca).

All quantities are positive, so multiply the three inequalities:

    (a + b)(b + c)(c + a) ≥ 8·sqrt(ab)·sqrt(bc)·sqrt(ca).

Now compute the product under the square roots:

    sqrt(ab)·sqrt(bc)·sqrt(ca) = sqrt(a²b²c²) = abc.

Consequently,

    (a + b)(b + c)(c + a) ≥ 8abc.

Equality holds exactly when a = b = c.  ∎
```

First run downloads the model (~8 GB) and takes 2–3 min. Subsequent runs start in ~30 s.

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

## Training

### SFT (`train_sft.sh`)

Fine-tunes Qwen3-4B on FineProofs-SFT using LoRA. Single GPU, ~8 min for a smoke test. Works on any GPU supported by CUDA 12.8 (including Blackwell SM120).

```bash
bash train_sft.sh                    # smoke test: 100 samples, 30 steps
bash train_sft.sh --max-steps 2000   # full run
```

### RL (`train_rl.sh`)

GRPO reinforcement learning via [PipelineRL](https://github.com/CMU-AIRe/QED-Nano). Requires 4+ GPUs and an OpenAI-compatible API for proof grading.

```bash
OPENAI_API_KEY=sk-... bash train_rl.sh --build   # build image, then run (4 GPUs)
```

**Known limitation — SM120 (Blackwell) not supported.** PipelineRL pins `vllm==0.8.5.post1`, which in turn hard-pins `torch==2.6.0`. PyTorch 2.6.0 has no CUDA 12.8 wheels, so there is no path to SM120 kernel support in this dependency chain. RL training runs on Ampere (A100) or Hopper (H100) GPUs. A future RL pipeline built on a newer vLLM (without PipelineRL's internal API coupling) would unblock Blackwell.

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
