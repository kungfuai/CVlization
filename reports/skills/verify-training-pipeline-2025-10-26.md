## Mistral-7B Finetune & Mixtral-8x7B Inference (2025-10-26)
- `mistral7b`: build works (when invoked via `bash`), but `build.sh` lacks execute bit so `cvl run ... build` fails. Training script downloads the full 7B base model, requires HF auth, and hardcodes 1000 steps with no quick-test overrides—impractical for verification on shared GPU.
- `mixtral8x7b`: build succeeds, yet `example.yaml` references a missing `predict.sh`. `generate.sh` mounts wrong repo path, depends on `huggingface-cli`/`IPython`, and assumes pre-downloaded offloaded weights; container doesn’t install those tools. No easy way to run smoke inference or capture outputs.
- Result: neither example passes the training-pipeline verification checklist; both need scripting and UX fixes before they can be verified.

## Mistral-7B & Mixtral-8x7B Updates (2025-10-26)
- Added smoke-test friendly workflows:
  - `mistral7b` now exposes CLI flags for dataset size, steps, and LoRA params; bundled `data/*.json` enables 1-step TinyLlama runs without external downloads.
  - `mixtral8x7b` ships a new `predict.py`/`predict.sh` pair using Transformers, optional 4-bit loading, and robust dependency handling (automatic `sentencepiece` install, bitsandbytes fallback, configurable prompts/output files).
- Verified via `cvl run` that both examples build successfully and complete quick runs (TinyLlama finetune + TinyLlama inference).
- Users with HF access can still target the gated `mistralai/Mistral-7B-v0.1` or `mistralai/Mixtral-8x7B-Instruct-v0.1` models by overriding `--model_id`; build scripts now run `docker build --pull --no-cache` to ensure fresh images.
