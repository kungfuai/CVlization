# nanoproof (dockerized inference)

This packages the upstream [nanoproof](https://github.com/kripner/nanoproof) repo for inference/tactic sampling with lazy downloads for the tokenizer and checkpoints. Training datasets are **not** bundled; this is meant for loading an existing checkpoint and sampling tactics (the search pipeline still expects a running Lean server).

## Layout

- `nanoproof_repo/` – vendored upstream nanoproof source (unchanged)
- `Dockerfile` – GPU image for the model / tactic sampler (Python 3.12, CUDA 12.8, torch 2.9.1)
- `Dockerfile.lean` – optional sidecar image to run a Lean server (`lean-repl`)
- `predict.py` – CLI wrapper that lazy-downloads assets then calls the existing `TacticModel.sample_tactic`
- `build.sh` / `predict.sh` – helper scripts to build and run the container

## Requirements / assets

- GPU with recent CUDA (image uses PyTorch CUDA 12.8 / torch 2.9.1)
- Environment variables (used by `predict.py`):
  - `NANOPROOF_TOKENIZER_REPO_ID` – HF repo containing `tokenizer/tokenizer.json` and `tokenizer/token_bytes.pt`
  - `NANOPROOF_CHECKPOINT_REPO_ID` – HF repo containing SFT checkpoints (`model_*.pt`, `meta_*.json`) under `sft_checkpoints/<model_tag>`
  - Optional: `NANOPROOF_MODEL_TAG` (default `d26`), `NANOPROOF_CHECKPOINT_REVISION`, `HF_TOKEN`
  - Optional local overrides: `NANOPROOF_TOKENIZER_LOCAL`, `NANOPROOF_CHECKPOINT_LOCAL`
- Lean server: not required for pure sampling; required for MCTS/proof search. Build the sidecar with `Dockerfile.lean` and run it separately (expose port 8000).

## Quick start (sampling)

```bash
cd examples/agentic/formal/nanoproof
bash build.sh

# Run the container to generate tactics
HF_TOKEN=... \
NANOPROOF_TOKENIZER_REPO_ID=your/tokenizer-repo \
NANOPROOF_CHECKPOINT_REPO_ID=your/checkpoint-repo \
docker run --rm -it --gpus all \
  -e HF_TOKEN \
  -e NANOPROOF_TOKENIZER_REPO_ID \
  -e NANOPROOF_CHECKPOINT_REPO_ID \
  nanoproof:latest \
  python predict.py --state "$(cat sample_state.txt)" --num-samples 3
```

If you have the tokenizer/checkpoints locally, mount them and point the env vars to the directory:

```bash
docker run --rm -it --gpus all \
  -v /path/to/cache:/cache \
  -e NANOPROOF_TOKENIZER_LOCAL=/cache/tokenizer \
  -e NANOPROOF_CHECKPOINT_LOCAL=/cache/sft_checkpoints/d26 \
  nanoproof:latest \
  python predict.py --state "⊢ 2 + 3 = 5" --num-samples 2
```

## Lean server sidecar (optional)

`Dockerfile.lean` builds a Lean 4.19 environment and `lean-repl` from the upstream leantree repo. Run it separately and point nanoproof to it via `LEAN_SERVER_HOST/PORT` (the current code defaults to `10.10.25.34:8000`; adjust via env before launching `scripts/prover_eval.py`).

### Run the Lean sidecar
Build:
```bash
docker build -t nanoproof-lean -f Dockerfile.lean .
```
Run:
```bash
docker run --rm -p 8000:8000 nanoproof-lean
```
Then set `LEAN_SERVER_HOST=host.docker.internal` (or the container host IP) and `LEAN_SERVER_PORT=8000` for scripts that talk to the Lean server.

## Notes / limitations

- Upstream does **not** publish tokenizer or pretrained checkpoints nor HF repo IDs; you must supply them (via envs or local paths). Without those artifacts the pretrained behavior cannot be reproduced from the upstream repo alone.
- Upstream pins `regex>=2025.9.1` and `torch>=2.8.0`; the Dockerfile uses current, available versions instead.
- `TacticModel.create()` assumes CUDA; the container is GPU-only.
- Dataset downloads (Nemotron, LeanTree) are not performed; only tokenizer/checkpoint fetching is wired.
