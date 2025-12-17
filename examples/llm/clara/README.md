# CLaRa Inference (vendored modeling code)

Dockerized, in-process inference demo for Apple's CLaRa 7B (RAG compressor/generator). Uses the torch 2.9.1 CUDA12.8 runtime, vendors `modeling_clara.py`, and runs a single QA with locally copied model code (no external server).

## Quick start
```bash
# From repo root
bash examples/llm/clara/build.sh
HF_TOKEN=... bash examples/llm/clara/predict.sh --max-tokens 64
```

Defaults:
- Model: `apple/CLaRa-7B-Instruct` (set `MODEL_ID` or `--model` to change)
- Compression variant: `compression-16` (override via `COMPRESSION_DIR` or `--compression-dir`)
- GPU: expects ≥24GB for bf16; runs on A10 in testing
- Base: `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`

What this provides on top of upstream:
- Vendors `modeling_clara.py` so inference works offline after HF weight download.
- Copies the vendored file into the HF cache before loading to avoid remote code execution.
- Minimal dependencies (transformers/peft/sentencepiece/hf-hub).

Plan for training support (future): add a heavier preset with Deepspeed/FlashAttention matching upstream scripts; not included yet to keep the inference image slim.

## Usage notes
- Mounts `~/.cache/huggingface`; set `HF_TOKEN` for gated or high-throughput downloads.
- Modes:
  - `--mode text` → calls `generate_from_text` (stage1_2 instruction-tuned path). Inputs: one question + list of docs; generates an answer.
  - `--mode paraphrase` → calls `generate_from_paraphrase` (stage1 compression path). Inputs: one question + list of docs; returns a paraphrase-style answer.
  - `--mode questions` → calls `generate_from_questions` (stage3 end-to-end with retrieval/top-k). Requires a checkpoint that ships `query_reasoner_adapter` (the default `apple/CLaRa-7B-Instruct` does not); use `MODEL_ID=apple/CLaRa-7B-E2E`, `COMPRESSION_DIR=compression-16`.
- Default docs mirror the toy plant QA example from the upstream notebook; override via `--docs ...` or `--prompt ...`.
- Outputs are written to `outputs/result.txt`.
- Verified: `cvl run clara predict` on A10 (24GB) with `apple/CLaRa-7B-Instruct`, `compression-16`, `--max-tokens 32` (bf16). Answer saved to `outputs/result.txt`.
- Stage3 (`questions`) verified on A10 with `MODEL_ID=apple/CLaRa-7B-E2E`, `COMPRESSION_DIR=compression-16`, `--max-tokens 64` (bf16).
