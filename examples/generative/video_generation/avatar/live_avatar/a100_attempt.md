# LiveAvatar on 8x A100-40GB — attempt log

Date: 2026-05-18
Host: `acasia` (`adx-server`) — 8x NVIDIA A100-PCIE-40GB (Ampere, SM80)

**Outcome: LiveAvatar could not be run on this hardware.** It needs >=80GB Hopper
GPUs (H800/H200). Documented here so the next person does not repeat the attempt.

## Environment setup (this part worked)

Set up the upstream repo directly (not the dockerized `cvl run` example, since the
host's Docker socket was not accessible without sudo):

- Cloned https://github.com/Alibaba-Quark/LiveAvatar to `~/zz/LiveAvatar`.
- Python 3.10 venv via `uv` (host only had system Python 3.12; repo wants 3.10).
- `torch==2.8.0+cu128`, `torchvision==0.23.0`, `torchaudio`.
- `requirements.txt` installed.
- `flash-attn==2.8.3` — built-from-source fails (no `nvcc`/`CUDA_HOME` on host);
  fixed by installing the prebuilt wheel:
  `flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`
  from the Dao-AILab GitHub releases.
- `deepspeed` had to be **uninstalled** — importing `transformers` wav2vec2
  triggers a deepspeed CUDA-op JIT compile that fails with no `CUDA_HOME`.
  deepspeed is training-only; not needed for inference.
- Checkpoints downloaded (48 GB): `Wan-AI/Wan2.2-S2V-14B` + `Quark-Vision/Live-Avatar`.

## Inference attempts (all failed)

| # | Config | Result |
|---|--------|--------|
| 1 | Multi-GPU TPP, FP8 | `ValueError: type fp8e4nv not supported in this architecture` |
| 2 | Multi-GPU TPP, bf16, no offload | CUDA OOM — 39.5 GB on rank 0 (T5 + DiT) |
| 3 | Multi-GPU TPP, bf16, `--t5_cpu` + `--offload_model` | Hung: 5 ranks 99% CPU / 0% GPU for 13+ min (NCCL spin-deadlock) |
| 4 | Multi-GPU TPP, bf16, `--t5_cpu`, no offload | Same hang |
| 5 | Single-GPU, bf16, `--offload_model --t5_cpu --offload_kv_cache` | Stalled at "Generating video..." — model on GPU (37 GB), 0% GPU 10+ min |

## Root cause — hardware mismatch

1. **FP8 is physically unavailable on A100.** LiveAvatar's 48 GB-capable path
   relies on FP8 `e4m3` (`fp8e4nv`), which requires Hopper/Ada (SM89/SM90).
   Ampere (SM80) only supports `fp8e5` / `fp8e4b15`. Triton aborts at compile time.
   This eliminates the only documented path that fits a <=48 GB GPU.

2. **bf16 needs ~80 GB.** The 14B DiT is ~28 GB in bf16, T5 (umt5-xxl) ~11 GB,
   plus VAE + activations + streaming KV-cache. The CVlization example README
   itself states "1x 80GB+ VRAM (H100, H200, A100-80GB)". 40 GB is below spec.

3. **The multi-GPU TPP pipeline deadlocks here.** Timestep-forcing pipeline
   parallelism is flagged "experimental" upstream and tuned for 5x H800. On this
   8x A100 box all ranks busy-spin on a NCCL collective that never completes.
   The single-GPU path also stalls during generation.

## Implication for this CVlization example

The `live_avatar` example is single-GPU and assumes >=80 GB. There is **no
sub-80 GB path** in either the example or upstream. To support 40 GB Ampere GPUs
would require code changes upstream does not provide — e.g. true FSDP weight
sharding of the DiT across GPUs (the streaming pipelines do not support this),
or an Ampere-compatible int8 quantization path instead of FP8.

## Recommendation

For 40 GB Ampere GPUs, use **SoulX-FlashTalk**
(https://github.com/Soul-AILab/SoulX-FlashTalk) instead — same model family
(Wan-based, DMD-distilled, 4-step, autoregressive streaming) but it uses
**int8 `optimum-quanto` quantization**, which runs on Ampere. Its single-GPU
`--cpu_offload` path is documented for 40 GB and was verified working on `acasia`
(~35 s per 33-frame chunk, GPU at 39.2 GB / 100% util).
