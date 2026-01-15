# Wan2GP CLI (LTX-2, Wan, LongCat)

This example wraps [Wan2GP](https://github.com/DeepBeepMeep/Wan2GP) by DeepBeepMeep for CLI usage. Wan2GP provides optimized video generation with smart memory management via [mmgp](https://github.com/deepbeepmeep/mmgp) (Memory Management for the GPU Poor).

It supports:
- LTX-2 (full + distilled)
- Wan T2V / I2V
- LongCat video + avatar

## Quick start

Build the image:
```bash
./build.sh
```

Generate with LTX-2:
```bash
cvl run wan2gp predict -- --model ltx2_19B --prompt "A cinematic city street at dusk, rain reflections, slow dolly" --output outputs/wan2gp_ltx2.mp4
```

Generate Wan T2V:
```bash
cvl run wan2gp predict -- --model wan_t2v --prompt "A wide aerial shot over rolling hills at sunrise" --output outputs/wan2gp_wan_t2v.mp4
```

Generate Wan I2V:
```bash
cvl run wan2gp predict -- --model wan_i2v --image inputs/start.jpg --prompt "The scene comes alive with gentle wind and drifting clouds" --output outputs/wan2gp_wan_i2v.mp4
```

Generate LongCat video:
```bash
cvl run wan2gp predict -- --model longcat_video --prompt "A tiny robot exploring a kitchen countertop, cinematic lighting" --output outputs/wan2gp_longcat.mp4
```

Generate LongCat avatar (audio optional):
```bash
cvl run wan2gp predict -- --model longcat_avatar --prompt "A friendly presenter speaks to camera" --audio inputs/voice.wav --output outputs/wan2gp_longcat_avatar.mp4
```

## Memory Profiles

Use `--mmgp-profile` to control memory usage vs speed tradeoff:

| Profile | RAM | VRAM | Speed | Use Case |
|---------|-----|------|-------|----------|
| 1 | 48GB+ | 24GB | Fastest | RTX 4090 with lots of RAM |
| 2 | 48GB+ | 12GB | Fast | RTX 4070/4080 with lots of RAM |
| 3 | 32GB | 24GB | Medium | RTX 4090 with less RAM |
| 4 | 32GB | 12GB | Slow | Most consumer GPUs (default) |
| 5 | 24GB | 10GB | Slowest | Minimal hardware |

Example:
```bash
cvl run wan2gp predict -- --model ltx2_19B --mmgp-profile 1 --prompt "..." --output out.mp4
```

## Upstream Relationship

This example vendors code from [Wan2GP](https://github.com/DeepBeepMeep/Wan2GP) with minimal patches for CLI compatibility. The patches are maintained in `patches/` and automatically applied during sync.

**Syncing from upstream:**
```bash
git clone https://github.com/DeepBeepMeep/Wan2GP.git /tmp/Wan2GP
WAN2GP_SRC=/tmp/Wan2GP ./sync_vendor.sh
```

See `patches/README.md` for details on what modifications are applied.

## Notes

- Vendored sources: `vendor/wan2gp/`
- Model cache: `~/.cache/wan2gp` (override with `WAN2GP_CKPT_CACHE` env var)
- HuggingFace cache: `~/.cache/huggingface` (override with `HF_HOME` env var)
- Output paths are resolved relative to the host working directory when using `cvl run`.
