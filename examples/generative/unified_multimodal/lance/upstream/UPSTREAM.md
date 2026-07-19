# Vendored Upstream Source

This directory contains the minimal inference subtree of the Lance unified
multimodal model by ByteDance Research, vendored for reproducibility and
reduced Docker image size.

## Source

- **Repository**: https://github.com/bytedance/Lance
- **Commit**: `3c757d257eecf9b88b83124c63883038bafccf72`
- **License**: Apache 2.0 (see `LICENSE` in this directory)
- **Vendored on**: 2026-07-19

## Included paths

Only the files required for the inference pipeline (`inference_lance.py`)
are included:

```
upstream/
  LICENSE                      # Apache 2.0 license
  UPSTREAM.md                  # This file
  inference_lance.py           # Main inference entrypoint
  common/                      # Utility modules (distributed, logging, misc, val)
  config/                      # Config factory + path_default.yaml
  data/                        # Dataset loading, transforms, video sampling
  modeling/                    # Lance model, Qwen2 LLM, VIT, VAE
```

Total: 47 files, ~788 KB (vs. 229 MB full repo with .git and assets).

## Intentionally omitted paths

| Path | Reason |
|------|--------|
| `.git/` (131 MB) | Git history not needed at runtime |
| `assets/` (92 MB) | README images, not used by inference |
| `benchmarks/` (1.8 MB) | Evaluation scripts, not inference |
| `config/examples/` (4 MB) | Upstream sample inputs (images/videos); CVlization provides its own via `zzsi/cvl` |
| `lance_gradio_t2v_v2t.py` | Gradio demo UI |
| `inference_lance.sh` | Upstream shell wrapper (CVlization uses `predict.sh`) |
| `setup_env.sh` | Environment setup script |
| `requirements.txt` | Upstream deps (CVlization has its own `requirements.txt`) |
| `README.md`, `README_zh.md`, `SECURITY.md` | Documentation |

## CVlization modifications

No upstream source files have been modified. All CVlization integration
logic is in the root-level `predict.py` wrapper, which constructs the
appropriate input JSON and invokes `inference_lance.py` via `accelerate`.
