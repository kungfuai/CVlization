# LingBot-Vision: Dense Spatial Perception

Extracts dense, boundary-aware patch features from images using a Vision
Transformer pretrained with **masked boundary modelling** (self-supervised).
Features are visualised via PCA projection to RGB, revealing the spatial
structure the model learns without any labelled data.

**Paper:** [Vision Pretraining for Dense Spatial Perception](https://arxiv.org/abs/2607.05247)
**Repo:** <https://github.com/Robbyant/lingbot-vision>
**License:** Apache 2.0

## Quickstart

```bash
# Build
./build.sh          # or: cvl run perception/vision_pretraining/lingbot_vision build

# Run (downloads sample images on first run)
CVL_ENABLE_GPU=1 ./predict.sh
# or: CVL_ENABLE_GPU=1 cvl run perception/vision_pretraining/lingbot_vision predict
```

## Sample outputs

Canonical inputs and outputs are hosted at
[`zzsi/cvl/lingbot_vision/`](https://huggingface.co/datasets/zzsi/cvl/tree/main/lingbot_vision).

**Indoor scene** -- PCA features segment the room into distinct regions (wall,
window, desk, monitor) with visible colour transitions at object boundaries:

| Input | PCA Features |
|-------|-------------|
| ![sample_indoor](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_vision/sample_indoor.jpg) | *(see panel below)* |

![indoor panel](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_vision/sample_indoor_panel.png)

**Person scene** -- the backbone separates the person from the sky, with
distinct feature clusters for skin, clothing, hair, and background foliage:

![scene panel](https://huggingface.co/datasets/zzsi/cvl/resolve/main/lingbot_vision/sample_scene_panel.png)

## What it does

1. Loads the LingBot-Vision ViT-Large backbone from HuggingFace
   (`robbyant/lingbot-vision-vit-large`).
2. Resizes each input image to 512x512 and extracts 32x32 = 1024 patch
   tokens, each a 1024-dimensional feature vector.
3. Projects the top-3 PCA components to RGB channels and saves:
   - `<name>_pca.png` -- upscaled PCA feature map
   - `<name>_panel.png` -- side-by-side input vs PCA comparison

## First-run cost

| Item | Size | Cached after first run? |
|------|------|------------------------|
| Docker image build | ~6 GB | Yes |
| Model weights (ViT-L) | ~1.2 GB | Yes (`~/.cache/huggingface`) |
| Sample images | ~0.9 MB | Yes (`data/images/`) |

## Output

Results are written to `outputs/` (or the directory specified by `--output`).

| File | Format | Description |
|------|--------|-------------|
| `*_pca.png` | PNG | PCA feature map (full resolution) |
| `*_panel.png` | PNG | Side-by-side: input image + PCA features |
| `metrics.json` | JSON | Processing summary |

## Options

```
--input PATH      Image file or directory (default: bundled HF samples)
--output PATH     Output directory (default: outputs)
--variant NAME    Model size: small, base, large, giant (default: large)
--image-size N    Resize to NxN pixels (default: 512)
--resize-mode M   square or shortest (default: square)
--dtype TYPE      bf16, fp16, fp32 (default: bf16)
--device DEV      cuda or cpu (default: auto)
```

## Custom images

```bash
CVL_ENABLE_GPU=1 ./predict.sh --input /path/to/my/images --output /path/to/results
```

## Requirements

- Docker with NVIDIA GPU support (or CPU-only with `--device cpu`)
- ~2 GB VRAM for ViT-Large at 512px (measured: 1862 MiB process peak)

## Model variants

Only **ViT-Large** is verified. Other variants are available via `--variant`
but have not been tested; VRAM and output quality may differ.

| Variant | Backbone | Embed dim | HuggingFace | Status |
|---------|----------|-----------|-------------|--------|
| small | ViT-S/16 | 384 | `robbyant/lingbot-vision-vit-small` | untested |
| base | ViT-B/16 | 768 | `robbyant/lingbot-vision-vit-base` | untested |
| **large** | ViT-L/16 | 1024 | `robbyant/lingbot-vision-vit-large` | **verified** |
| giant | ViT-g/16 (1.1B) | 1536 | `robbyant/lingbot-vision-vit-giant` | untested |

## References

- Fu et al., "Vision Pretraining for Dense Spatial Perception", arXiv 2607.05247, 2026
- [robbyant/lingbot-vision-vit-large](https://huggingface.co/robbyant/lingbot-vision-vit-large)
