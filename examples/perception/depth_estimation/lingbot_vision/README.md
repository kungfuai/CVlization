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
./build.sh          # or: cvl run lingbot_vision build

# Run (uses bundled sample images on first run)
./predict.sh        # or: cvl run lingbot_vision predict
```

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
./predict.sh --input /path/to/my/images --output /path/to/results
```

## Requirements

- Docker with NVIDIA GPU support (or CPU-only with `--device cpu`)
- ~2 GB VRAM for ViT-Large at 512px
- ~8 GB VRAM for ViT-Giant at 512px

## Model variants

| Variant | Backbone | Embed dim | HuggingFace |
|---------|----------|-----------|-------------|
| small | ViT-S/16 | 384 | `robbyant/lingbot-vision-vit-small` |
| base | ViT-B/16 | 768 | `robbyant/lingbot-vision-vit-base` |
| large | ViT-L/16 | 1024 | `robbyant/lingbot-vision-vit-large` |
| giant | ViT-g/16 (1.1B) | 1536 | `robbyant/lingbot-vision-vit-giant` |

## References

- Fu et al., "Vision Pretraining for Dense Spatial Perception", arXiv 2607.05247, 2026
- [robbyant/lingbot-vision-vit-large](https://huggingface.co/robbyant/lingbot-vision-vit-large)
