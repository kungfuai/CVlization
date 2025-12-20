# Qwen-Image-Layered (RGBA Decomposition)

Decompose an input image into multiple RGBA layers with Qwen-Image-Layered, export a layered PPTX, and optionally edit a single RGBA layer with Qwen-Image-Edit.

## Requirements

- Docker + NVIDIA GPU (16GB+ VRAM recommended)
- `nvidia-container-toolkit` installed on the host

## Build

```bash
./build.sh
```

## CLI Decomposition

```bash
./predict.sh assets/test_images/1.png
```

Optional arguments (passed through):

```bash
./predict.sh assets/test_images/1.png --layers 6 --num-inference-steps 30 --pptx outputs/layers.pptx
./predict.sh assets/test_images/1.png --device-map balanced --num-inference-steps 20 --layers 3
```

Outputs are saved to `outputs/`.

## Gradio UI (Decomposition + PPTX Export)

```bash
./serve.sh
```

Open `http://localhost:7869` in your browser. Pass a different port as the first argument:

```bash
./serve.sh 7871
```

## RGBA Layer Edit UI

```bash
./edit.sh
```

Open `http://localhost:7870`. Pass a different port as the first argument:

```bash
./edit.sh 7872
```

## Notes

- Models are downloaded from HuggingFace on first run:
  - `Qwen/Qwen-Image-Layered`
  - `Qwen/Qwen-Image-Edit-2509`
  - `briaai/RMBG-2.0`
- Cache location: `~/.cache/cvlization/qwen_image_layered`.
- Set `QWEN_LAYERED_DEVICE_MAP` to `balanced` (default) or `cuda` for the Gradio app.
- The example uses `diffusers` from GitHub to ensure compatibility with the latest pipeline.
- VRAM can spike during the denoising loop. If you hit CUDA OOM with the default 50 steps, reduce `--num-inference-steps` and/or `--layers` (for example: `--num-inference-steps 20 --layers 3`).

## License

The upstream Qwen-Image-Layered project is Apache 2.0 licensed. See `LICENSE` in this example directory for details.
