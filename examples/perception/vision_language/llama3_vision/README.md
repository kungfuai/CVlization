# Llama 3.2 Vision (11B) with Transformers

Self-contained Docker example that runs the `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit` checkpoint via Hugging Face Transformers. Works with a local image (single-image, single-turn).

## Quick start

```bash
cd examples/perception/vision_language/llama3_vision

# Build the image
bash build.sh
# or: cvl run llama3_vision build

# Single-image caption (uses shared test image)
bash predict.sh --prompt "Describe the document."

# Smoke test helper
bash test.sh
```

> Requires an NVIDIA GPU (tested with CUDA 12.x). Export `HUGGING_FACE_HUB_TOKEN` (or `HF_TOKEN`) if you use a gated model; the RedHatAI default is public at the time of writing.

## Script options

```
python predict.py \
  [--model HF_ID]                  # default: unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit
  [--image path]                  # local image (default: test_images/checkbox_page.png)
  [--prompt "Describe this image"]# user message appended after <|image|>
  [--max-tokens 64]               # generation length
  [--device auto|cpu|cuda|mps]    # auto-detect by default
  [--output outputs/result.txt]   # optional save path
```

The prompt is formatted via `processor.apply_chat_template` for a single user turn with an image attachment.

## CVL presets

- `cvl run llama3_vision build` – build the Docker image
- `cvl run llama3_vision predict -- --prompt "Describe the scene." --image /path/to/img.jpg`
- `cvl run llama3_vision test` – quick smoke test

## Model compatibility notes

### 8-bit quantization issues

We attempted to load `unsloth/Llama-3.2-11B-Vision-Instruct` (non-quantized) with 8-bit quantization via `BitsAndBytesConfig(load_in_8bit=True)` but encountered compatibility issues with bitsandbytes 0.48.2:

1. Initial error: `RuntimeError: view size is not compatible with input tensor's size and stride` in `bitsandbytes/backends/cuda/ops.py:145`
2. After patching `.view(-1)` → `.reshape(-1)`: CUDA kernel errors (`index out of bounds`, `device-side assert triggered`)

**Conclusion**: The `MllamaForConditionalGeneration` architecture has fundamental incompatibilities with bitsandbytes 8-bit quantization as of November 2025.

**Recommended**: Use the pre-quantized 4-bit model (`unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`, 9.6GB VRAM) which works reliably, or use the full fp16/bf16 model on GPUs with 40GB+ VRAM.

## Files

- `predict.py` – minimal Transformers runner
- `predict.sh` – docker wrapper mounting HuggingFace cache and repo
- `test.sh` – runs `predict.sh` with a short caption prompt
- `Dockerfile` – PyTorch CUDA base + `vllm`, `pillow`
- `example.yaml` – CVL CLI metadata
