# GLM-5V-Turbo (Z.ai API)

Multimodal agent VLM from Z.ai. GLM-5V-Turbo integrates visual perception as a core component of reasoning, planning, and tool use -- positioned as a native foundation model for multimodal agents rather than a vision module bolted onto a language model.

## Highlights

- **Model**: `glm-5v-turbo` via Z.ai API (OpenAI-compatible)
- **Tasks**: captioning, OCR, VQA, GUI grounding, visual reasoning
- **No GPU required**: runs via API, lightweight Docker image
- **Agent-oriented**: designed for GUI grounding, visual tool use, multimodal coding

## Setup

Get an API key from [Z.ai Open Platform](https://open.z.ai), then:

```bash
export ZAI_API_KEY="your-key-here"
```

## Quick Start

```bash
cd examples/perception/vision_language/glm5v_turbo

# Build the Docker image
bash build.sh

# Caption an image
bash predict.sh --image test_images/sample.jpg --task caption

# GUI grounding on a screenshot
bash predict.sh --image screenshot.png --task gui_grounding

# Visual reasoning with step-by-step thinking
bash predict.sh --image diagram.png --task reasoning --thinking

# VQA with custom prompt
bash predict.sh --image chart.png --task vqa --prompt "What trend is shown?"

# Multi-image input
bash predict.sh --images page1.png page2.png --task ocr
```

## CVL CLI Presets

```bash
cvl run glm5v_turbo build
cvl run glm5v_turbo predict --image test_images/sample.jpg --task caption
cvl run glm5v_turbo test
```

## Script Arguments

```
python predict.py \
  [--image path|url]                # single image
  [--images p1 p2 ...]             # multi-image
  [--task caption|ocr|vqa|gui_grounding|reasoning]
  [--prompt "question"]            # required for vqa
  [--max-tokens N]                 # default: 1024
  [--temperature T]                # default: 0.2
  [--thinking]                     # enable step-by-step reasoning
  [--output outputs/result.txt]
  [--format txt|json]
  [--api-key KEY]                  # or set ZAI_API_KEY
  [--base-url URL]                 # override API endpoint
```

## Notes

- GLM-5V-Turbo does not have open-source weights. Inference runs via the Z.ai cloud API.
- The Z.ai API is OpenAI-compatible, so this example uses the `openai` Python package with a custom `base_url`.
- No GPU or HuggingFace cache is needed since the model runs server-side.
- GUI grounding and visual tool use are the distinguishing capabilities vs other VLMs in CVlization (gemma3_vision, qwen3_vl, etc.).

## References

- [GLM-5V-Turbo paper](https://huggingface.co/papers/2604.26752)
- [Z.ai API docs](https://docs.z.ai/guides/vlm/glm-5v-turbo)
- [GLM-V GitHub](https://github.com/zai-org/GLM-V)

## License

API usage is subject to Z.ai's terms of service. This example script is MIT, consistent with the rest of CVlization.
