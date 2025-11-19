# LLaVA-NeXT-Video-7B (video captioning)

Dockerized transformers example for `llava-hf/LLaVA-NeXT-Video-7B-hf` (video+text → text).

## Quick Start

```bash
cd examples/perception/vision_language/llava_next_video

# Build image
bash build.sh

# Run a caption request (defaults to a tiny sample video URL, cached under /root/.cache/llava_next_video)
bash predict.sh
```

## Presets (via CVL CLI)

```bash
cvl run llava-next-video build
cvl run llava-next-video predict    # caption the default sample video URL
cvl run llava-next-video test       # same as predict
```

## Options

- `--video`: Path/URL to a video (MP4). Defaults to a small sample URL (cached in `/root/.cache/llava_next_video`).
- `--prompt`: Prompt to apply (default: “Describe the video in detail.”).
- `--max-frames`: Number of frames to sample uniformly from the video (default: 8).
- `--max-new-tokens`: Generation cap (default: 128).
- `--temperature`: Sampling temperature (default: 0.2).
- `--top-p`: Nucleus sampling (default: 0.9).
- `--output`: Output path (default: `outputs/llava_next_video.txt`).
- `--format`: `txt` or `json`.

Environment vars:
- `LLAVA_NEXT_VIDEO_MODEL_ID`: Override model (default: `llava-hf/LLaVA-NeXT-Video-7B-hf`).
- `LLAVA_NEXT_VIDEO_CACHE`: Override video cache dir (default: `/root/.cache/llava_next_video`, mounted from host).
- `CVL_IMAGE`: Override Docker image tag (default: `llava-next-video`).

## Notes

- Uses `AutoProcessor` + `LlavaNextVideoForConditionalGeneration` in bf16 with device map auto-detected.
- Video frames are decoded with `decord` and uniformly sampled.
- Remote downloads (video URLs) are saved under `/tmp`.
- Hugging Face cache is mounted into the container to reuse weights.
