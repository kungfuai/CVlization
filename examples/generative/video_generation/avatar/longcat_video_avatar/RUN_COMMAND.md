# LongCat-Video-Avatar Run Commands

## Generate ~30s talking head video (6 segments)

```bash
cd examples/generative/video_generation/avatar/longcat_video_avatar
CUDA_VISIBLE_DEVICES=0 ./predict.sh --num-segments 6 --verbose
```

Uses default sample inputs (man.png + man.mp3).

## Custom inputs

```bash
CUDA_VISIBLE_DEVICES=0 ./predict.sh \
  --image /path/to/image.png \
  --audio /path/to/speech.wav \
  --num-segments 6 \
  --output my_video.mp4
```

## Segment duration

Each segment adds ~5 seconds. VRAM stays constant regardless of total length.

| Segments | Duration |
|----------|----------|
| 1        | ~6s      |
| 3        | ~16s     |
| 6        | ~31s     |
| 12       | ~1 min   |
