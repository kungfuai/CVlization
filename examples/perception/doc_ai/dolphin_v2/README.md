# Dolphin-v2 (document parsing)

Dockerized inference example for `ByteDance/Dolphin-v2` (Qwen2.5-VL-based) to parse documents from images into structured text.

## Quickstart
```bash
./build.sh
./predict.sh --output outputs/result.txt
# or point to your own image:
# ./predict.sh --image /path/to/your_image.jpg --output outputs/result.txt
```

## Details
- GPU-recommended (tested for A10-class VRAM; weights ~7.5GB, BF16 by default).
- Downloads weights from Hugging Face; cache is mounted from `~/.cache/huggingface`.
- Default prompt keeps tables as HTML and formulas as LaTeX; override with `--prompt`.
- Default sample input reuses `examples/perception/doc_ai/leaderboard/test_data/sample.jpg` from the repo.

## License
The model is distributed under the **Qwen Research License** (research/non-commercial). See the upstream license in the model repo: https://huggingface.co/ByteDance/Dolphin-v2/blob/main/LICENSE.
