#!/bin/bash

# Example script to run VACE (Video Animation Control Extension) video generation
# This calls the predict.py script with sample arguments

python examples/video_gen/vace_comfy/predict.py \
    --prompt "A beautiful woman dancing gracefully in a garden, her flowing dress moving in the wind, cinematic lighting, high quality animation" \
    --negative-prompt "static, blurry, low quality, distorted, ugly, deformed" \
    --input-images "examples/video_gen/animate_x/data/images/1.jpg" \
    --output-dir "output/vace_example" \
    --fps 16 \
    --cfg 4.0 \
    --steps 20 \
    --width 720 \
    --height 720 \
    --length 49 \
    --seed 42 \
    --model-shift 8.0