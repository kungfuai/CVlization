# LTX2 Run Commands

## Basic usage

```bash
cd examples/generative/video_generation/ltx2
CUDA_VISIBLE_DEVICES=0 ./predict.sh --prompt "A cat sitting on a windowsill"
```

## Using Community LoRAs

LoRAs can be specified by HuggingFace repo ID (auto-downloads) or local path:

```bash
# Inflate effect LoRA - makes objects expand like balloons
./predict.sh \
  --prompt "A rubber duck on a table. Then infl4t3 inflates it, expanding into a giant balloon." \
  --lora kabachuha/ltx2-inflate-it 1.0

# Multiple LoRAs with different strengths
./predict.sh \
  --prompt "..." \
  --lora username/lora1 0.8 \
  --lora /local/path/lora2.safetensors 0.5
```

## Advanced examples

```bash
CUDA_VISIBLE_DEVICES=1 docker run --name ltx2_debug2 --gpus=all --shm-size=16g --workdir /workspace --mount "type=bind,src=$(pwd)/examples/generative/video_generation/ltx2,dst=/workspace/local" --mount "type=bind,src=$(pwd),dst=/cvlization_repo,readonly" --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" --mount "type=bind,src=$(pwd),dst=/mnt/cvl/workspace" --env "CUDA_VISIBLE_DEVICES=1" --env "PYTHONPATH=/cvlization_repo:/workspace/local/vendor" --env "CVL_INPUTS=/mnt/cvl/workspace" --env "CVL_OUTPUTS=/mnt/cvl/workspace" --env "HF_HOME=/root/.cache/huggingface" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" cvlization/ltx2:latest python /workspace/local/predict.py --pipeline two_stage --height 768 --width 1280 --num-frames 241 --num-inference-steps 40 --cfg-guidance-scale 4.0 --enhance-prompt --prompt "Cinematic close-up of a friendly presenter speaking directly to camera for about ten seconds, clear lip movement, natural head and shoulder motion, expressive eyes. Background: a bustling daytime open-air market with colorful produce stalls, shoppers moving, vendors gesturing, sunlight filtering through awnings, soft bokeh, light camera sway. Realistic skin detail, natural lighting, shallow depth of field, documentary realism. Clear, synchronized audio of the presenter speaking." --output outputs/ltx2_avatar_market_10s.mp4
```

```bash
CUDA_VISIBLE_DEVICES=1 docker run --name ltx2_streamer_drone_10s --gpus=all --shm-size=16g --workdir /workspace --mount "type=bind,src=$(pwd)/examples/generative/video_generation/ltx2,dst=/workspace/local" --mount "type=bind,src=$(pwd),dst=/cvlization_repo,readonly" --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" --mount "type=bind,src=$(pwd),dst=/mnt/cvl/workspace" --env "CUDA_VISIBLE_DEVICES=1" --env "PYTHONPATH=/cvlization_repo:/workspace/local/vendor" --env "CVL_INPUTS=/mnt/cvl/workspace" --env "CVL_OUTPUTS=/mnt/cvl/workspace" --env "HF_HOME=/root/.cache/huggingface" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" cvlization/ltx2:latest python /workspace/local/predict.py --pipeline two_stage --height 768 --width 1280 --num-frames 241 --num-inference-steps 40 --cfg-guidance-scale 4.0 --enhance-prompt --prompt "A charismatic livestream host sits at a desk, speaking directly to camera about a compact folding drone resting on the tabletop. The host gestures naturally to highlight features, clear lip movement, friendly tone. Background: a cozy streaming studio with soft LED accent lights, blurred shelves of tech gear, subtle RGB glow, shallow depth of field. The drone’s matte texture and small camera gimbal are visible. Stable framing, crisp facial detail, realistic lighting, documentary realism. Clear, synchronized audio of the streamer presenting the product." --output outputs/ltx2_streamer_drone_10s.mp4
```

```bash
CUDA_VISIBLE_DEVICES=1 docker run --name ltx2_inflate_strawberry_5s_fix --gpus=all --shm-size=16g --workdir /workspace --mount "type=bind,src=$(pwd)/examples/generative/video_generation/ltx2,dst=/workspace/local" --mount "type=bind,src=$(pwd),dst=/cvlization_repo,readonly" --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" --mount "type=bind,src=$(pwd),dst=/mnt/cvl/workspace" --env "CUDA_VISIBLE_DEVICES=1" --env "PYTHONPATH=/cvlization_repo:/workspace/local/vendor" --env "CVL_INPUTS=/mnt/cvl/workspace" --env "CVL_OUTPUTS=/mnt/cvl/workspace" --env "HF_HOME=/root/.cache/huggingface" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" cvlization/ltx2:latest python /workspace/local/predict.py --pipeline two_stage --height 768 --width 1280 --num-frames 121 --num-inference-steps 40 --cfg-guidance-scale 4.0 --enhance-prompt --lora kabachuha/ltx2-inflate-it 1.0 --prompt "A friendly woman presenter faces the camera, holding a fresh strawberry at chest height. She says: today I will show some magic. She gently blows on the strawberry, and then infl4t3 inflates it into a giant, balloon-like fruit with smooth elastic texture and playful bounce. Clear lip movement synced to a female voice, natural head and shoulder motion. Background: bright, clean studio kitchen, crisp detail, balanced exposure, no haze or fog, natural contrast, soft daylight. Realistic lighting, documentary realism." --output outputs/ltx2_inflate_strawberry_5s.mp4
```

```bash
CUDA_VISIBLE_DEVICES=1 docker run --name ltx2_inflate_strawberry_nolora_5s --gpus=all --shm-size=16g --workdir /workspace --mount "type=bind,src=$(pwd)/examples/generative/video_generation/ltx2,dst=/workspace/local" --mount "type=bind,src=$(pwd),dst=/cvlization_repo,readonly" --mount "type=bind,src=${HOME}/.cache/huggingface,dst=/root/.cache/huggingface" --mount "type=bind,src=$(pwd),dst=/mnt/cvl/workspace" --env "CUDA_VISIBLE_DEVICES=1" --env "PYTHONPATH=/cvlization_repo:/workspace/local/vendor" --env "CVL_INPUTS=/mnt/cvl/workspace" --env "CVL_OUTPUTS=/mnt/cvl/workspace" --env "HF_HOME=/root/.cache/huggingface" --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" cvlization/ltx2:latest python /workspace/local/predict.py --pipeline two_stage --height 768 --width 1280 --num-frames 121 --num-inference-steps 40 --cfg-guidance-scale 4.0 --enhance-prompt --prompt "A friendly woman presenter faces the camera, holding a fresh strawberry at chest height. She says: today I will show some magic. She gently blows on the strawberry, and it inflates into a giant, balloon-like fruit with smooth elastic texture and playful bounce. Clear lip movement synced to a female voice, natural head and shoulder motion. Background: bright, clean studio kitchen, balanced exposure, no haze or fog, soft daylight, natural contrast, shallow depth of field, crisp detail. Realistic lighting, documentary realism." --output outputs/ltx2_inflate_strawberry_nolora_5s.mp4
```
