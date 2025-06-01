docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/vace_comfy/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -v $(pwd)/examples/video_gen/animate_x/data/images:/workspace/examples/video_gen/animate_x/data/images \
    -e CUDA_VISIBLE_DEVICES='0' \
	vace_comfy \
    python predict.py --prompt "A beautiful woman dancing gracefully in a garden, her flowing dress moving in the wind, cinematic lighting, high quality animation" \
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
	
	