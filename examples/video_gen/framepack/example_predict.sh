docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/framepack/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	framepack \
    python predict.py --input_image data/1.jpg --prompt "A character doing some simple body movements" --total_seconds 10.0 --seed 42 --steps 30 --output_dir ./data/