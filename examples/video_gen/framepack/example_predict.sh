docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/framepack/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	framepack \
    python predict.py --mode i2v --input_image data/1.jpg --prompt "A character doing some simple body movements" --total_seconds 10.0 --seed 42 --steps 30 --output_dir ./data/

# Use the following for video extension (conservative approach)
# python predict.py --mode extend --input_video data/1.mp4 --prompt "continues dancing" --extend_seconds 3 --max_context_frames 9 --output_dir ./data/