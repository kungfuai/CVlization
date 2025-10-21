INPUT=0bfacc_0.mp4
OUTPUT=radar-0bfacc_0.mp4

docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/sports/soccer_game_visual_tracking:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	soccer_game_visual_tracking \
	python main.py --source_video_path $INPUT \
        --target_video_path $OUTPUT --device cuda --mode RADAR