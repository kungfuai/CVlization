# python main.py --source_video_path 0bfacc_0.mp4 --target_video_path 0bfacc_0-radar.mp4 --device cuda --mode RADAR
docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/sports/soccer_game_visual_tracking:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	soccer_game_visual_tracking \
	python main.py --source_video_path 0bfacc_0.mp4 \
        --target_video_path 0bfacc_0-radar.mp4 --device cuda --mode RADAR