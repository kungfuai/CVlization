docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/animate_x/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	animate-x \
	python extract_pose.py
	