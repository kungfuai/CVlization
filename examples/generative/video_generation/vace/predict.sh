docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/vace/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	vace \
    python vace/vace_pipeline.py "$@"