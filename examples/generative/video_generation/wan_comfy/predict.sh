docker run --shm-size 16G --runtime nvidia \
	-v $(pwd)/examples/video_gen/wan_comfy/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-v $(pwd)/examples/video_gen/animate_x/data/images:/workspace/examples/video_gen/animate_x/data/images \
	-v $(pwd)/examples/video_gen/mimic_motion/example_data:/workspace/examples/video_gen/mimic_motion/example_data \
    -e CUDA_VISIBLE_DEVICES='0' \
	wan_comfy \
    python predict.py "$@"
	
	