docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/wan/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-v $(pwd)/examples/video_gen/animate_x/data/images:/workspace/examples/video_gen/animate_x/data/images \
    -e CUDA_VISIBLE_DEVICES='0' \
	wan \
    python predict.py
	
	