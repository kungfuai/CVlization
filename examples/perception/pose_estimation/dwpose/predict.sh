docker run --shm-size 16G --runtime nvidia \
	-v $(pwd)/examples/pose_estimation/dwpose:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	dwpose \
	python predict.py