docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	mmpose \
	python -m examples.pose_estimation.mmpose.train $@
