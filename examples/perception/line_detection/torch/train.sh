docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	line_detection_torch \
	python -m examples.line_detection.torch.train $@
