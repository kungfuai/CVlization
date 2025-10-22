docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	torchvision_od \
	python -m examples.object_detection.torchvision.train $@
