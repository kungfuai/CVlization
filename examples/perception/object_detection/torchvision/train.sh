docker run --shm-size 16G --runtime nvidia \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	torchvision_od \
	python -m examples.instance_segmentation.torchvision.train $@
