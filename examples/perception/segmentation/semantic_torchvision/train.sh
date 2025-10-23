docker run --shm-size 16G --runtime nvidia \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	torchvision_ss \
	python -m examples.semantic_segmentation.torchvision.train $@
