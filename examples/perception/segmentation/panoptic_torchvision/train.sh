docker run --shm-size 16G --runtime nvidia \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	torchvision_ps \
	python -m examples.panoptic_segmentation.torchvision.train $@
