docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	torchvision_is \
	python -m examples.instance_segmentation.torchvision.train $@
