docker run --shm-size 16G --runtime nvidia \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	mmdet_ps \
	python -m examples.panoptic_segmentation.mmdet.train $@
