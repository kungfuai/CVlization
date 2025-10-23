docker run --shm-size 16G --runtime nvidia \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	mmseg_ss \
	python -m examples.semantic_segmentation.mmseg.train $@
