docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/dit:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	dit \
	python train.py
