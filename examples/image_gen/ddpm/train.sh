docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/ddpm:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-v $(pwd)/data/:/workspace/data \
	ddpm \
	python train.py