docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/vqgan:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	vqgan \
	$@
