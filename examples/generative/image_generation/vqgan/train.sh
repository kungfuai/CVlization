docker run --runtime nvidia \
	-v $(pwd)/examples/image_gen/vqgan:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	vqgan \
	$@
