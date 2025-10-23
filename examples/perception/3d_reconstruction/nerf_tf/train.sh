docker run --shm-size 16G --runtime nvidia --name nerf --rm \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	nerf_tf \
	python -m examples.nerf.tf.train $@
