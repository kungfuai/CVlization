docker run --shm-size 16G --runtime nvidia \
	-v $(pwd)/examples/image_gen/flux:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	flux \
	python /workspace/generate.py