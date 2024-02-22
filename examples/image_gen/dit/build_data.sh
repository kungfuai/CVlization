docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/dit:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e "RANK=0" \
    -e "WORLD_SIZE=1" \
    -e "MASTER_ADDR=localhost" \
    -e "MASTER_PORT=29500" \
	dit \
	python extract_features.py
