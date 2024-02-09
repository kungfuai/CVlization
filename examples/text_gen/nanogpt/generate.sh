docker run --runtime nvidia -it \
	-v $(pwd)/examples/text_gen/nanogpt:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	nanogpt \
	python sample.py --out_dir=out-shakespeare-char
