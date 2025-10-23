docker run --runtime nvidia \
	-v $(pwd)/examples/text_gen/nanogpt:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	nanogpt \
	python sample.py --out_dir=logs/nanogpt/batch64_block256
