docker run --runtime nvidia -it \
	-v $(pwd)/examples/text_gen/mixtral8x7b/generate.py:/workspace/generate.py \
	-v $(pwd)/data/container_cache:/root/.cache \
	mixtral \
	python3 generate.py