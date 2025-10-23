docker run --runtime nvidia \
	-v $(pwd)/examples/text_gen/mixtral8x7b/generate.py:/workspace/generate.py \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e HF_TOKEN=$HF_TOKEN \
	mixtral \
	python3 generate.py
