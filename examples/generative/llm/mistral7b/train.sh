docker run --runtime nvidia \
	-v $(pwd)/examples/text_gen/mistral7b:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e HF_TOKEN=$HF_TOKEN \
	mistral \
	python3 train.py
