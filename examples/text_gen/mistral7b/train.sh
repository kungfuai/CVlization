docker run --runtime nvidia -it \
	-v $(pwd)/examples/text_gen/mistral7b:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	mistral \
	python3 train.py
