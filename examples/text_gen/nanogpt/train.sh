docker run --runtime nvidia -it \
	-v $(pwd)/examples/text_gen/nanogpt:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	nanogpt \
	python train.py config/train_shakespeare_char.py
