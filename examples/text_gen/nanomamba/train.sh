docker run --runtime nvidia -it \
	-v $(pwd)/examples/text_gen/nanomamba:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	nanomamba \
	python train.py
