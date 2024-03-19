docker run --runtime nvidia -it \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e PYTHONPATH=.:examples/text_gen/nanomamba \
	nanomamba \
	python examples/text_gen/nanomamba/train.py

	# -v $(pwd)/examples/text_gen/nanomamba:/workspace \
