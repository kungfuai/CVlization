docker run --runtime nvidia \
	-v $(pwd):/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e PYTHONPATH=.:examples/text_gen/nanomamba \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	nanomamba \
	python examples/text_gen/nanomamba/train.py $@

	# -v $(pwd)/examples/text_gen/nanomamba:/workspace \
