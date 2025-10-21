docker run --runtime nvidia -it \
	-v $(pwd)/examples/text_gen/nanogpt:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-v $(pwd)/cvlization:/workspace/cvlization \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	nanogpt \
	python train.py config/train_shakespeare_char.py $@
