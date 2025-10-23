docker run --runtime nvidia \
	-v $(pwd)/examples/image_gen/cfm:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-e WANDB_API_KEY=$WANDB_API_KEY \
	-e WANDB_PROJECT=$WANDB_PROJECT \
	nano \
	python train_cifar10.py $@
