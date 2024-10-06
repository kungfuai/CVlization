docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/cfm:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	cfm \
	python train_cifar10.py

	# python train.py
