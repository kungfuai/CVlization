docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/dit:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	dit \
	accelerate launch --mixed_precision fp16 train.py \
	--model "DiT-XL/2" --feature-path features \
	--global-batch-size 2

	# python train.py
