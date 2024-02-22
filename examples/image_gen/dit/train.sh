docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/dit:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	dit \
	accelerate launch --mixed_precision fp16 train.py \
	--model "DiT-XL/2" --feature-path features \
	--global-batch-size 128 # 128 works on RTX3090 (24GB), 256

	# python train.py
