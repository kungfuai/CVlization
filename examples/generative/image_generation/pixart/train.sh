docker run --shm-size 16G --runtime nvidia \
	-v $(pwd)/examples/image_gen/pixart:/workspace \
    -v $(pwd)/cvlization:/workspace/cvlization \
    -v $(pwd)/data:/workspace/data \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e CUDA_VISIBLE_DEVICES='0' \
	pixart \
	$@