docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/videogpt:/workspace \
    -v $(pwd)/cvlization:/workspace/cvlization \
    -v $(pwd)/cvlization/wandb:/workspace/wandb \
    -v $(pwd)/data:/workspace/data \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e CUDA_VISIBLE_DEVICES='0' \
	videogpt \
	$@
