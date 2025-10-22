# TODO: simplify the mounts. Only mount $(pwd) and container cache.
# PYTHONPATH=/workspace/cvlization/examples/video_gen/minisora python my_script.py
docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/minisora:/workspace \
    -v $(pwd)/cvlization:/workspace/cvlization \
    -v $(pwd)/data:/workspace/data \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e CUDA_VISIBLE_DEVICES='0' \
	minisora \
	$@

# -v $(pwd)/cvlization/wandb:/workspace/wandb \