docker run --shm-size 16G --runtime nvidia \
	-v $(pwd)/examples/video_gen/mimic_motion/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	-e HF_TOKEN=$HF_TOKEN \
	mimic_motion \
	python predict.py
	