docker run --shm-size 16G --runtime nvidia \
	-v $(pwd)/examples/video_gen/vace/:/workspace \
	-v $(pwd)/data/container_cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B:Wan2.1-T2V-1.3B \
	-v $(pwd)/data/container_cache/huggingface/hub/bytedance-research/Phantom:Phantom-Wan-Models \
    -e CUDA_VISIBLE_DEVICES='0' \
	phantom \
    python phantom/predict.py "$@"