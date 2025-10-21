docker run --shm-size 16G --runtime nvidia -it --rm \
	-v $(pwd):/workspace \
    -e CUDA_VISIBLE_DEVICES='0' \
    -p 7860:7860 \
	wangp \
    python gradio_server.py --i2v --server-name 0.0.0.0
