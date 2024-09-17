docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/object_tracking/global_tracking_transformer:/workspace \
    -v $(pwd)/data:/workspace/data \
    -e CUDA_VISIBLE_DEVICES='0' \
	gtr \
	python predict.py data/soccer1_clip1.mp4 data/output/