#!/bin/bash

# Example script to run VACE (Video Animation Control Extension) video generation
# This calls the predict.py script with sample arguments

# docker run --shm-size 16G --runtime nvidia -it \
# 	-v $(pwd)/examples/video_gen/vace/:/workspace \
# 	-v $(pwd)/data/container_cache:/root/.cache \
#     -e CUDA_VISIBLE_DEVICES='0' \
# 	vace \
#     python vace/vace_pipeline.py --base wan \
# 		--task frameref \
# 		--mode firstframe \
# 		--image "assets/images/drone0.png" \
# 		--prompt "a guy happily explains he likes the drone"

docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/vace/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	vace \
    python vace/vace_pipeline.py --base wan \
		--task image_reference \
		--mode salientmasktrack \
		--image "assets/images/drone0.png" \
		--prompt "a guy happily explains he likes the drone"