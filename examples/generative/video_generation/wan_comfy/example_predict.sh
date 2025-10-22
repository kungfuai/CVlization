# Run this at the root directory of CVlization.
docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/wan_comfy/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	-v $(pwd)/examples/video_gen/animate_x/data/images:/workspace/examples/video_gen/animate_x/data/images \
	-v $(pwd)/examples/video_gen/mimic_motion/example_data:/workspace/examples/video_gen/mimic_motion/example_data \
    -e CUDA_VISIBLE_DEVICES='0' \
	wan_comfy \
    python predict.py -p "a beautiful girl" \
        -n "ugly, deformed, bad anatomy, bad hands, text, error, missing fingers, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad art, bad composition, distorted face, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature" \
        -i examples/video_gen/animate_x/data/images/1.jpg -o output