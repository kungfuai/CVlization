SkyReelsModel='Skywork/SkyReels-V1-Hunyuan-I2V'
docker run --shm-size 16G --gpus=all \
	-v $(pwd)/examples/video_gen/skyreals/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	-e HF_TOKEN=$HF_TOKEN \
	skyreals \
	python video_generate.py \
        --model_id ${SkyReelsModel} \
        --task_type i2v \
        --guidance_scale 6.0 \
        --height 544 \
        --width 960 \
        --num_frames 97 \
        --image prompt_img.png \
        --prompt "a cartoon movie where the lady stand still, and then slowly nodded her head and smiled, in subtle motion" \
        --negative_prompt "slideshow-style video, Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" \
        --embedded_guidance_scale 1.0 \
        --quant \
        --offload \
        --high_cpu_memory \
        --parameters_level
	