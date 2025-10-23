SkyReelsModel='Skywork/SkyReels-V1-Hunyuan-T2V'
docker run --shm-size 16G --runtime nvidia \
	-v $(pwd)/examples/video_gen/skyreals/:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	-e HF_TOKEN=$HF_TOKEN \
	skyreals \
	python video_generate.py \
        --model_id ${SkyReelsModel} \
        --task_type t2v \
        --guidance_scale 6.0 \
        --height 544 \
        --width 960 \
        --num_frames 97 \
        --prompt "FPS-24, A cat wearing sunglasses and working as a lifeguard at a pool" \
        --embedded_guidance_scale 1.0 \
        --quant \
        --offload \
        --high_cpu_memory \
        --parameters_level
	