docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/phantom/:/workspace \
	-v $(pwd)/data/container_cache/huggingface/hub/Wan-AI/Wan2.1-T2V-1.3B:/workspace/Wan2.1-T2V-1.3B \
	-v $(pwd)/data/container_cache/huggingface/hub/bytedance-research/Phantom:/workspace/Phantom-Wan-Models \
    -e CUDA_VISIBLE_DEVICES='0' \
	phantom \
    python generate.py --task s2v-1.3B \
        --size 832*480 \
        --ckpt_dir ./Wan2.1-T2V-1.3B \
        --phantom_ckpt ./Phantom-Wan-Models/Phantom-Wan-1.3B.pth \
        --ref_image "examples/ref1.png,examples/ref2.png" \
        --ulysses_size 1 \
        --ring_size 1 \
        --prompt "暖阳漫过草地，扎着双马尾、头戴绿色蝴蝶结、身穿浅绿色连衣裙的小女孩蹲在盛开的雏菊旁。她身旁一只棕白相间的狗狗吐着舌头，毛茸茸尾巴欢快摇晃。小女孩笑着举起黄红配色、带有蓝色按钮的玩具相机，将和狗狗的欢乐瞬间定格。"