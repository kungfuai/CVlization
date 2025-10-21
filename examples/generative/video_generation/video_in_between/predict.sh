CHECKPOINT_DIR=svd_reverse_motion_with_attnflip
MODEL_NAME=stabilityai/stable-video-diffusion-img2vid-xt
OUT_DIR=results
noise_injection_steps=5
noise_injection_ratio=0.5
out_fn=$OUT_DIR/001.gif

docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/video_in_between:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e CUDA_VISIBLE_DEVICES='0' \
	video_in_between \
	python keyframe_interpolation.py \
        --frame1_path=examples/example_001/frame1.png \
        --frame2_path=examples/example_001/frame2.png \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --checkpoint_dir=$CHECKPOINT_DIR \
        --noise_injection_steps=$noise_injection_steps \
        --noise_injection_ratio=$noise_injection_ratio \
        --decode_chunk_size=4 \
        --num_inference_steps=50 \
        --out_path=$out_fn