export NUM_GPUS=1
export MODEL_FLAGS="--image_size 256 --mask_ratio 0.30 --decode_layer 4 --model MDTv2_S_2"
export DIFFUSION_FLAGS="--diffusion_steps 1000"
export TRAIN_FLAGS="--batch_size 32"
export DATA_PATH=/dataset/imagenet

docker run --shm-size 16G --runtime nvidia -it \
	-v $(pwd)/examples/video_gen/mdt:/workspace \
    -v $(pwd)/data:/workspace/data \
	-v $(pwd)/data/container_cache:/root/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e CUDA_VISIBLE_DEVICES='0' \
    -e OPENAI_LOGDIR=output_mdtv2_s2 \
	mdt \
    python -m scripts.image_train --batch_size 1 --image_size 256 --mask_ratio 0.30 --decode_layer 4 --model MDTv2_S_2 --diffusion_steps 1000
	# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
    # scripts/image_train.py --data_dir $DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS