docker run --runtime nvidia -it \
	-v $(pwd)/examples/image_gen/edm:/workspace \
	-v $(pwd)/data/container_cache:/root/.cache \
	edm2 \
	torchrun --standalone --nproc_per_node=1 train_edm2.py \
    --outdir=training-runs/00000-edm2-img512-xs \
    --data=datasets/img512-sd.zip \
    --preset=edm2-img512-xs \
    --batch-gpu=32

	# python train.py
