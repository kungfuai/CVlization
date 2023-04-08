# --gpus all

# ./bin/run_example.sh examples.doc_ai.huggingface.donut.doc_type.train

# docker-compose run --rm app python -m $@

# docker-compose run --rm torch python -m cvlization.torch.training_pipeline.image_gen.ebm.uva_energy.pipeline

# docker-compose run --rm diffuser \
#     python3 -m cvlization.torch.training_pipeline.image_gen.diffuser_unconditional.example \
#     --dataset_name="huggan/flowers-102-categories" \
#     --resolution=64 --center_crop --random_flip \
#     --output_dir="ddpm-ema-flowers-64" \
#     --train_batch_size=16 \
#     --num_epochs=100 \
#     --gradient_accumulation_steps=1 \
#     --use_ema \
#     --learning_rate=1e-4 \
#     --lr_warmup_steps=500 \
#     --mixed_precision=no

docker-compose run --rm diffuser \
    python3 -m cvlization.torch.training_pipeline.image_gen.diffuser_unconditional.pipeline \
    --dataset_name="huggan/flowers-102-categories" \
    --resolution=64 --center_crop --random_flip \
    --output_dir="logs/ddpm-ema-flowers-64" \
    --train_batch_size=16 \
    --num_epochs=100 \
    --gradient_accumulation_steps=1 \
    --use_ema \
    --learning_rate=1e-4 \
    --lr_warmup_steps=500 \
    --mixed_precision=no