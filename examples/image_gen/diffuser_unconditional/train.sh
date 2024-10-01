image=cvlization-diffuser-unconditional
module=examples.image_gen.diffuser_unconditional.train
docker run --runtime nvidia --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_PROJECT=diffuser-unconditional \
    $image \
    python -m $module $@