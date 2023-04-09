image=cvlization-diffuser-unconditional
module=examples.image_gen.diffuser_unconditional.train
docker run --runtime nvidia --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    $image \
    python -m $module $@