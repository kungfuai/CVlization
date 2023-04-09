image=cvlization-torch-gpu
module=examples.image_gen.uva_energy.train
docker run --runtime nvidia --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    $image \
    python -m $module $@