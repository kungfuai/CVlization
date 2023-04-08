image=cvlization-torch-gpu
module=examples.image_gen.uva_energy.train
docker run --runtime nvidia --rm \
    -v $(pwd):/workspace \
    $image \
    python -m $module