image=cvlization-ebm
module=examples.image_gen.uva_energy.train

docker run --runtime nvidia --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    -v $HOME/.netrc:/root/.netrc \
    $image \
    python -m $module $@