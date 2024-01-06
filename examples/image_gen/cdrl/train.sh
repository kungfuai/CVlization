image=cvlization-diffuser-gpu
module=examples.image_gen.cdrl.train

docker run --runtime nvidia --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    -v $HOME/.netrc:/root/.netrc \
    $image \
    python3 -m $module $@