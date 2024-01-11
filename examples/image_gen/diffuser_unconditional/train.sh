image=cvlization-diffuser-unconditional
module=examples.image_gen.diffuser_unconditional.train
docker run -it --runtime nvidia --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    -v $HOME/.netrc:/root/.netrc \
    $image \
    python -m $module $@