image=cvlization-mdsm
module=examples.image_gen.mdsm.train

docker run -it --runtime nvidia --gpus '"device=1"' --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    -v $HOME/.netrc:/root/.netrc \
    $image \
    python -m $module $@

    # nvidia-smi