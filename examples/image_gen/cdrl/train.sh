image=cvlization-cdrl
# module=examples.image_gen.cdrl.train
module=examples.image_gen.cdrl.train_discrete_time_t6

docker run --runtime nvidia --gpus '"device=1"' --rm \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    -v $HOME/.netrc:/root/.netrc \
    $image \
    python3 -m $module $@
