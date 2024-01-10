image=cvlization-cdrl
# module=examples.image_gen.cdrl.train
module=examples.image_gen.cdrl.train_discrete_time_t6

docker run -it --runtime nvidia --gpus '"device=0"' --rm \
    --shm-size 30g \
    -v $(pwd):/workspace \
    -v ./data/container_cache:/root/.cache \
    -v $HOME/.netrc:/root/.netrc \
    $image \
    python3 -m $module $@
