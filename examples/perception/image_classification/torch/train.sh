docker run --shm-size 16G --runtime nvidia \
    -v $(pwd):/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    img_classify_torch \
    python -m examples.image_classification.torch.train --net resnet18