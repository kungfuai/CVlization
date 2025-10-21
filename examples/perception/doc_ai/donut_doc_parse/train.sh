docker run --shm-size 16G --runtime nvidia -it \
    -v $(pwd):/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    doc_parse \
    python -m examples.doc_ai.huggingface.donut.doc_parse.train --net donut