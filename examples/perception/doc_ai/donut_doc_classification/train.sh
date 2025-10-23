docker run --shm-size 16G --runtime nvidia \
    -v $(pwd):/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    doc_classification \
    python -m examples.doc_ai.huggingface.donut.doc_classification.train --net donut