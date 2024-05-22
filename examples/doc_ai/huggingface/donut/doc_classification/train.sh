docker run --shm-size 16G --runtime nvidia -it \
    -v $(pwd):/workspace \
    -v $(pwd)/data/container_cache:/root/.cache \
    doc_classification \
    python -m examples.doc_ai.huggingface.donut.doc_classification.train --net donut