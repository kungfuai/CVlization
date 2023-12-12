FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Install git, g++
RUN apt-get update && apt-get install -y git gcc \
    python3-dev git wget sudo ninja-build \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY examples/instance_segmentation/seem/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/MaureenZOU/detectron2-xyz.git