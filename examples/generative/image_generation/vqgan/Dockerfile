FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /workspace
RUN pip install "numpy<2" datasets==2.17.* torchvision==0.17.1
RUN pip install "numpy<2" albumentations==1.4.* matplotlib==3.8.*
# RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# RUN pip install flash-attn==2.5.5 # requires git, cuda source