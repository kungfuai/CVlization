FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /workspace
RUN pip install accelerate==0.27.2 transformers==4.37.2 datasets==2.17.* timm==0.9.16 diffusers==0.26.3
# RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
# RUN pip install flash-attn==2.5.5 # requires git, cuda source