FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    accelerate==1.7.0 \
    transformers==4.48.3 \
    datasets==3.6.0 \
    timm==1.0.15 \
    diffusers==0.36.0
