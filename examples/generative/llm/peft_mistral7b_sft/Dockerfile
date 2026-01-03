FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir \
    transformers==4.48.2 \
    bitsandbytes==0.45.5 \
    peft==0.14.0 \
    accelerate==1.2.1 \
    datasets==3.6.0 \
    scipy==1.15.1 \
    sentencepiece==0.2.0 \
    protobuf==5.29.5

WORKDIR /workspace
