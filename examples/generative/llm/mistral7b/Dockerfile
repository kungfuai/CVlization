# FROM huggingface/transformers-pytorch-gpu:4.35.2  # cuda is 11.8 for this one..
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN pip install transformers==4.37.2 bitsandbytes==0.42.0 peft==0.8.2 accelerate==0.26.1 datasets scipy
WORKDIR /workspace
