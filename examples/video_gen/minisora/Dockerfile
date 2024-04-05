# FROM cvlization-torch-gpu
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN pip install scikit-video==1.1.11 av==11.0.* wandb==0.16.* torchvision==0.17.1 lightning==2.2.1
RUN pip install einops==0.7.* moviepy==1.0.*
RUN pip install diffusers==0.26.3 timm==0.9.16
RUN pip install xformers==0.0.25  # needed by ColossalAI open-sora (STDiT denoising model)
RUN pip install mamba-ssm==1.2.* causal-conv1d==1.2.*  # this does not work on *-runtime base images
RUN pip install accelerate==0.28.*
# TODO: move this above
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
# RUN pip install opencv-python==4.9.*