FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel



# The original conda environment.yml file is as follows:
# - torchvision=0.15.2
#   - numpy=1.23.1
#   - pip:
#     - diffusers==0.27.0
#     - albumentations==0.4.3
#     - opencv-python==4.6.0.66
#     - pudb==2019.2
#     - imageio==2.9.0
#     - imageio-ffmpeg==0.4.2 
#     - omegaconf==2.1.1
#     - test-tube>=0.7.5
#     - einops==0.3.0
#     - torch-fidelity==0.3.0
#     - torchmetrics==0.11.0
#     - transformers==4.36.0
#     - webdataset==0.2.5
#     - open-clip-torch==2.7.0
#     - invisible-watermark>=0.1.5
#     - accelerate==0.25.0
#     - xformers==0.0.23
#     - peft==0.7.0
#     - torch-ema==0.3
#     - moviepy
#     - tensorboard
#     - Jinja2
#     - ftfy
#     - datasets
#     - wandb
#     - pytorch-fid
#     - notebook
#     - matplotlib
#     - kornia==0.7.2
#     - -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
#     - -e git+https://github.com/openai/CLIP.git@main#egg=clip
#     - -e git+https://github.com/Stability-AI/stablediffusion.git@main#egg=stable-diffusion


RUN pip --verbose install transformers==4.36.* torchvision==0.15.2 numpy==1.23.1 diffusers==0.27.0  \
    imageio-ffmpeg==0.4.2 imageio==2.9.0 omegaconf==2.1.1 test-tube>=0.7.5 einops==0.3.0 \
    torch-fidelity==0.3.0 torchmetrics==0.11.0 webdataset==0.2.5 open-clip-torch==2.7.0 invisible-watermark>=0.1.5
RUN pip install accelerate==0.34.* xformers==0.0.27.* peft==0.12.0 torch-ema==0.3 \
    pytorch-fid notebook matplotlib kornia==0.7.2
RUN pip install pudb==2019.2 moviepy tensorboard Jinja2 ftfy datasets wandb

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    git+https://github.com/openai/CLIP.git@main#egg=clip \
    git+https://github.com/Stability-AI/stablediffusion.git@main#egg=stable-diffusion \
    gdown