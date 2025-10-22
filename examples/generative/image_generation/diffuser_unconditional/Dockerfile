# FROM cvlization-torch-gpu
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
RUN apt-get update -y && apt-get install -y git
# RUN pip install transformers==4.27.4 accelerate==0.18.0 wandb==0.14.2 timm

# TODO: can we use a newer verions of transformers instead of this commit?
# RUN pip install git+https://github.com/huggingface/diffusers.git@1c96f82ed99eede072606e8cc33a975bf4453e90

# The following versions are newer but caused some errors
RUN pip install transformers==4.45.1 accelerate==0.34.2 wandb==0.18.3 timm==1.0.9 diffusers==0.30.3 datasets==3.0.*
RUN pip install tensorboard==2.18.*
RUN pip install clean-fid==0.1.35