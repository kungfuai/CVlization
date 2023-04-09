FROM cvlization-torch-gpu
RUN pip install transformers==4.27.4 accelerate==0.18.0 wandb==0.14.2
RUN apt-get update -y && apt-get install -y git
RUn pip install git+https://github.com/huggingface/diffusers.git@1c96f82ed99eede072606e8cc33a975bf4453e90
