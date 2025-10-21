FROM cvlization-torch-gpu

WORKDIR /workspace
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install tiktoken==0.5.2 einops==0.7.0 wandb==0.16.*