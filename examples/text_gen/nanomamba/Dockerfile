# mamba_ssm requires nvcc
# UserWarning: mamba_ssm was requested, but nvcc was not found.
#
# If you're installing within a container from
# https://hub.docker.com/r/pytorch/pytorch, only images
# whose names contain 'devel' will provide nvcc.

# Error: causal_conv1d undefined symbol
# Related issue: https://github.com/state-spaces/mamba/issues/55#issuecomment-1858638484
# FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

WORKDIR /workspace
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install tiktoken==0.5.2 mamba-ssm==1.0.1 causal-conv1d==1.0.1
# causal-conv1d==1.0.1