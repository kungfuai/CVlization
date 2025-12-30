FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
RUN pip install uv

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

WORKDIR /workspace