FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python-is-python3 \
    python3-pip \
    python3.10-venv \
    libgl1 \
    libgl1-mesa-glx \ 
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt .

RUN pip install -r requirements.txt