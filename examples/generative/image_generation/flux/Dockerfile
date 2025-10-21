FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python-is-python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

ADD requirements.txt .
RUN pip install -r requirements.txt

