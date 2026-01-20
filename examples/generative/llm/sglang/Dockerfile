FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        numactl \
        libxcb1 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix CuDNN compatibility issue for VLMs (Conv3d bug in PyTorch 2.9.1 + CuDNN < 9.15)
# See: https://github.com/pytorch/pytorch/issues/168167
RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.16.0.29

# Fix SGLang SM100 detection bug: should be >= 10 not == 10 for Blackwell (SM 120)
RUN sed -i 's/get_device_capability(device)\[0\] == 10/get_device_capability(device)[0] >= 10/' \
    /opt/conda/lib/python3.11/site-packages/sglang/srt/utils/common.py

COPY . .

CMD ["python3", "serve.py"]
