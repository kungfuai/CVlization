FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libopenblas-dev \
        liblapack-dev \
        libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir pip setuptools wheel

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.4.1"

RUN pip install --no-cache-dir \
        "uni2ts==2.0.0" \
        "gluonts==0.14.4" \
        "matplotlib==3.8.4" \
        "pandas==2.1.4" \
        "plotly==5.22.0" \
        "numpy==1.26.4" \
        "lightning==2.5.5" \
        "huggingface_hub==0.23.5"

CMD ["bash"]
