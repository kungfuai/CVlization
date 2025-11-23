FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface

WORKDIR /workspace

# Install python dependencies (inference only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone SAM3 repo at fixed commit for reproducibility
RUN git clone https://github.com/facebookresearch/sam3 /opt/sam3 && \
    cd /opt/sam3 && \
    git checkout 84cc43bca4347b772f17d1078a1ddb4c054655c2 && \
    pip install --no-cache-dir -e .

# Make SAM3 importable everywhere
ENV PYTHONPATH=/opt/sam3

CMD ["bash"]
