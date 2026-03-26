FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# System dependencies
# ffmpeg + libsm6/libxext6: required by opencv
# imagemagick: used by SMT's SynthGenerator for rendering
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Clone SMT repository — provides smt_model, data_augmentation, utils, eval_functions
RUN git clone https://github.com/antoniorv6/SMT.git /opt/SMT

# Make SMT modules importable
ENV PYTHONPATH=/opt/SMT

# Install uv for fast package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
# Note: torch and torchvision are already in the base image
COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

CMD ["python3", "predict.py"]
