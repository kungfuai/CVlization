FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install flash_attn from pre-built wheel (much faster than compiling)
# Source: https://flashattn.dev/
RUN pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

# Install Python dependencies
RUN pip install --no-cache-dir \
    diffusers==0.34.0 \
    transformers==4.52.0 \
    accelerate==1.10.0 \
    omegaconf==2.3.0 \
    easydict==1.13 \
    ftfy==6.3.1 \
    librosa==0.11.0 \
    moviepy==2.1.2 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    ffmpeg-python==0.2.0 \
    opencv-python==4.11.0.86 \
    insightface==0.7.3 \
    onnxruntime-gpu==1.22.0 \
    audio-separator==0.30.2 \
    decord==0.6.0 \
    einops \
    huggingface_hub \
    gradio==5.42.0 \
    tqdm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/local/vendor

# Default command
CMD ["python", "--version"]
