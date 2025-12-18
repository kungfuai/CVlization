FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Limit parallel jobs for wheel builds (e.g., dlib)
ENV MAX_JOBS=4

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone IMTalker repository
RUN git clone https://github.com/bigai-nlco/IMTalker.git /workspace/IMTalker

WORKDIR /workspace/IMTalker

# Install Python dependencies
# Note: face_alignment requires dlib which needs cmake and takes time to build
# Pin numpy<2 for opencv-python compatibility
RUN pip install --no-cache-dir \
    "numpy<2" \
    pyyaml \
    opencv-python==4.10.0.82 \
    pandas \
    tqdm \
    matplotlib \
    flow-vis \
    librosa \
    tensorboard \
    pytorch_lightning \
    transformers==4.30.2 \
    albucore==0.0.16 \
    torchdiffeq==0.2.5 \
    timm==1.0.9 \
    face_alignment==1.4.1 \
    av==12.0.0 \
    gradio \
    huggingface_hub

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/IMTalker

# Default command
CMD ["python", "--version"]
