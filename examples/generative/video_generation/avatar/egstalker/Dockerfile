# Base image options:
# 1. pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel (current, 17.3GB)
#    - Official PyTorch with CUDA 11.8
#    - Includes all build tools
# 2. gaetanlandreau/pytorch3d (has pytorch3d pre-installed but older versions)
#    - Python 3.8, PyTorch 1.13
#    - Would need PyTorch upgrade for compatibility
# 3. Custom Gaussian Splatting images (rhene/gaussian-splatting, etc.)
#    - Not maintained, unknown compatibility
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Build arguments for CUDA architecture targeting
# CUDA Compute Capability to GPU mapping:
#   6.0 - Tesla P100
#   6.1 - GTX 1080, GTX 1070, Titan Xp
#   7.0 - Tesla V100
#   7.5 - Tesla T4, RTX 2080, Quadro RTX 6000
#   8.0 - A100, A30
#   8.6 - A10, RTX 3090, RTX 3080, A40 (this machine)
#   8.9 - L4, L40, RTX 4090, RTX 4080
#   9.0 - H100, H200
#
# To compile for multiple GPUs: TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"
# To check your GPU: nvidia-smi --query-gpu=name,compute_cap --format=csv
ARG TORCH_CUDA_ARCH_LIST="8.6"
ARG MAX_JOBS=4

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential wget \
    ffmpeg libsm6 libxext6 libglm-dev \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies and utilities (combined for fewer layers)
RUN pip install --no-cache-dir \
    decord==0.6.0 \
    einops==0.8.0 \
    matplotlib==3.9.2 \
    opencv-python==4.10.0.84 \
    imageio==2.36.0 \
    imageio-ffmpeg==0.5.1 \
    numpy==1.23.5 \
    scipy==1.13.1 \
    scikit-image==0.24.0 \
    tqdm==4.66.6 \
    pyyaml==6.0.2 \
    pillow==11.0.0 \
    pandas==2.2.3 \
    pyquaternion==0.9.9 \
    pytorch-msssim==1.0.0

# Install 3DGS and related packages
RUN pip install --no-cache-dir \
    lpips==0.1.4 \
    open3d==0.18.0 \
    plyfile==1.1

# Install face processing and config management
RUN pip install --no-cache-dir \
    face-alignment==1.4.1 \
    mmcv==1.6.0

# Set CUDA compilation options BEFORE compiling pytorch3d
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV MAX_JOBS=${MAX_JOBS}

# Install pytorch3d (will use MAX_JOBS for parallel compilation)
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/pytorch3d.git" || \
    pip install --no-cache-dir pytorch3d

# Install audio processing and DeepSpeech dependencies
RUN pip install --no-cache-dir \
    soundfile==0.12.1 \
    resampy==0.4.3 \
    python-speech-features==0.6 \
    tensorflow==2.13.0 && \
    pip install --no-cache-dir --upgrade typing_extensions

# Install MediaPipe for BFM-free face tracking
# Pin numpy<2 to avoid compatibility issues with PyTorch 2.1.2
# Install librosa and transformers for audio feature extraction
# Install wandb for experiment tracking
RUN pip install --no-cache-dir \
    "numpy<2" \
    mediapipe==0.10.9 \
    librosa==0.10.1 \
    transformers==4.36.2 \
    wandb==0.16.1

WORKDIR /workspace

# Clone EGSTalker and compile custom CUDA extensions
RUN git clone https://github.com/ZhuTianheng/EGSTalker.git /workspace/egstalker && \
    cd /workspace/egstalker && \
    mkdir -p submodules && \
    cd submodules && \
    git clone https://github.com/joungbinlee/custom-bg-depth-diff-gaussian-rasterization.git custom-bg-depth-diff-gaussian-rasterization && \
    git clone https://github.com/camenduru/simple-knn.git simple-knn && \
    cd custom-bg-depth-diff-gaussian-rasterization && \
    pip install --no-cache-dir -e . && \
    cd ../simple-knn && \
    pip install --no-cache-dir -e . && \
    cd /workspace/egstalker/scene/KAN && \
    pip install --no-cache-dir -e . && \
    cd /workspace

# Set PYTHONPATH to include egstalker
ENV PYTHONPATH=/workspace/egstalker:${PYTHONPATH}
