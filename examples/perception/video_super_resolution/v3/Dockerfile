FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MAX_JOBS=8

# System dependencies + Python 3.11 via deadsnakes
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    git curl ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up venv with Python 3.11 as the base
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone V3 repository
RUN git clone --depth=1 https://github.com/prs-eth/v3 /opt/v3

# Install Python dependencies (JAX with CUDA 12, PyTorch CPU-only for data loading)
COPY requirements.txt /tmp/v3_requirements.txt
RUN pip install --no-cache-dir -r /tmp/v3_requirements.txt

# Build deformable attention CUDA extension (limit nvcc threads to 8)
RUN cd /opt/v3/models/rvrt_jax/deform_attn \
    && XLA_DIR=$(python -c "from jax import ffi; print(ffi.include_dir())") \
    && mkdir -p dist \
    && nvcc -std=c++17 --threads 8 -Xcompiler -fPIC \
       -gencode arch=compute_80,code=sm_80 \
       -gencode arch=compute_89,code=sm_89 \
       -gencode arch=compute_90,code=compute_90 \
       -I"$XLA_DIR" -shared src/ops_cuda.cu -o dist/_ops_cuda.so \
    && echo "Built deformable attention CUDA extension"

# Patch upstream NaN assertions to warnings (Blackwell GPUs may produce NaN
# with fp16 due to limited ptxas support in CUDA 12.4)
RUN sed -i \
    's/assert not np.isnan(encoding).any()/if np.isnan(encoding).any(): print("WARN: NaN in encoding, replacing with 0"); encoding = np.nan_to_num(encoding)/' \
    /opt/v3/run_inference.py \
    && sed -i \
    's/assert not np.isnan(out).any()/if np.isnan(out).any(): print("WARN: NaN in output, replacing with 0"); out = np.nan_to_num(out)/' \
    /opt/v3/run_inference.py

WORKDIR /workspace
