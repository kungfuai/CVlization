# PyTorch 2.9.1 with CUDA 12.8 for Blackwell GPU support
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir decord==0.6.0 einops==0.8.1
RUN pip install --no-cache-dir matplotlib==3.10.0 opencv-python==4.10.0.84
RUN pip install --no-cache-dir onnxruntime-gpu==1.20.1

# Add nvidia libs from pip to LD_LIBRARY_PATH for onnxruntime-gpu to find cuDNN
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cusolver/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace