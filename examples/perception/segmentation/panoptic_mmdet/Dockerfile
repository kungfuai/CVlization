FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST=12.0
ENV MAX_JOBS=16
ENV CMAKE_BUILD_PARALLEL_LEVEL=16

RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y build-essential git wget unzip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install "setuptools>=80,<81" wheel cython ninja
RUN pip install "mmengine>=0.10.5"
RUN pip install --no-build-isolation "mmcv>=2.0.0rc4,<2.2.0"
RUN pip install "mmdet>=3.3.0"
RUN pip install scikit-image==0.24.0 wandb==0.17.*
RUN pip install git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0
