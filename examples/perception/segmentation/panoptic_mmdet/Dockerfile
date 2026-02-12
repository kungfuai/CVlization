# FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime  # RuntimeError: CUDA error: no kernel image is available for execution on the device
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime  # not compatible with mmdv-full==1.5.0

WORKDIR /workspace
RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y git wget unzip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 # for imgaug \ 
    && rm -rf /var/lib/apt/lists/*
RUN pip install mmdet==2.25.0 scikit-image==0.19.3 torch==1.11.0
RUN pip install -U mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html # https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html
RUN pip install git+https://github.com/cocodataset/panopticapi.git
RUN pip install yapf==0.40.1  # https://github.com/open-mmlab/mmdetection/issues/10962
RUN pip install wandb==0.16.6
