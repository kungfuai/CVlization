# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libgl1-mesa-glx libglib2.0-0 git gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# RUN pip install torch==1.8.1 torchvision==0.9.1 \
#     opencv-python==4.5.5.64 timm==0.5.4 lvis==0.5.3 scipy imageio-ffmpeg==0.4.7
# RUN pip install git+https://github.com/TAO-Dataset/tao

RUN pip install torchvision==0.11.* \
    opencv-python==4.5.5.64 timm==0.5.4 lvis==0.5.3 scipy imageio-ffmpeg==0.4.7 \
    git+https://github.com/TAO-Dataset/tao

RUN pip install lap==0.4.0
# RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html 
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
