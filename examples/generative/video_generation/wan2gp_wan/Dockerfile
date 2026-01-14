FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel


COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install sageattention==1.0.6

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN pip install flash-attn==2.7.2.post1

EXPOSE 7860
WORKDIR /workspace