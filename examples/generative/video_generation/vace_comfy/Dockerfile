FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /workspace