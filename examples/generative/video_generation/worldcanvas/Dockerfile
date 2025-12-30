FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vendor/WorldCanvas /workspace/WorldCanvas
RUN pip install -e /workspace/WorldCanvas

COPY predict.py /workspace/predict.py

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

CMD ["python3", "predict.py", "--help"]
