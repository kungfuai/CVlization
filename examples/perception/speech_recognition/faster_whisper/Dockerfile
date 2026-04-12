FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "predict.py"]
