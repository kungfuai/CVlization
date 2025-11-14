FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

WORKDIR /workspace

CMD ["python3", "predict.py"]
