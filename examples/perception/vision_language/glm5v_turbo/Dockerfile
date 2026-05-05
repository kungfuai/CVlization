FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

WORKDIR /workspace

CMD ["python3", "predict.py"]
