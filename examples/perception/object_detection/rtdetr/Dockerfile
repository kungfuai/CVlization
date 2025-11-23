FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Lightweight runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python dependencies (inference only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /workspace

CMD ["python3", "predict.py"]
