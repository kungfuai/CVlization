FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "predict.py"]
