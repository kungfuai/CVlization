FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# Set working directory
WORKDIR /workspace

# Default command
CMD ["python3", "train.py"]
