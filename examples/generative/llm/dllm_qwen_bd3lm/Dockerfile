FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone and install dllm
RUN git clone https://github.com/ZHZisZZ/dllm.git /opt/dllm && \
    cd /opt/dllm && \
    pip install --no-cache-dir -e .

# Copy and install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "predict.py"]
