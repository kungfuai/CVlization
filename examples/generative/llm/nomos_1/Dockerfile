FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "predict.py"]
