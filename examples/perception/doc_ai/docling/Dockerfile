FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install docling and docling-serve
RUN pip install --no-cache-dir \
    docling \
    docling-ibm-models \
    fastapi \
    uvicorn \
    python-multipart

# Create directories
RUN mkdir -p /app/inputs /app/outputs

# Copy prediction script
COPY predict.py /app/

# Expose port for API (optional, if you want to run as a service later)
EXPOSE 8000

# Default command runs the predict script
CMD ["python", "predict.py"]
