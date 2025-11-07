FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN --mount=type=cache,mode=0777,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,mode=0777,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code
COPY forecast.py .

CMD ["python", "forecast.py", "--help"]
