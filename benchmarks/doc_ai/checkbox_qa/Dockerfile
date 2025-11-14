FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies for PDF processing (pdftoppm/pdfinfo) and curl/git for convenience
RUN --mount=type=cache,mode=0777,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,mode=0777,target=/root/.cache \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy benchmark scripts
COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["bash"]
