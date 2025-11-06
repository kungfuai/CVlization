FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scipy==1.12.0 \
    lightgbm==4.3.0 \
    scikit-learn==1.4.2 \
    requests==2.32.3

CMD ["bash"]
