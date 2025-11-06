FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchtuples==0.2.2 \
    pycox==0.2.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    pyarrow==17.0.0 \
    requests==2.32.3 \
    matplotlib==3.8.4

CMD ["bash"]
