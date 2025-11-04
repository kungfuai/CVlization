FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    pyod==2.0.0 \
    imbalanced-learn==0.12.3 \
    requests==2.32.3 \
    joblib==1.4.2

CMD ["bash"]
