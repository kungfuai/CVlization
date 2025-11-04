FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    lightgbm==4.3.0 \
    pyarrow==17.0.0 \
    requests==2.32.3 \
    joblib==1.4.2 \
    gbt==0.3

CMD ["bash"]
