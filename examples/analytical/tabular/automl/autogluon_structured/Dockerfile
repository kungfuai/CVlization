FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    autogluon.tabular[all]==1.1.0 \
    pandas==2.2.2 \
    pyarrow==17.0.0 \
    joblib==1.4.2

CMD ["bash"]
