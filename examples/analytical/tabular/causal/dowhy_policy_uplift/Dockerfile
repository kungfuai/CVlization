FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    dowhy==0.10.1 \
    econml==0.15.1 \
    pyarrow==17.0.0 \
    requests==2.32.3 \
    joblib==1.4.2

CMD ["bash"]
