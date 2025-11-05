FROM python:3.10-slim

WORKDIR /workspace

RUN mkdir -p /tmp/apt-cache/partial \
    && apt-get update \
    && apt-get -o Dir::Cache::Archives=/tmp/apt-cache install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        curl \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-cache

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.11.4 \
    pandas==2.1.4 \
    scikit-learn==1.6.1 \
    joblib==1.4.2 \
    xgboost==1.7.6 \
    lightgbm==4.1.0 \
    shap==0.41.0 \
    ipython==8.18.1 \
    causalml==0.15.5

CMD ["bash"]
