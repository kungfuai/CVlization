FROM python:3.10-slim

WORKDIR /workspace

RUN mkdir -p /tmp/apt-cache/partial \
    && apt-get update \
    && apt-get -o Dir::Cache::Archives=/tmp/apt-cache install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        python3-dev \
        libopenblas-dev \
        liblapack-dev \
        cmake \
        curl \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-cache

RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.9.3 \
    pandas==1.5.3 \
    scikit-learn==1.1.3 \
    joblib==1.4.2 \
    shap==0.38.1 \
    ipython==8.18.1 \
    econml==0.9.2

CMD ["bash"]
