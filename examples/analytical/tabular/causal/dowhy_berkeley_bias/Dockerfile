FROM python:3.11-slim

WORKDIR /workspace

RUN mkdir -p /tmp/apt-cache/partial \
    && apt-get update \
    && apt-get -o Dir::Cache::Archives=/tmp/apt-cache install -y --no-install-recommends graphviz gcc g++ gfortran \
    && rm -rf /var/lib/apt/lists/* /tmp/apt-cache

RUN pip install --no-cache-dir \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    dowhy==0.13 \
    pydot==1.4.2

CMD ["bash"]
