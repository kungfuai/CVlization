FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install gluonts==0.14.3 pandas==2.3.3 matplotlib==3.10.7 python-dotenv==1.0.0 \
    && pip install git+https://github.com/SalesforceAIResearch/uni2ts.git@1.2.0

COPY . .

CMD ["bash"]
