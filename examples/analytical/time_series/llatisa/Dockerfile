FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,mode=0777,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
CMD ["bash"]
