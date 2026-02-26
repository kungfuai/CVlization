FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ARG STABLE_WORLDMODEL_REF=v0.0.1b0

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew2.2 \
    libosmesa6 \
    libxrender1 \
    libxext6 \
    libsm6 \
    swig \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    git clone --depth 1 --branch "${STABLE_WORLDMODEL_REF}" https://github.com/galilai-group/stable-worldmodel.git /opt/stable-worldmodel && \
    python -m pip install --no-cache-dir "/opt/stable-worldmodel[env,train]" && \
    python -m pip install --no-cache-dir "jepa"

COPY . /workspace

ENV PYTHONUNBUFFERED=1
ENV XDG_CACHE_HOME=/cvl-cache
ENV HF_HOME=/cvl-cache/huggingface
ENV STABLEWM_HOME=/cvl-cache/stable-worldmodel
ENV MUJOCO_GL=egl

CMD ["python", "run_eval.py", "--help"]
