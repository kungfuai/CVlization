FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NANOBOT_SKIP_WEBUI_BUILD=1

# System deps: git for pip-install-from-git, adb for Android backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        adb \
    && rm -rf /var/lib/apt/lists/*

# Pin upstream commit for reproducibility
ARG KNOWACT_COMMIT=2d317daa57187b8a90f5b94964a577441aa70e4f
RUN pip install --no-cache-dir \
    "nanobot-ai @ git+https://github.com/HITsz-TMG/KnowAct.git@${KNOWACT_COMMIT}#subdirectory=GUIClaw" \
    pyyaml \
    huggingface_hub

WORKDIR /workspace
COPY predict.py .
