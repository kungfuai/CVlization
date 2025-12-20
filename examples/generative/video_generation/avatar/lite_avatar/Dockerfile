FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        unzip \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

RUN python - <<'PY'
from pathlib import Path
path = Path("/opt/conda/lib/python3.11/site-packages/modelscope/utils/ast_utils.py")
text = path.read_text()
if "getattr(node, field, None)" not in text:
    text = text.replace("getattr(node, field)", "getattr(node, field, None)")
    path.write_text(text)
PY

COPY . /workspace
RUN chmod +x download_model.sh

ENV PYTHONPATH=/workspace

CMD ["python", "predict.py"]
