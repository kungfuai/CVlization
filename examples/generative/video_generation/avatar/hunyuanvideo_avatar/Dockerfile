FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar /workspace/HunyuanVideo-Avatar

WORKDIR /workspace/HunyuanVideo-Avatar
RUN python - <<'PY'
from pathlib import Path

path = Path('/workspace/HunyuanVideo-Avatar/hymm_sp/text_encoder/__init__.py')
text = path.read_text()
old = '        text_encoder.final_layer_norm = text_encoder.language_model.model.norm\n'
new = (
    '        if hasattr(text_encoder.language_model, "model"):\n'
    '            text_encoder.final_layer_norm = text_encoder.language_model.model.norm\n'
    '        else:\n'
    '            text_encoder.final_layer_norm = text_encoder.language_model.norm\n'
)
if old not in text:
    raise SystemExit('Expected text_encoder.final_layer_norm line not found')
path.write_text(text.replace(old, new, 1))
PY
COPY requirements.txt /workspace/HunyuanVideo-Avatar/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip uninstall -y flash-attn flash_attn || true
RUN python -m pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.0/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-linux_x86_64.whl

COPY predict.py /workspace/predict.py

ENV PYTHONUNBUFFERED=1
ENV MODEL_BASE=/workspace/HunyuanVideo-Avatar/weights

WORKDIR /workspace
CMD ["python3", "/workspace/predict.py", "--help"]
