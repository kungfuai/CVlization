FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        numactl \
        libxcb1 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix CuDNN compatibility issue for VLMs (Conv3d bug in PyTorch 2.9.1 + CuDNN < 9.15)
# See: https://github.com/pytorch/pytorch/issues/168167
RUN pip install --no-cache-dir nvidia-cudnn-cu12==9.16.0.29

# Fix SGLang SM100 detection bug: should be >= 10 not == 10 for Blackwell (SM 120)
RUN sed -i 's/get_device_capability(device)\[0\] == 10/get_device_capability(device)[0] >= 10/' \
    /opt/conda/lib/python3.11/site-packages/sglang/srt/utils/common.py

# Fix SGLang v_head_dim=null bug: HunYuan-A13B config.json sets "v_head_dim": null explicitly,
# causing getattr(..., self.head_dim) to return None (present key beats the default),
# which then crashes get_cell_size_per_token with TypeError: int + NoneType.
# Replace the bare getattr with a None-safe fallback.
RUN python - <<'EOF'
import pathlib
p = pathlib.Path("/opt/conda/lib/python3.11/site-packages/sglang/srt/configs/model_config.py")
old = '        self.v_head_dim = getattr(\n            self.hf_text_config,\n            "v_head_dim",\n            self.head_dim,\n        )'
new = '        _v_head_dim = getattr(self.hf_text_config, "v_head_dim", None)\n        self.v_head_dim = _v_head_dim if _v_head_dim is not None else self.head_dim'
text = p.read_text()
assert old in text, f"v_head_dim patch target not found in {p} -- update patch for new SGLang version"
p.write_text(text.replace(old, new, 1))
print("SGLang v_head_dim null-safety patch applied")
EOF

COPY . .

CMD ["python3", "serve.py"]
