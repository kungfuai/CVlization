FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
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

# Note: SGLang SM100 detection bug (== 10 instead of >= 10) was fixed upstream in v0.5.10.

# Fix SGLang OLMo-3 rope_theta bug (v0.5.10): Olmo3Config absorbs rope_theta as a direct
# attribute, leaving it absent from rope_parameters dict. olmo2.py then crashes with
# KeyError: 'rope_theta' when doing config.rope_parameters["rope_theta"].
# Fix: use .get() with fallback to config.rope_theta.
RUN sed -i \
    's/self\.rope_theta = config\.rope_parameters\["rope_theta"\]/self.rope_theta = config.rope_parameters.get("rope_theta", config.rope_theta)/' \
    /usr/local/lib/python3.12/dist-packages/sglang/srt/models/olmo2.py

# Fix SGLang v_head_dim=null bug: HunYuan-A13B config.json sets "v_head_dim": null explicitly,
# causing getattr(..., self.head_dim) to return None (present key beats the default),
# which then crashes get_cell_size_per_token with TypeError: int + NoneType.
# Replace the bare getattr with a None-safe fallback.
RUN python - <<'EOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/sglang/srt/configs/model_config.py")
old = '        self.v_head_dim = getattr(\n            self.hf_text_config,\n            "v_head_dim",\n            self.head_dim,\n        )'
new = '        _v_head_dim = getattr(self.hf_text_config, "v_head_dim", None)\n        self.v_head_dim = _v_head_dim if _v_head_dim is not None else self.head_dim'
text = p.read_text()
assert old in text, f"v_head_dim patch target not found in {p} -- update patch for new SGLang version"
p.write_text(text.replace(old, new, 1))
print("SGLang v_head_dim null-safety patch applied")
EOF

# Fix Voxtral/MistralCommonBackend tokenizer: sglang calls tokenizer.get_added_vocab()
# unconditionally, but Mistral's MistralCommonBackend doesn't implement it.
# Guard the call with hasattr to skip gracefully for incompatible tokenizers.
RUN sed -i \
    's/if "<|eom_id|>" in tokenizer\.get_added_vocab()/if hasattr(tokenizer, "get_added_vocab") and "<|eom_id|>" in tokenizer.get_added_vocab()/' \
    /usr/local/lib/python3.12/dist-packages/sglang/srt/utils/hf_transformers_utils.py

COPY . .

CMD ["python3", "serve.py"]
