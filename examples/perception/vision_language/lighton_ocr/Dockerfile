FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    python3-dev \
 && rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install vLLM (pinned version + commit wheel index) and helpers
ARG VLLM_COMMIT=e88bdd60d9a25d985168c9f4a60ab10095236d7c
RUN /root/.local/bin/uv pip install --system \
    git+https://github.com/vllm-project/vllm.git@${VLLM_COMMIT} \
    'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' \
    --prerelease=allow

COPY requirements.txt .
RUN /root/.local/bin/uv pip install --system -r requirements.txt

# Provide a minimal pyairports module (outlines dependency sometimes ships a broken wheel)
RUN python - <<'PY'
import os
import sysconfig

site = sysconfig.get_paths()["purelib"]
mod_dir = os.path.join(site, "pyairports")
os.makedirs(mod_dir, exist_ok=True)
with open(os.path.join(mod_dir, "__init__.py"), "w", encoding="utf-8") as f:
    f.write("from .airports import AIRPORT_LIST\n")
with open(os.path.join(mod_dir, "airports.py"), "w", encoding="utf-8") as f:
    f.write("AIRPORT_LIST = []\n")
PY

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_CACHE=/root/.cache/huggingface

WORKDIR /workspace

CMD ["bash"]
