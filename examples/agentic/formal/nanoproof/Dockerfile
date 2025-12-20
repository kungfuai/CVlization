FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gnupg \
    build-essential \
    ca-certificates \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12 (needed for leantree)
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install pip for python3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install uv for faster Python installs
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

# Copy vendored source and wrapper scripts
COPY nanoproof_repo /workspace/nanoproof_repo
COPY predict.py /workspace/predict.py

# Install torch + deps against Python 3.12
RUN uv pip install --python python3.12 --system --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

RUN uv pip install --python python3.12 --system --no-cache-dir \
    datasets \
    fastapi \
    huggingface_hub \
    psutil \
    "regex<2026" \
    setuptools \
    tiktoken \
    tqdm \
    tokenizers \
    wandb \
    termplotlib \
    PrettyPrintTree==2.0.1 \
    "cmd2<3"

# Install leantree Python package but skip building lean-repl (handled separately in Lean sidecar)
RUN set -e; \
    git clone --depth 1 https://github.com/Kripner/leantree /tmp/leantree; \
    printf '%s\n' \
        "from pathlib import Path" \
        "p = Path('/tmp/leantree/pyproject.toml')" \
        "lines = p.read_text().splitlines()" \
        "out = []" \
        "skip = False" \
        "for line in lines:" \
        "    if line.strip() == '[tool.poetry.build]':" \
        "        skip = True" \
        "        continue" \
        "    if skip and line.startswith('['):" \
        "        skip = False" \
        "    if skip:" \
        "        continue" \
        "    out.append(line)" \
        "p.write_text('\\n'.join(out) + '\\n')" \
        > /tmp/strip_leantree.py; \
    python3.12 /tmp/strip_leantree.py; \
    python3.12 -m pip install --no-cache-dir poetry-core; \
    python3.12 -m pip install --no-cache-dir --no-build-isolation /tmp/leantree; \
    rm -rf /tmp/leantree /tmp/strip_leantree.py

# Expose cache location for assets
ENV NANOPROOF_BASE_DIR=/root/.cache/nanoproof \
    PYTHONPATH=/workspace/nanoproof_repo \
    PYTHON=python3.12

WORKDIR /workspace

CMD ["python3.12", "predict.py", "--help"]
