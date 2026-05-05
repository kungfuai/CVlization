FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone and install dllm (for BD3LM inference sampling)
# Patch: guard optional imports (lm_eval, trl) that are not needed for
# training or inference but cause ImportError if absent.
RUN git clone https://github.com/ZHZisZZ/dllm.git /opt/dllm && \
    cd /opt/dllm && \
    printf '%s\n' \
      'from . import core, utils' \
      'for _mod in ("data", "pipelines"):' \
      '    try:' \
      '        __import__(f"dllm.{_mod}", fromlist=[_mod])' \
      '    except ImportError:' \
      '        pass' \
      > dllm/__init__.py && \
    printf '%s\n' \
      'from . import samplers, schedulers, trainers' \
      'try:' \
      '    from . import eval' \
      'except ImportError:' \
      '    pass' \
      > dllm/core/__init__.py && \
    pip install --no-cache-dir -e .

# Copy and install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "train.py", "--help"]
