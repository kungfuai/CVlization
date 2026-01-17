FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust/Cargo for rustbpe tokenizer
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace/nanochat

# Copy only dependency metadata (source is mounted at runtime)
COPY nanochat/pyproject.toml nanochat/uv.lock nanochat/README.md ./

# Dependencies will be installed via uv run at runtime
# .venv in the mounted project directory persists to host

# Default command
CMD ["bash"]
