FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

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
WORKDIR /workspace

# Copy nanochat source code
COPY nanochat /workspace/nanochat

# Install nanochat dependencies using uv
WORKDIR /workspace/nanochat
RUN uv sync

# Default command
CMD ["bash"]
