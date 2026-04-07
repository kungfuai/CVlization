FROM node:20-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl python3 python3-pip ripgrep jq \
    && rm -rf /var/lib/apt/lists/*

# Install coding agent CLIs
RUN npm install -g @anthropic-ai/claude-code @openai/codex

# Install graphify and tree-sitter language bindings
RUN pip3 install --no-cache-dir --break-system-packages \
    graphifyy \
    tree-sitter-python \
    tree-sitter-javascript \
    tree-sitter-typescript

# Pre-install graphify skill for both platforms
RUN graphify install --platform claude \
 && graphify install --platform codex

WORKDIR /workspace
CMD ["bash"]
