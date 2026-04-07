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

# Pre-install graphify skill to a shared location.
# At runtime the entrypoint copies this into the container user's $HOME.
RUN HOME=/opt/graphify-skill graphify install --platform claude \
 && HOME=/opt/graphify-skill graphify install --platform codex \
 && chmod -R 755 /opt/graphify-skill

# Create the runtime home dir with the .claude subdirectory so Docker can
# bind-mount the credentials file into it even before the entrypoint runs.
RUN mkdir -p /tmp/cvl-home/.claude && chmod -R 777 /tmp/cvl-home

WORKDIR /workspace
CMD ["bash"]
