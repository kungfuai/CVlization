FROM node:20-slim

RUN apt-get update && apt-get install -y python3 --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Pin Claude Code CLI version for reproducibility
RUN npm install -g @anthropic-ai/claude-code@2.1.63

# node:20-slim ships with a non-root 'node' user (uid 1000)
# --dangerously-skip-permissions requires non-root
USER node

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1
