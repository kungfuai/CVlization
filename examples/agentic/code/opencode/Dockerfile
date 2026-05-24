FROM node:22-bookworm-slim

# git/curl: opencode shells out to git for project root + diffs.
# ca-certs: HTTPS to npm registry + any external provider endpoints.
# tini: clean SIGINT handling so Ctrl+C exits the TUI cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

# Pin opencode-ai: upstream ships multiple releases per week (~800 to date),
# so floating @latest would make this preset unreproducible.
ARG OPENCODE_VERSION=1.15.10
RUN npm install -g "opencode-ai@${OPENCODE_VERSION}" "@ai-sdk/openai-compatible"

WORKDIR /workspace

ENV NO_UPDATE_NOTIFIER=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    OPENCODE_BASE_URL=http://localhost:8000/v1 \
    VLLM_API_KEY=sk-local

ENTRYPOINT ["tini", "--"]
CMD ["opencode"]
