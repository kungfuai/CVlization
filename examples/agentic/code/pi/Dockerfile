# Extends upstream oh-my-pi/pi:dev. The base image is built locally
# from /tmp/cvl/oh-my-pi (cloned + built by build.sh) -- no prebuilt
# image is published by upstream as of 2026-05.
ARG PI_BASE=oh-my-pi/pi:dev
FROM ${PI_BASE}

# pi's LSP tool drives language servers but does NOT bundle their
# binaries -- they must be on $PATH inside the container. Bundle a
# Python LSP (python-lsp-server) and a TypeScript LSP so the LSP
# rename demo works for the two most common smoke languages.
#
# The upstream pi-runtime image is python:3.12-slim + bun, so we get
# both ecosystems for free.
RUN pip install --no-cache-dir 'python-lsp-server[all]==1.13.1' \
    && bun install -g typescript@5.7.3 typescript-language-server@4.4.1 \
    && /opt/bun/bin/pylsp --version || true

# Defaults pi looks up at runtime for OpenAI-compatible providers.
# run.sh overrides via -e if the user sets them.
ENV VLLM_API_KEY=sk-local

# Inherit upstream ENTRYPOINT [/usr/bin/tini, --, /usr/local/bin/omp]
# and CMD [--help] -- our run.sh overrides CMD with the user's args.
