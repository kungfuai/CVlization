##############################
# Builder
##############################
FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System deps for Lean + build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    build-essential cmake pkg-config \
    libgmp-dev libssl-dev libuv1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install elan (Lean toolchain)
ARG LEAN_TOOLCHAIN=leanprover/lean4:4.19.0
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf \
    | bash -s -- -y --default-toolchain ${LEAN_TOOLCHAIN}
ENV PATH="/root/.elan/bin:${PATH}"

WORKDIR /opt
RUN git clone --recurse-submodules https://github.com/Kripner/leantree.git
WORKDIR /opt/leantree

# Python deps + LeanTree install
RUN python -m venv .venv \
    && . .venv/bin/activate \
    && pip install -U pip \
    && make install

# Build lean-repl via lake
RUN cd lean-repl && lake update && lake build

##############################
# Runtime
##############################
FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy elan toolchain and leantree artifacts
COPY --from=builder /root/.elan /root/.elan
ENV PATH="/root/.elan/bin:${PATH}"

WORKDIR /app/leantree
COPY --from=builder /opt/leantree /app/leantree

# Reuse venv from builder
ENV VIRTUAL_ENV=/app/leantree/.venv
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"

EXPOSE 8000
CMD ["leanserver", "--project-path", "/app/leantree/leantree_project", "--repl-exe", "/app/leantree/lean-repl/.lake/build/bin/repl", "--max-processes", "2", "--address", "0.0.0.0", "--port", "8000"]
