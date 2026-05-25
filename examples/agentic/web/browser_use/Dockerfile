FROM python:3.12-slim

# Tell browser-use it's in a container -- this flips BrowserProfile defaults
# (chromium_sandbox=False, IN_DOCKER chrome args enabled).
ENV BROWSER_USE_IN_DOCKER=1 \
    IN_DOCKER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps that Playwright's `install --with-deps chromium` needs.
# We could let `playwright install --with-deps` apt-get them itself, but
# pre-installing keeps the Dockerfile layer cache stable across pip-install
# layer churn.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
      tini \
    && rm -rf /var/lib/apt/lists/*

# Pin browser-use version (upstream releases multiple times per week).
ARG BROWSER_USE_VERSION=0.12.8
RUN pip install --no-cache-dir \
      "browser-use==${BROWSER_USE_VERSION}" \
      "playwright>=1.50" \
      "python-dotenv>=1.0"

# Install Chromium + its system dependencies (~300 MB + libs).
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY agent.py /workspace/agent.py

ENTRYPOINT ["tini", "--", "python", "/workspace/agent.py"]
# No CMD -- with no docker args, agent.py runs with no args and uses
# its default task (read example.com's H1). Add args to override.
