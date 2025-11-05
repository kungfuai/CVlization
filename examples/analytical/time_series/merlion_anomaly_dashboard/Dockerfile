FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8050 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir pip setuptools wheel \
    && pip install --no-cache-dir \
        "gunicorn==21.2.0" \
        "dash[diskcache]==2.14.2" \
        "salesforce-merlion[dashboard]==2.0.1"

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "merlion.dashboard.server:server"]
