# syntax=docker/dockerfile:1.2
FROM python:3.8-slim
# Use the following base image if you need gpu.
# FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

WORKDIR /workspace

# Allows python to stream logs rather than buffer them for output.
ENV PYTHONUNBUFFERED=1

# The official Debian/Ubuntu Docker Image automatically removes the cache by default!
# Removing the docker-clean file manages that issue.
RUN rm -rf /etc/apt/apt.conf.d/docker-clean

RUN --mount=type=cache,mode=0777,target=/var/cache/apt apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Install pip packages
COPY requirements_dev.txt .
RUN --mount=type=cache,mode=0777,target=/root/.cache pip install --upgrade pip \
    && pip install -r requirements_dev.txt
COPY requirements.txt .
RUN --mount=type=cache,mode=0777,target=/root/.cache pip install --upgrade pip \
    && pip install -r requirements.txt

# Switch to non-root user
RUN useradd -m appuser && chown -R appuser /workspace
USER appuser

# Copy project files
COPY . .
