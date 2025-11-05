# Troubleshooting Common Issues

This document lists solutions for the most frequent problems when running CVlization examples.

## Docker Build Failures

- Ensure Docker is running and you have permission to build images.
- Clear build cache with `docker builder prune` if dependency conflicts appear.

## GPU Not Detected

- Confirm the NVIDIA Container Toolkit is installed.
- Run `nvidia-smi` on the host to verify drivers.
- Set `--gpus=all` when launching containers.

## Large Model Downloads

- All examples share `~/.cache/cvlization` for models.
- Mount this directory into containers to avoid re-downloading.

