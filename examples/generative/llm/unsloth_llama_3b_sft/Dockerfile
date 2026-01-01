# Use official unsloth image with Blackwell GPU support (PyTorch 2.9.0, CUDA 12.8)
FROM unsloth/unsloth:2025.12.9-pt2.9.0-cu12.8-trans4.57.1-fixes

# Upgrade unsloth from git to get psutil fix (PR #3780 merged Dec 29, 2025)
# Commit 3455f02 includes the fix for missing psutil import
RUN pip install --no-cache-dir --upgrade --force-reinstall \
    "unsloth @ git+https://github.com/unslothai/unsloth.git@3455f0244e2dbde8dba8230a34ca401266767e75" \
    "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"

# Speed up hub downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=180

WORKDIR /workspace
