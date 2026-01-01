FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Speed up hub downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=180

# Install Python dependencies (from doclayout-yolo pyproject.toml)
# Note: pillow, pyyaml, requests, tqdm, psutil are in base image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "matplotlib==3.10.8" \
      "opencv-python==4.12.0.88" \
      "scipy==1.16.3" \
      "py-cpuinfo==9.0.0" \
      "thop==0.1.1.post2209072238" \
      "pandas==2.3.3" \
      "seaborn==0.13.2" \
      "albumentations==2.0.8" \
      "huggingface_hub==1.2.3" \
      "hf-transfer==0.1.9"

# Copy source code and inference script
COPY doclayout_yolo /workspace/doclayout_yolo
COPY predict.py /workspace/

ENV PYTHONPATH=/workspace

CMD ["python", "predict.py"]
