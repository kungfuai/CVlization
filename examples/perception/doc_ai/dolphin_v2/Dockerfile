FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# Speed up hub downloads and keep installs minimal
ENV HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=180

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "transformers==4.57.3" \
      "hf-transfer==0.1.9" \
      "accelerate==1.12.0" \
      "timm==1.0.22" \
      "sentencepiece==0.2.1" \
      "einops==0.8.1" \
      "pillow==10.4.0"

# Copy inference script
COPY predict.py /workspace/

CMD ["python", "predict.py"]
