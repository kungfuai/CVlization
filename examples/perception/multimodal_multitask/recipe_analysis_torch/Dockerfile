FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "numpy<2" \
        pyarrow==15.0.0 \
        datasets==2.20.0 \
        pillow==10.0.0 \
        pytorch-lightning==2.0.0 \
        torchmetrics==1.0.0 \
        scikit-learn==1.3.0 \
        timm==0.9.16 \
        transformers==4.40.0
