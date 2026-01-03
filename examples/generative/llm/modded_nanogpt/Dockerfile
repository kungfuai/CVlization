FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

WORKDIR /workspace

# Copy requirements first for better Docker layer caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Additional dependencies
RUN pip install wandb==0.19.1

# Copy the rest of the code
COPY . /workspace/

# Set environment variables for multi-GPU training
ENV PYTHONUNBUFFERED=1
ENV NCCL_P2P_DISABLE=1
# Don't set CUDA_VISIBLE_DEVICES here - let train.sh control it

CMD ["bash"]
ENTRYPOINT []
