FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Download Tiny Shakespeare dataset
RUN mkdir -p /opt/data && \
    apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && \
    curl -sL -o /opt/data/tinyshakespeare.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
