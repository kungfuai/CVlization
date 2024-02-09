FROM cvlization-torch-gpu

WORKDIR /workspace
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install tiktoken==0.5.2