FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
RUN apt-get update && apt-get install -y git build-essential python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip install transformers==4.37.2 sentencepiece==0.1.99 bitsandbytes==0.42.0 \
    git+https://github.com/mobiusml/hqq.git@37502bea31f2969c6680c0c4a88ca74b3bb234a5
RUN git clone https://github.com/dvmazur/mixtral-offloading.git --quiet
# RUN git clone https://github.com/dvmazur/mixtral-offloading.git --quiet
# !cd mixtral-offloading && pip install -q -r requirements.txt
