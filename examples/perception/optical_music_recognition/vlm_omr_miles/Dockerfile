# Miles GRPO for VLM OMR (Optical Music Recognition)
# Uses radixark/miles with VLM support (Qwen3-VL/Qwen3.5)

FROM radixark/miles:latest

ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# OMR-specific dependencies
RUN pip install --no-cache-dir pyyaml datasets pillow

WORKDIR /workspace

CMD ["python", "train.py"]
