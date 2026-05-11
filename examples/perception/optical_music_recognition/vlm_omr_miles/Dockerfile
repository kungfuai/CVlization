# Miles GRPO for VLM OMR (Optical Music Recognition)
# Uses radixark/miles with VLM support (Qwen3-VL/Qwen3.5)

FROM radixark/miles:latest

ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface
ENV HF_DATASETS_CACHE=/root/.cache/huggingface/datasets

# OMR-specific dependencies
RUN pip install --no-cache-dir pyyaml datasets pillow

# Patch Miles qwen3_5 mbridge to handle MTP layer rename
# (Megatron upstream renamed transformer_layer -> mtp_model_layer)
# See https://github.com/THUDM/slime/issues/1894
COPY qwen3_5_mtp_patch.py /tmp/qwen3_5_mtp_patch.py
RUN python3 /tmp/qwen3_5_mtp_patch.py

WORKDIR /workspace

CMD ["python", "train.py"]
