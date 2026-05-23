# Same base as vlm_omr_sft (unsloth + Qwen3-VL stack) + ultralytics so
# the per-measure trainer can also call the multi-task YOLO detector when
# building training crops from images that don't already have GT cells.
FROM cvlization/vlm-omr-sft:latest

# Needed for cv2 (ultralytics pulls it).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ultralytics==8.3.40

WORKDIR /workspace
