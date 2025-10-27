FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training and data preparation scripts
COPY train.py .
COPY prepare_captcha_data.py .
COPY test_model.py .

# Copy moondream modules (official implementation)
COPY moondream /workspace/moondream

# Default command
CMD ["python", "train.py", "--help"]
