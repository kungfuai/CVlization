FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace

CMD ["python", "verify.py", "submissions/reference_python_add.py"]
