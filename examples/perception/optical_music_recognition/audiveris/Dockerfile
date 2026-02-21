FROM ubuntu:24.04

# System deps: Java (from Audiveris bundled JRE), Tesseract OCR, Xvfb (headless display),
# Python 3 (for predict.py wrapper), wget for downloading the installer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
        tesseract-ocr \
        tesseract-ocr-eng \
        xvfb \
        python3 \
        python3-pip \
    # Download and install Audiveris 5.9.0 (bundled JRE; apt-get resolves all deps).
    # The postinst script registers desktop MIME types via xdg-utils, which fails
    # in a headless Docker container — ignore that failure; the binary still installs.
    && wget -q "https://github.com/Audiveris/audiveris/releases/download/5.9.0/Audiveris-5.9.0-ubuntu24.04-x86_64.deb" \
    && (apt-get install -y ./Audiveris-5.9.0-ubuntu24.04-x86_64.deb || true) \
    && test -f /opt/audiveris/bin/Audiveris \
    && rm Audiveris-5.9.0-ubuntu24.04-x86_64.deb \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/audiveris/bin:${PATH}"

# Point Audiveris to system Tesseract data (apt installs to this path)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
CMD ["python3", "predict.py"]
