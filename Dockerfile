# Dockerfile for Coqui TTS (XTTS) RunPod Serverless
# Based on NVIDIA CUDA image with PyTorch

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

WORKDIR /app

# Install system deps and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    espeak-ng \
    libsndfile1-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install PyTorch 2.4 with CUDA 12.1
RUN pip3 install --no-cache-dir \
    torch>=2.4.0 \
    torchaudio>=2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install Coqui TTS (includes XTTS model)
RUN pip3 install --no-cache-dir \
    TTS>=0.22.0 \
    soundfile>=0.12.0 \
    librosa>=0.10.0 \
    scipy>=1.11.0

# Upgrade transformers for BeamSearchScorer compatibility
RUN pip3 install --no-cache-dir \
    transformers>=4.40.0

# Install Python deps for handler and RunPod
RUN pip3 install --no-cache-dir \
    boto3>=1.26.0 \
    requests>=2.28.0 \
    numpy>=1.24.0 \
    pynvml>=11.5.0 \
    psutil>=5.9.0 \
    runpod>=0.9.0

# Copy only the necessary files
COPY handler.py /app/handler.py
COPY concurrency.py /app/concurrency.py
COPY worker.py /app/worker.py

ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 8000

CMD ["python3", "/app/worker.py"]
