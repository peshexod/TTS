# Dockerfile for Coqui TTS (XTTS) RunPod Serverless
# Based on PyTorch CUDA image

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    espeak-ng \
    libsndfile1-dev \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Coqui TTS (includes XTTS model)
RUN pip3 install --no-cache-dir \
    TTS>=0.22.0 \
    soundfile>=0.12.0 \
    librosa>=0.10.0 \
    scipy>=1.11.0

# Install Python deps for handler
RUN pip3 install --no-cache-dir \
    boto3>=1.26.0 \
    requests>=2.28.0 \
    numpy>=1.24.0

# Copy handler
COPY handler.py /app/handler.py

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=60s --start-period=300s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

EXPOSE 8000

# Run handler directly (RunPod will wrap this)
CMD ["python3", "/app/handler.py"]
