FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir runpod opencv-python

# Install flash-attn with CUDA support
RUN pip3 install --no-cache-dir flash-attn

# Copy project files
COPY . .

# Set environment variables
ENV MODEL_PATH="/runpod-volume/wan-models"
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "runpod.serverless.start", "--handler", "handler.py:handler"] 