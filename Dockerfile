FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Set environment variables for non-interactive apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir opencv-python-headless

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /tmp

# Set environment variables
ENV MODEL_PATH="/runpod-volume/wan-models"
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "runpod.serverless.start", "--handler", "handler.py:handler"] 