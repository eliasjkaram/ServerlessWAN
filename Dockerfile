FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables for non-interactive apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    python3-pip \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir runpod opencv-python

# Skip flash-attn for now - we'll use PyTorch's built-in attention mechanism
# Don't try to install from source, as it requires CUDA development files
RUN pip3 install --no-cache-dir pip install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir transformers>=4.49.0 tokenizers>=0.20.3 accelerate>=1.1.1

# Copy project files
COPY . .

# Set environment variables
ENV MODEL_PATH="/runpod-volume/wan-models"
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "runpod.serverless.start", "--handler", "handler.py:handler"] 