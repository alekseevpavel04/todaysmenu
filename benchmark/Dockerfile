FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime


# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
# Set environment variable for Hugging Face to use the cache directory
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create and set permissions for model cache directory
RUN mkdir -p /root/.cache/huggingface && \
    chmod -R 777 /root/.cache/huggingface

# Copy benchmark script
COPY benchmark.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Entry point
CMD ["python", "benchmark.py"]