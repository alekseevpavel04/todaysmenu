# Start with CUDA supported Python image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
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

# Pre-download the model to cache it in the image
# Uncomment if you want to include the model in the image (makes bigger images but faster startup)
# RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); print('Токенизатор загружен'); model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', torch_dtype='float16'); print('Модель загружена')"

# Copy the application code
COPY . .

# Create directory for model cache
RUN mkdir -p /root/.cache/huggingface

# Expose the port for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["python", "app.py"]