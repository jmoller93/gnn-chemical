# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yaml .

# Create output directory
RUN mkdir -p outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for TensorBoard (optional)
EXPOSE 6006

# Default command (run training script)
CMD ["python", "scripts/train.py"]

# Alternative commands for different use cases:
# Training: docker compose --profile train up train
# TensorBoard: docker compose --profile tensorboard up tensorboard  
# Interactive: docker compose --profile dev run --rm dev 