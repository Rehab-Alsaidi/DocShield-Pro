# deployment/Dockerfile
# Multi-stage Docker build for PDF Content Moderator

# Stage 1: Base image with system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    gcc \
    g++ \
    make \
    # PDF processing dependencies
    poppler-utils \
    # Image processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PostgreSQL client
    libpq-dev \
    # File type detection
    libmagic1 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Stage 2: Python dependencies
FROM base as dependencies

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY deployment/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Stage 3: Production image
FROM dependencies as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs \
             /app/static/uploads \
             /app/static/results \
             /app/static/reports \
             /app/temp && \
    chown -R appuser:appuser /app

# Download required NLP models
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')" || echo "SpaCy model download failed"
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')" || echo "NLTK data download failed"

# Set up model cache directory
ENV TRANSFORMERS_CACHE=/app/model_cache
RUN mkdir -p /app/model_cache && chown -R appuser:appuser /app/model_cache

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["python", "run.py", "--mode", "prod", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Development image
FROM dependencies as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest pytest-cov pytest-flask \
    black flake8 mypy isort \
    jupyter notebook ipython

# Create non-root user
RUN groupadd -r devuser && useradd -r -g devuser devuser

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs \
             /app/static/uploads \
             /app/static/results \
             /app/static/reports \
             /app/temp && \
    chown -R devuser:devuser /app

# Switch to non-root user
USER devuser

# Expose ports (app + jupyter)
EXPOSE 8000 8888

# Development command
CMD ["python", "run.py", "--mode", "dev", "--host", "0.0.0.0", "--port", "8000", "--debug"]

# Stage 5: GPU-enabled image
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq-dev \
    libmagic1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies
COPY deployment/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install CUDA-enabled PyTorch
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Create non-root user
RUN groupadd -r gpuuser && useradd -r -g gpuuser gpuuser

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/logs \
             /app/static/uploads \
             /app/static/results \
             /app/static/reports \
             /app/temp \
             /app/model_cache && \
    chown -R gpuuser:gpuuser /app

# Download NLP models
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')" || echo "SpaCy model download failed"

# Set environment variables
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV CUDA_AVAILABLE=true

# Switch to non-root user
USER gpuuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# GPU command
CMD ["python", "run.py", "--mode", "prod", "--host", "0.0.0.0", "--port", "8000"]

# Build arguments for different targets
ARG BUILD_TARGET=production

# Build-time labels
LABEL maintainer="PDF Content Moderator Team"
LABEL version="1.0.0"
LABEL description="AI-powered PDF content moderation system"
LABEL build-target=${BUILD_TARGET}

# Runtime labels
LABEL org.opencontainers.image.title="PDF Content Moderator"
LABEL org.opencontainers.image.description="AI-powered content analysis for PDF documents"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/your-repo/pdf-content-moderator"
LABEL org.opencontainers.image.licenses="MIT"

# Additional metadata
LABEL ai.models="Florence-2,CLIP,NSFW-Detection,Sentence-Transformers"
LABEL ai.frameworks="PyTorch,Transformers,Flask"
LABEL deployment.platform="Docker,Kubernetes,Railway,AWS"