# Shared base image for all Python ML services:
#   inference_consumer, feedback_embedder, online_learner, fastapi, streamlit
#
# CMD is overridden per service in docker-compose.yml
# All services share the same installed packages — one image, multiple containers

FROM python:3.12-slim-bookworm

WORKDIR /app

# System dependencies:
# - build-essential: needed for some Python package native extensions
# - libgomp1: OpenMP runtime required by XGBoost
# - curl: health check support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first — cached layer unless requirements.txt changes
COPY python-ml/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir setuptools

# Copy Python ML source
COPY python-ml/ .

# Create non-root user with home directory
RUN groupadd -r mluser && useradd -r -g mluser -m mluser

# 1. Set cache path
# Set HuggingFace cache to app directory (not /home/mluser)
# so it is accessible regardless of user home permissions
ENV HF_HOME=/app/.hf_cache
ENV TRANSFORMERS_CACHE=/app/.hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache

# 2. Create directory and Download model — now uses /app/.hf_cache
# Pre-download sentence-transformers model at build time as root
# into /app/.hf_cache — avoids permission errors and download delays at runtime
RUN mkdir -p /app/.hf_cache && \
    python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')" || true

# Fix all ownership after downloads complete
RUN chown -R mluser:mluser /app

# Create data directory for ChromaDB volume mount
# Must exist before USER switch so mluser can write to it
RUN mkdir -p /data/chroma_db && chown -R mluser:mluser /data

USER mluser

# PYTHONPATH ensures all modules resolve correctly regardless of CMD
ENV PYTHONPATH=/app

# Default CMD — overridden per service in docker-compose.yml
CMD ["python", "-m", "consumers.inference_consumer"]