# CareCompass — Hugging Face Spaces (Docker SDK) image
#
# Hugging Face runs Space containers as UID 1000, so every path the app writes
# to (ChromaDB, audit logs, pip's install dir, caches) must be owned by that
# user. Creating `user` up front and copying with --chown keeps the container
# working the same way locally and on Spaces.

FROM python:3.11-slim

# System dependencies (root layer) — build-essential covers source builds for
# chroma-hnswlib, curl backs the health check.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Match the UID Hugging Face Spaces runs containers as.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Install dependencies first so code edits don't invalidate the pip layer.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Writable working directories owned by UID 1000.
RUN mkdir -p chroma_db logs data

# CHROMA_PERSIST_DIRECTORY and AUDIT_LOG_PATH are deliberately left unset: the
# app defaults them inside this (user-owned) workdir, and the entrypoint
# redirects them to /data when persistent storage is attached.
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["./scripts/entrypoint.sh"]
