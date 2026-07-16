# syntax=docker/dockerfile:1
# ── API serving image (CPU-only) ─────────────────────────────────────────────
# Nothing model-related is baked in at build time: HF checkpoints are fetched
# on first startup into the /models volume (HF_HOME), keeping the image small
# and rebuilds fast. A future ONNX/int8 serving path (config: serving.*) may
# replace the transformers runtime here entirely.
#
# Build (repo root as context):  docker build -f api/Dockerfile -t fashion-api .
# Run:                           docker run -p 8000:8000 -v models:/models fashion-api

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    PYTHONPATH=/app

WORKDIR /app

# deps first (cached layer) — CPU torch wheels via the extra index
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# application code (ML logic + config + serving layer)
COPY config/ config/
COPY core/ core/
COPY indexer/ indexer/
COPY retriever/ retriever/
COPY api/ api/

RUN mkdir -p /models data/artifacts data/images

EXPOSE 8000

# api.startup syncs index artifacts from S3 (no-op in local mode), then serve.
CMD ["/bin/sh","-c","python -m api.startup && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
