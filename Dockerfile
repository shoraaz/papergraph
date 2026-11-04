# PaperGraph API + static frontend in one image.
#   docker build -t papergraph .                       # demo profile (snapshot)
#   docker build -t papergraph --build-arg WITH_ETL=true .   # + ingestion deps
# Runs anywhere that runs a container: Cloud Run, Fly, Render, Koyeb, HF Spaces.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_CACHE_DIR=/opt/models \
    STORE_PROFILE=demo \
    SERVE_FRONTEND=true \
    PORT=8080

WORKDIR /app

COPY requirements.txt requirements-etl.txt ./
RUN pip install -r requirements.txt
ARG WITH_ETL=false
RUN if [ "$WITH_ETL" = "true" ]; then pip install -r requirements-etl.txt; fi

# Bake the ONNX models into the image so a cold start never downloads
# anything (set BAKE_MODELS=false only for offline CI builds).
ARG BAKE_MODELS=true
COPY config.py embeddings.py ./
RUN mkdir -p /opt/models && if [ "$BAKE_MODELS" = "true" ]; then \
      python -c "from embeddings import get_embedder; get_embedder().warm()"; fi

COPY . .
RUN useradd --create-home --uid 10001 app && chown -R app /app /opt/models
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# Keep-alive above typical load-balancer idle timeouts: the load test showed
# uvicorn's 5 s default racing client connection reuse (spurious resets).
CMD exec uvicorn server:app --host 0.0.0.0 --port "$PORT" --timeout-keep-alive 75 --workers "${WEB_CONCURRENCY:-1}"
