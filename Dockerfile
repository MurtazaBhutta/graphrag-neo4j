FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRAPHRAG_CACHE=/app/.cache

WORKDIR /app
COPY pyproject.toml README.md ./
COPY graphrag ./graphrag
RUN pip install . && useradd --create-home app && mkdir -p .cache && chown -R app /app
USER app
# Fetch the embedding model at build time, so the first request is not a 70 MB download.
RUN python -c "from graphrag.embeddings import FastEmbedder; FastEmbedder()"

EXPOSE 8000
CMD ["uvicorn", "graphrag.api:app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
