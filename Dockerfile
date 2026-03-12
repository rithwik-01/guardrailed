FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

COPY pyproject.toml poetry.lock* ./

RUN echo "Installing runtime dependencies..." && \
    poetry install --no-dev --no-root --sync

RUN echo "Downloading models" && \
    python -m spacy download en_core_web_sm

FROM python:3.10-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENVIRONMENT=production \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface \
    POLICIES_FILE_PATH=policies.yaml \
    SPACY_MODEL_FOR_PRESIDIO=en_core_web_sm \
    PRESIDIO_TRANSFORMERS_MODEL=dslim/bert-base-NER

RUN groupadd --gid 1001 guardrailed && \
    useradd --uid 1001 --gid guardrailed --shell /bin/bash --create-home guardrailed

RUN mkdir -p ${HF_HOME} && chown -R guardrailed:guardrailed ${HF_HOME}

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.10/site-packages/en_core_web_sm /usr/local/lib/python3.10/site-packages/en_core_web_sm
COPY --chown=guardrailed:guardrailed src ./src
COPY --chown=guardrailed:guardrailed policies.yaml ./policies.yaml

RUN chown -R guardrailed:guardrailed /app

USER guardrailed

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail http://localhost:${PORT}/health || exit 1

CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT}