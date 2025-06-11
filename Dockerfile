FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app
RUN pip install --upgrade pip
RUN pip install poetry

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --ingroup appgroup --home /home/appuser appuser

FROM base AS builder
COPY --chown=appuser:appgroup pyproject.toml poetry.lock* ./
RUN poetry install --no-root
ENTRYPOINT ["poetry", "run"]

FROM base AS production
COPY --chown=appuser:appgroup pyproject.toml poetry.lock* ./

RUN poetry install --no-root --only main

COPY --chown=appuser:appgroup src ./src

USER appuser
ENTRYPOINT ["poetry", "run"]
