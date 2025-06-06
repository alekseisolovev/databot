FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV APP_USER=appuser \
    APP_GROUP=appgroup \
    HOME=/home/appuser

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd --system --gid ${GROUP_ID} ${APP_GROUP} \
    && useradd --system --uid ${USER_ID} --gid ${APP_GROUP} --no-log-init --home ${HOME} --create-home --shell /bin/bash ${APP_USER}

WORKDIR /app
ENV PYTHONPATH="/app"

RUN pip install --upgrade pip

FROM base AS app

COPY --chown=${APP_USER}:${APP_GROUP} requirements.app.txt .
RUN pip install -r requirements.app.txt

USER ${APP_USER}

FROM base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --chown=${APP_USER}:${APP_GROUP} requirements.app.txt requirements.dev.txt ./
RUN pip install -r requirements.app.txt 
RUN pip install -r requirements.dev.txt

COPY --chown=${APP_USER}:${APP_GROUP} . .

USER ${APP_USER}
