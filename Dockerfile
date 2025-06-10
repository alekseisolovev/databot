FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ENV USER_NAME=user
ENV GROUP_NAME=group
ENV HOME=/home/${USER_NAME}

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd --system --gid ${GROUP_ID} ${GROUP_NAME} \
    && useradd --system --uid ${USER_ID} --gid ${GROUP_NAME} --no-log-init --home ${HOME} --create-home --shell /bin/bash ${USER_NAME}

WORKDIR /app

RUN pip install --upgrade pip

FROM base AS app

COPY --chown=${USER_NAME}:${GROUP_NAME} requirements.app.txt .
RUN pip install -r requirements.app.txt

USER ${USER_NAME}

FROM base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --chown=${USER_NAME}:${GROUP_NAME} requirements.app.txt requirements.dev.txt ./
RUN pip install -r requirements.app.txt
RUN pip install -r requirements.dev.txt

COPY --chown=${USER_NAME}:${GROUP_NAME} . .

USER ${USER_NAME}
