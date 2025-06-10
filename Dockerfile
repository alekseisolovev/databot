FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    PATH="/app/.venv/bin:$PATH"

ENV USER_NAME=user
ENV GROUP_NAME=group
ENV HOME=/home/${USER_NAME}

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd --system --gid ${GROUP_ID} ${GROUP_NAME} \
    && useradd --system --uid ${USER_ID} --gid ${GROUP_NAME} --no-log-init --home ${HOME} --create-home --shell /bin/bash ${USER_NAME}

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install poetry

FROM base AS app

COPY --chown=${USER_NAME}:${GROUP_NAME} pyproject.toml poetry.lock* ./
RUN poetry install --no-root --without dev

COPY --chown=${USER_NAME}:${GROUP_NAME} src ./src

USER ${USER_NAME}

ENTRYPOINT ["poetry", "run"]

FROM base AS dev

RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --chown=${USER_NAME}:${GROUP_NAME} pyproject.toml poetry.lock* ./
RUN poetry install --no-root

COPY --chown=${USER_NAME}:${GROUP_NAME} . .

USER ${USER_NAME}

ENTRYPOINT ["poetry", "run"]
