FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV APP_USER=appuser
ENV APP_GROUP=appgroup
ENV HOME=/home/appuser

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd --system --gid ${GROUP_ID} ${APP_GROUP} \
    && useradd --system --uid ${USER_ID} --gid ${APP_GROUP} --no-log-init --home ${HOME} --create-home --shell /sbin/nologin ${APP_USER}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R ${APP_USER}:${APP_GROUP} /app
USER ${APP_USER}

