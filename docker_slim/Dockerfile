
FROM python:3.9-slim-bullseye

USER root

ARG DEBIAN_FRONTEND="noninteractive"

COPY . .


RUN apt-get update \
    && apt-get -qq install git \
    && pip install --no-cache-dir -e . \
    && pip install PyWavelets

ENV PYTHONPATH "${PYTHONPATH}:/connPFM"