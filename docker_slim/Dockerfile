
FROM python:3.9-slim-bullseye

USER root

ARG DEBIAN_FRONTEND="noninteractive"



RUN apt-get update \
    && apt-get -qq install git \
    && cd connPFM \
    && pip install --no-cache-dir -e . \
    && pip install PyWavelets

ENV PYTHONPATH "${PYTHONPATH}:/connPFM"