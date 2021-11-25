
FROM python:3.9-slim-bullseye

USER root

ARG DEBIAN_FRONTEND="noninteractive"



RUN apt-get update \
    && apt-get -qq install git \
    && git clone https://github.com/SPiN-Lab/connPFM.git \ # change during the testing so it downloads the PR version
    && cd connPFM \
    && pip install --no-cache-dir -e . \
    && pip install PyWavelets

ENV PYTHONPATH "${PYTHONPATH}:/connPFM"