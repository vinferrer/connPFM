
FROM python:3.9-slim-bullseye

USER root

ARG DEBIAN_FRONTEND="noninteractive"



RUN apt-get update \
    && apt-get -qq install git \
    && git clone https://github.com/SPiN-Lab/connPFM.git \
    && cd connPFM \
    && pip install --no-cache-dir -e .

ENV PYTHONPATH "${PYTHONPATH}:/connPFM"