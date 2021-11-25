
FROM python:3.9-slim-bullseye

USER root

ARG DEBIAN_FRONTEND="noninteractive"



RUN apt-get update \
    && apt-get -qq install git \
    && pr=$(echo https://api.github.com/repos/${CIRCLE_PULL_REQUEST:19} | sed "s/\/pull\//\/pulls\//") \
    && base=$(curl -s $pr | jq '.base.ref')
    && cd connPFM \
    && pip install --no-cache-dir -e . \
    && pip install PyWavelets

ENV PYTHONPATH "${PYTHONPATH}:/connPFM"