# hash:sha256:35da401dd2165404320f7f84cf3593e69773d2ba8b29ff6aa5e5d48cab9c3492
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    cloudpickle \
    lazynwb \
    numba \
    pedantic-settings \
    polars-ds \
    polars-vec-ops
