# hash:sha256:59b138b7ccc9b9433b3ea7003fd266a89a06a70f062651f97ab115b655db9279
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
    polars-ds \
    polars-vec-ops \
    pydantic-settings
