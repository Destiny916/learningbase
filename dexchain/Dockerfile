ARG BASE_IMAGE=192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v3
FROM ${BASE_IMAGE}

USER root

ARG DEXECHAIN_VERSION=0.1.6
ARG EMBODICHAIN_VERSION=0.2.4
ARG OPEN3DV_INDEX_URL=http://pyp.open3dv.site:2345/simple/
ARG DEXFORCE_INDEX_URL=http://192.168.3.43:8080/simple/

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_TRUSTED_HOST="pyp.open3dv.site 192.168.3.43" \
    PYTHONUNBUFFERED=1

RUN python3 -m pip install --no-cache-dir \
        "embodichain==${EMBODICHAIN_VERSION}" \
        "dexechain==${DEXECHAIN_VERSION}" \
        --extra-index-url "${OPEN3DV_INDEX_URL}" \
        --extra-index-url "${DEXFORCE_INDEX_URL}" \
        --trusted-host pyp.open3dv.site \
        --trusted-host 192.168.3.43

WORKDIR /workspace/dexchain

ENTRYPOINT []
CMD ["/bin/bash"]
