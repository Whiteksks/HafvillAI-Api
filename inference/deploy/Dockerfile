FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

WORKDIR /app
COPY hafvillai_runtime/ hafvillai_runtime/
COPY mamba-source/ mamba-source/
COPY inference/ inference/
RUN python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install 'setuptools>=61' wheel packaging ninja \
    && python -m pip install -r inference/requirements.txt --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -e ./mamba-source --no-build-isolation \
    && python -m pip check

RUN useradd --uid 10001 --create-home hafvillai \
    && mkdir -p /var/cache/hafvillai \
    && chown 10001:10001 /var/cache/hafvillai
ENV HAFVILLAI_HOST=0.0.0.0 \
    HAFVILLAI_PORT=8000 \
    HAFVILLAI_ARTIFACTS_DIR=/app/inference/artifacts \
    HF_HUB_OFFLINE=1 \
    HOME=/var/cache/hafvillai \
    CUDA_CACHE_PATH=/var/cache/hafvillai/cuda \
    XDG_CACHE_HOME=/var/cache/hafvillai \
    TRITON_CACHE_DIR=/var/cache/hafvillai/triton \
    TILELANG_CACHE_DIR=/var/cache/hafvillai/tilelang \
    TORCH_EXTENSIONS_DIR=/var/cache/hafvillai/torch_extensions
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).close()"]
CMD ["python", "-u", "-m", "inference.api"]
