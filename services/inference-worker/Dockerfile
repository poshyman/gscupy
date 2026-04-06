FROM nvcr.io/nvidia/cuda:12.3-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ── Workspace metadata ────────────────────────────────────────────────────────
WORKDIR /app
COPY pyproject.toml                              ./pyproject.toml
COPY shared/pyproject.toml                      ./shared/pyproject.toml
COPY services/frame-reader/pyproject.toml       ./services/frame-reader/pyproject.toml
COPY services/inference-worker/pyproject.toml   ./services/inference-worker/pyproject.toml
COPY services/frame-archiver/pyproject.toml     ./services/frame-archiver/pyproject.toml
COPY uv.lock* ./

# ── Install dependencies ──────────────────────────────────────────────────────
RUN uv sync \
    --package cv-pipeline-inference-worker \
    --extra gpu \
    --no-install-project \
    --no-dev

# ── Source ────────────────────────────────────────────────────────────────────
COPY shared/                       ./shared/
COPY services/inference-worker/    ./services/inference-worker/

RUN uv sync \
    --package cv-pipeline-inference-worker \
    --extra gpu \
    --no-dev

CMD ["python3", "-u", "-m", "inference_worker.worker"]
