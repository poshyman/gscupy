FROM nvcr.io/nvidia/cuda:12.3-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3 python3-pip curl \
    # GStreamer core + RTSP plugins
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    # NVDEC + CUDAMemory GStreamer elements
    gstreamer1.0-cuda \
    libgstcuda-1.0-0 \
    libgstcuda-1.0-dev \
    # Python GObject bindings (not installable via pip)
    python3-gi \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstcuda-1.0 \
    && rm -rf /var/lib/apt/lists/*

# ── uv ────────────────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ── Workspace metadata (cached layer — only invalidated by pyproject changes) ─
# Copy every pyproject.toml and the root workspace config first.
# uv needs the full workspace to resolve inter-package dependencies.
WORKDIR /app
COPY pyproject.toml                              ./pyproject.toml
COPY shared/pyproject.toml                      ./shared/pyproject.toml
COPY services/frame-reader/pyproject.toml       ./services/frame-reader/pyproject.toml
COPY services/inference-worker/pyproject.toml   ./services/inference-worker/pyproject.toml
COPY services/frame-archiver/pyproject.toml     ./services/frame-archiver/pyproject.toml

# Copy lock file if it exists — produces a reproducible install.
# Run `uv lock` locally to generate uv.lock before building.
COPY uv.lock* ./

# ── Install dependencies ──────────────────────────────────────────────────────
# --package installs only this service + its transitive deps (not the whole workspace).
# --extra gpu pulls in torch and cupy-cuda12x declared in [project.optional-dependencies].
# --no-install-project skips installing the package itself (source not copied yet).
RUN uv sync \
    --package cv-pipeline-frame-reader \
    --extra gpu \
    --no-install-project \
    --no-dev

# ── Source (separate layer — invalidated only by code changes) ────────────────
COPY shared/                     ./shared/
COPY services/frame-reader/      ./services/frame-reader/

# Install the package itself now that source is present
RUN uv sync \
    --package cv-pipeline-frame-reader \
    --extra gpu \
    --no-dev

CMD ["python3", "-u", "-m", "frame_reader.reader"]
