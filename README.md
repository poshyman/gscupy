# gscupy

Real-time computer vision pipeline: RTSP/file → NVDEC (zero-copy) → Redis → inference workers → event-driven archival.

## Architecture

```
RTSP camera / video file
        │
        ▼
  frame-reader          GStreamer NVDEC → CUDA IPC handle → Redis
        │                JPEG side-channel → Redis (for archiver)
        │ PUBLISH frame_ready:{cam_id}
        ▼
  inference-worker(s)   Import CUDA IPC → preprocess on GPU → model → detections
        │
        │ PUBLISH detection:{cam_id}
        ▼
  frame-archiver        Fetch JPEG from Redis → annotate → write to disk / S3
```

## Quickstart

### Prerequisites

- Docker Desktop with the NVIDIA container runtime
- NVIDIA GPU with NVDEC support (L40S, A100, RTX series)
- VSCode with the Dev Containers extension

### 1. Clone and configure

```bash
git clone <repo>
cd gscupy
cp .env.example .env
# Edit .env with your RTSP_URL and other settings
```

### 2. Open in dev container

Open the repo in VSCode. When prompted, click **Reopen in Container**.

VSCode will:
1. Build the lightweight dev container image
2. Start Redis, frame-reader, inference-worker, and frame-archiver via Compose
3. Run `.devcontainer/post-create.sh` to install the uv virtual environment

### 3. Validate the pipeline

```bash
bash scripts/validate-pipeline.sh
```

### 4. Watch it run

```bash
make logs          # tail all service logs
make shell-redis   # redis-cli to inspect keys live
bash scripts/redis-monitor.sh cam0
```

## Development workflow

```bash
make lint          # ruff + black + mypy
make test-unit     # fast tests, no services needed
make test-integration  # requires redis (already running in Compose)
make fmt           # auto-fix formatting
```

To test GPU pipeline code that can't run in the dev container:

```bash
bash scripts/run-in-service.sh frame-reader
# You're now inside the frame-reader container with NVDEC available
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Project structure

```
gscupy/
├── .devcontainer/
│   ├── devcontainer.json           VSCode dev container config
│   ├── docker-compose.devcontainer.yml  Adds devcontainer service to stack
│   ├── Dockerfile.dev              Lightweight dev image (no CUDA)
│   └── post-create.sh              uv venv setup, pre-commit install
│
├── services/
│   ├── frame-reader/               GStreamer RTSP/file → NVDEC → Redis
│   ├── inference-worker/           CUDA IPC → model → detection events
│   └── frame-archiver/             Detection events → annotated JPEG → disk
│
├── shared/
│   └── gscupy/
│       └── keys.py                 Single source of truth for Redis key schema
│
├── tests/
│   ├── unit/                       No services required
│   └── integration/                Requires redis on cv_net
│
├── scripts/
│   ├── validate-pipeline.sh        Smoke test GStreamer pipeline
│   ├── redis-monitor.sh            Live Redis key/channel monitor
│   └── run-in-service.sh           Exec into a GPU service container
│
├── videos/                         Mount test video files here (gitignored)
├── archive/                        Detection output (gitignored)
├── docker-compose.yml
├── pyproject.toml                  uv workspace + ruff/mypy/pytest config
├── Makefile
└── .env.example
```

## Environment variables

See `.env.example` for all variables with documentation.

Key ones:

| Variable | Default | Description |
|---|---|---|
| `SOURCE` | `rtsp` | `rtsp` or `file` |
| `RTSP_URL` | — | Full RTSP URL including credentials |
| `VIDEO_PATH` | — | Absolute path inside container when `SOURCE=file` |
| `CODEC` | `h264` | `h264` or `h265` |
| `FRAME_RATE` | `5` | Frames per second passed to inference |
| `FRAME_TTL_MS` | `2000` | How long the CUDA IPC handle lives in Redis |
| `JPEG_TTL_S` | `30` | How long the JPEG side-channel lives in Redis |
| `MODEL_TYPE` | `maskdino` | `maskdino` or `yolov8` |

## GPU memory lifecycle

```
NVDEC decode → GstCudaMemory (GStreamer owns, callback lifetime)
    │
    └─ .contiguous() → PyTorch VRAM alloc (PyTorch owns)
            │
            ├─ share_memory_() → CUDA IPC handle → Redis (FRAME_TTL_MS)
            │       └─ worker imports handle → CUDA ref-count +1
            │               └─ del tensor + empty_cache() → ref-count -1
            │
            └─ _expire_tensor thread → del + empty_cache() after JPEG_TTL_S
                    → CUDA ref-count -1 → cudaFree when all workers released
```
