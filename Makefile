.PHONY: help up down logs lint test test-unit test-integration build shell-reader shell-worker shell-archiver clean

# Default target
help:
	@echo "cv-pipeline dev targets"
	@echo ""
	@echo "  up               Start all services (redis + GPU services + mediamtx)"
	@echo "  up-test-stream   Start mediamtx + rtsp-streamer only (no GPU required)"
	@echo "  down             Stop and remove containers"
	@echo "  logs             Tail logs from all services"
	@echo "  logs-reader      Tail frame-reader logs"
	@echo "  logs-worker      Tail inference-worker logs"
	@echo "  logs-streamer    Tail rtsp-streamer logs"
	@echo ""
	@echo "  build            Build all service images"
	@echo "  build-reader     Build frame-reader image only"
	@echo ""
	@echo "  lint             Run ruff + black + mypy"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only (no services needed)"
	@echo "  test-integration Run integration tests (requires redis)"
	@echo ""
	@echo "  shell-reader     Exec bash inside the frame-reader container"
	@echo "  shell-worker     Exec bash inside the inference-worker container"
	@echo "  shell-redis      redis-cli inside the redis container"
	@echo ""
	@echo "  clean            Remove __pycache__ and .pyc files"

# ── Services ──────────────────────────────────────────────────────────────────

up:
	docker compose up -d

up-test-stream:
	docker compose up -d mediamtx rtsp-streamer

down:
	docker compose down

logs:
	docker compose logs -f

logs-reader:
	docker compose logs -f frame-reader

logs-worker:
	docker compose logs -f inference-worker

logs-streamer:
	docker compose logs -f rtsp-streamer mediamtx

build:
	docker compose build

build-reader:
	docker compose build frame-reader

build-worker:
	docker compose build inference-worker

# ── Shells ────────────────────────────────────────────────────────────────────

shell-reader:
	docker compose exec frame-reader bash

shell-worker:
	docker compose exec inference-worker bash

shell-archiver:
	docker compose exec frame-archiver bash

shell-redis:
	docker compose exec redis redis-cli

# ── Linting & formatting ──────────────────────────────────────────────────────

lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy shared/cv_pipeline services/frame-archiver/frame_archiver

fmt:
	uv run ruff check --fix .
	uv run black .

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v -m unit

test-integration:
	uv run pytest tests/integration/ -v -m integration

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name ".mypy_cache" -exec rm -rf {} +
	find . -name ".ruff_cache" -exec rm -rf {} +
