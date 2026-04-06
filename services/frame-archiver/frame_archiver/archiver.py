"""
frame-archiver/archiver.py

Responsibilities
────────────────
This service owns all CPU-side frame processing. It subscribes to two channels:

  FRAME_CHANNEL     — fired by frame-reader on every decoded frame.
                      Imports the CUDA IPC handle, runs tensor.cpu() to transfer
                      the frame to system RAM, encodes a JPEG, and writes it to
                      Redis under jpeg_key. This pre-encodes every frame so the
                      JPEG is ready before any detection event arrives.

  DETECTION_CHANNEL — fired by inference-worker when detections are found.
                      Reads jpeg_key from Redis, annotates with bounding boxes,
                      and writes the annotated JPEG + JSON sidecar to OUTPUT_DIR.
                      Falls back to importing the IPC handle directly if jpeg_key
                      has already expired (race condition safety net).

Why this service holds a GPU reference
───────────────────────────────────────
The tensor.cpu() call requires a valid CUDA IPC handle, which means this service
needs GPU access even though it is otherwise CPU-bound. The GPU is used only for
the PCIe DMA transfer (cudaMemcpy device→host) — no compute kernels run here.
Each imported IPC handle is released immediately after the cpu() call.

Memory lifecycle (archiver side)
─────────────────────────────────
1. Receive frame_id on FRAME_CHANNEL
2. GET ipc_key → reconstruct tensor via _new_shared_cuda (CUDA ref-count +1)
3. tensor.cpu() → numpy array in system RAM (PCIe transfer, ~2ms at 1080p)
4. del tensor + empty_cache() — CUDA ref-count -1
5. cv2.imencode() → jpeg_bytes
6. SETEX jpeg_key → Redis
"""

import os
import json
import time
import pickle
import pathlib
import datetime
import threading
from queue import Queue, Empty

import torch
import numpy as np
import cv2
import redis

from cv_pipeline.keys import make_frame_keys as make_keys, make_result_key, channels

# ── env ───────────────────────────────────────────────────────────────────────
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
CAM_ID = os.environ.get("CAM_ID", "cam0")
OUTPUT_DIR = pathlib.Path(os.environ.get("OUTPUT_DIR", "/archive"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
JPEG_TTL_S = int(os.environ.get("JPEG_TTL_S", "30"))

# ── redis channels ────────────────────────────────────────────────────────────
FRAME_CHANNEL = channels.frame_ready(CAM_ID)
DETECTION_CHANNEL = channels.detection(CAM_ID)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── CUDA IPC import ───────────────────────────────────────────────────────────


def import_ipc_tensor(ipc_bytes: bytes, h: int, w: int) -> torch.Tensor | None:
    """Reconstruct a (H, W, 3) uint8 BGR tensor from a CUDA IPC handle.

    Returns None if the handle has expired or is otherwise invalid.
    Caller is responsible for del + torch.cuda.empty_cache() after use.
    """
    try:
        ipc_tuple = pickle.loads(ipc_bytes)
        storage = torch.Storage._new_shared_cuda(*ipc_tuple)
        return torch.tensor(storage, dtype=torch.uint8).reshape(h, w, 3)
    except Exception as e:
        print(f"[archiver] IPC import failed: {e}")
        return None


# ── JPEG encode ───────────────────────────────────────────────────────────────


def tensor_to_jpeg(frame_bgr: torch.Tensor) -> bytes:
    """Transfer tensor to CPU and encode as JPEG. Releases IPC reference."""
    try:
        frame_cpu = frame_bgr.cpu().numpy()  # PCIe transfer: ~2ms at 1080p
        ok, buf = cv2.imencode(
            ".jpg",
            frame_cpu,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        return buf.tobytes() if ok else b""
    finally:
        del frame_bgr
        torch.cuda.empty_cache()


# ── FRAME_CHANNEL handler ─────────────────────────────────────────────────────


def handle_frame(rdb: redis.Redis, frame_id: str):
    """Import IPC handle, encode JPEG, write jpeg_key to Redis.

    Called for every frame regardless of whether a detection fires.
    Pre-encoding ensures the JPEG is ready when archive_detection needs it.
    """
    ipc_key, jpeg_key, meta_key = make_keys(frame_id)

    pipe = rdb.pipeline(transaction=False)
    pipe.get(ipc_key)
    pipe.hgetall(meta_key)
    ipc_bytes, meta_raw = pipe.execute()

    if not ipc_bytes:
        print(f"[archiver] IPC handle expired before encode — skipped {frame_id}")
        return

    meta = {k.decode(): v.decode() for k, v in meta_raw.items()}
    h, w = int(meta.get("h", 0)), int(meta.get("w", 0))
    if h == 0 or w == 0:
        return

    frame_bgr = import_ipc_tensor(ipc_bytes, h, w)
    if frame_bgr is None:
        return

    jpeg_bytes = tensor_to_jpeg(frame_bgr)  # releases IPC ref internally
    if jpeg_bytes:
        rdb.setex(jpeg_key, JPEG_TTL_S, jpeg_bytes)


# ── DETECTION_CHANNEL handler ─────────────────────────────────────────────────


def handle_detection(rdb: redis.Redis, event: dict):
    """Fetch JPEG, annotate with detections, write to disk.

    Falls back to importing the IPC handle directly if the JPEG hasn't been
    written yet (e.g. archiver fell behind) or has already expired.
    """
    frame_id = event["frame_id"]
    cam_id = event["cam_id"]
    detections = event.get("detections", [])
    ts_ms = event.get("ts_ms", int(time.time() * 1000))

    ipc_key, jpeg_key, meta_key = make_keys(frame_id)

    # ── Fetch JPEG — optimistic path (handle_frame already wrote it) ──────────
    pipe = rdb.pipeline(transaction=False)
    pipe.get(jpeg_key)
    pipe.hgetall(meta_key)
    jpeg_bytes, meta_raw = pipe.execute()

    meta = {k.decode(): v.decode() for k, v in (meta_raw or {}).items()}
    h, w = int(meta.get("h", 0)), int(meta.get("w", 0))

    # ── Fallback: re-import IPC handle if JPEG missing ────────────────────────
    if not jpeg_bytes:
        print(
            f"[archiver] jpeg_key missing for {frame_id} — falling back to IPC import"
        )
        ipc_bytes = rdb.get(ipc_key)
        if ipc_bytes and h > 0 and w > 0:
            frame_bgr = import_ipc_tensor(ipc_bytes, h, w)
            if frame_bgr is not None:
                jpeg_bytes = tensor_to_jpeg(frame_bgr)

    if not jpeg_bytes:
        print(
            f"[archiver] cannot recover frame {frame_id} — both jpeg_key and IPC expired"
        )
        return

    # ── Annotate and write ────────────────────────────────────────────────────
    annotated = (
        draw_detections(jpeg_bytes, detections, h, w) if detections else jpeg_bytes
    )

    dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
    stamp = dt.strftime("%Y%m%dT%H%M%S%f")[:-3]
    stem = f"{cam_id}_{stamp}"

    (OUTPUT_DIR / f"{stem}.jpg").write_bytes(annotated)
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps(
            {
                "frame_id": frame_id,
                "cam_id": cam_id,
                "timestamp": dt.isoformat(),
                "width": w,
                "height": h,
                "detections": detections,
                "worker_id": event.get("worker_id"),
            },
            indent=2,
        )
    )

    print(f"[archiver] wrote {stem}.jpg  {len(detections)} detections")


# ── Bounding box overlay ──────────────────────────────────────────────────────


def draw_detections(jpeg_bytes: bytes, detections: list[dict], h: int, w: int) -> bytes:
    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jpeg_bytes

        for det in detections:
            bbox = det.get("bbox", [])
            label = det.get("label", "")
            score = det.get("score", 0.0)
            if len(bbox) == 4:
                cx, cy, bw, bh = bbox
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{label} {score:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else jpeg_bytes
    except Exception as e:
        print(f"[archiver] draw_detections failed: {e}")
        return jpeg_bytes


# ── S3 upload stub ────────────────────────────────────────────────────────────


def upload_to_s3(jpeg_path: pathlib.Path, json_path: pathlib.Path):
    """
    Stub for S3/GCS upload. Uncomment and configure as needed.

    import boto3
    s3 = boto3.client("s3")
    bucket = os.environ["S3_BUCKET"]
    prefix = os.environ.get("S3_PREFIX", "detections")
    s3.upload_file(str(jpeg_path), bucket, f"{prefix}/{jpeg_path.name}")
    s3.upload_file(str(json_path), bucket, f"{prefix}/{json_path.name}")
    """
    pass


# ── Main loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rdb = redis.Redis(host=REDIS_HOST, port=6379, db=0, socket_keepalive=True)
    pubsub = rdb.pubsub()
    pubsub.subscribe(FRAME_CHANNEL, DETECTION_CHANNEL)

    print(
        f"[archiver] subscribed to {FRAME_CHANNEL} + {DETECTION_CHANNEL}  "
        f"output={OUTPUT_DIR}  jpeg_quality={JPEG_QUALITY}  jpeg_ttl={JPEG_TTL_S}s"
    )

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        channel = message["channel"].decode()
        data = message["data"].decode()

        try:
            if channel == FRAME_CHANNEL:
                handle_frame(rdb, frame_id=data)

            elif channel == DETECTION_CHANNEL:
                handle_detection(rdb, event=json.loads(data))

        except Exception as e:
            print(f"[archiver] error on {channel}: {e}")
