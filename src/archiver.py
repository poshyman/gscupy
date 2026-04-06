"""
frame-archiver/archiver.py

Listens for detection events on DETECTION_CHANNEL.
On each event:
  1. Fetch JPEG bytes from Redis (written by frame-reader, no GPU needed)
  2. Fetch detection result metadata
  3. Write annotated JPEG + JSON sidecar to OUTPUT_DIR
  4. Optionally push to object storage (S3/GCS stub included)

This service is intentionally CPU-only — it works entirely from the JPEG
side-channel that frame-reader wrote, so it never touches VRAM or IPC handles.
GPU memory lifecycle is fully decoupled from archival.
"""

import os
import json
import time
import pathlib
import datetime

import redis
import cv2
import numpy as np

# ── env ───────────────────────────────────────────────────────────────────────
REDIS_HOST  = os.environ.get("REDIS_HOST", "localhost")
CAM_ID      = os.environ.get("CAM_ID", "cam0")
OUTPUT_DIR  = pathlib.Path(os.environ.get("OUTPUT_DIR", "/archive"))

# ── redis channels & key schema ──────────────────────────────────────────────
from gscupy.keys import make_frame_keys as make_keys, channels

DETECTION_CHANNEL = channels.detection(CAM_ID)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_detections(jpeg_bytes: bytes, detections: list[dict], h: int, w: int) -> bytes:
    """
    Decode JPEG, draw bounding boxes, re-encode.
    Returns annotated JPEG bytes, or original bytes if draw fails.
    """
    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jpeg_bytes

        for det in detections:
            bbox  = det.get("bbox", [])
            label = det.get("label", "")
            score = det.get("score", 0.0)

            if len(bbox) == 4:
                # Normalised [cx, cy, w, h] → pixel coords
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


def archive_event(rdb: redis.Redis, event: dict):
    """
    Pull JPEG + metadata from Redis and write to disk.
    Called every time inference publishes a detection event.

    frame_id carried in the event is used to derive the exact Redis keys
    for this specific frame — no risk of reading a different frame's JPEG.
    """
    frame_id   = event["frame_id"]
    cam_id     = event["cam_id"]
    detections = event.get("detections", [])
    ts_ms      = event.get("ts_ms", int(time.time() * 1000))

    # Derive per-frame keys from frame_id — same schema as reader and worker
    _, jpeg_key, meta_key = make_keys(frame_id)

    # ── Fetch JPEG + metadata in one round-trip ───────────────────────────────
    pipe = rdb.pipeline(transaction=False)
    pipe.get(jpeg_key)
    pipe.hgetall(meta_key)
    jpeg_bytes, meta_raw = pipe.execute()

    if not jpeg_bytes:
        print(f"[archiver] JPEG for {frame_id} already expired — cannot archive")
        return

    meta = {k.decode(): v.decode() for k, v in meta_raw.items()}
    h = int(meta.get("h", 0))
    w = int(meta.get("w", 0))

    # ── Annotate JPEG with detection boxes ───────────────────────────────────
    if detections and h > 0 and w > 0:
        annotated = draw_detections(jpeg_bytes, detections, h, w)
    else:
        annotated = jpeg_bytes

    # ── Write to output directory ─────────────────────────────────────────────
    dt    = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
    stamp = dt.strftime("%Y%m%dT%H%M%S%f")[:-3]   # YYYYMMDDTHHMMSSmmm
    stem  = f"{cam_id}_{stamp}"

    jpeg_path = OUTPUT_DIR / f"{stem}.jpg"
    json_path = OUTPUT_DIR / f"{stem}.json"

    jpeg_path.write_bytes(annotated)
    json_path.write_text(json.dumps({
        "frame_id":   frame_id,
        "cam_id":     cam_id,
        "timestamp":  dt.isoformat(),
        "width":      w,
        "height":     h,
        "detections": detections,
        "worker_id":  event.get("worker_id"),
    }, indent=2))

    print(f"[archiver] wrote {jpeg_path.name}  {len(detections)} detections")

    # ── Optional: push to object storage ─────────────────────────────────────
    # upload_to_s3(jpeg_path, json_path)   # see stub below


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


# ── main loop ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rdb    = redis.Redis(host=REDIS_HOST, port=6379, db=0, socket_keepalive=True)
    pubsub = rdb.pubsub()
    pubsub.subscribe(DETECTION_CHANNEL)

    print(f"[archiver] subscribed to {DETECTION_CHANNEL}  output={OUTPUT_DIR}")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        try:
            event = json.loads(message["data"].decode())
            archive_event(rdb, event)
        except Exception as e:
            print(f"[archiver] error processing event: {e}")
