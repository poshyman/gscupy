"""
inference-worker/worker.py

Memory lifecycle (worker side)
───────────────────────────────
1. Subscribe to FRAME_CHANNEL
2. On notification: GET IPC handle bytes from Redis
3. Reconstruct tensor from IPC handle — zero PCIe transfer, same VRAM as frame-reader
4. Run inference
5. Publish detection event to DETECTION_CHANNEL if threshold met
6. del tensor + torch.cuda.empty_cache() — CUDA ref-count -1
   CUDA frees the allocation only when frame-reader also drops its reference.

IMPORTANT: the IPC handle has a short TTL (FRAME_TTL_MS).
If you cannot import within that window, skip the frame — do not attempt
to reconstruct from a stale handle; the device pointer may be reused.
"""

import os
import pickle
import time
import json

import torch
import redis

# ── env ───────────────────────────────────────────────────────────────────────
REDIS_HOST  = os.environ.get("REDIS_HOST", "localhost")
CAM_ID      = os.environ.get("CAM_ID", "cam0")
WORKER_ID   = os.environ.get("WORKER_ID", "seg-0")
MODEL_TYPE  = os.environ.get("MODEL_TYPE", "maskdino")

# ── redis channels & key schema ──────────────────────────────────────────────
from gscupy.keys import make_frame_keys as make_keys, make_result_key, channels

FRAME_CHANNEL     = channels.frame_ready(CAM_ID)
DETECTION_CHANNEL = channels.detection(CAM_ID)

# ImageNet normalisation constants (allocated once on GPU)
_MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1).half()
_STD  = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1).half()


def load_model(model_type: str):
    """Load whichever model is configured for this worker."""
    if model_type == "maskdino":
        from transformers import (
            MaskFormerForInstanceSegmentation,
            MaskFormerImageProcessor,
        )
        model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-large-coco"
        ).cuda().half().eval()
        return model
    elif model_type == "yolov8":
        from ultralytics import YOLO
        return YOLO("yolov8l.pt")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def preprocess(frame_bgr: torch.Tensor) -> torch.Tensor:
    """
    BGR uint8 (H,W,3) on GPU → normalised fp16 (1,3,H,W) on GPU.
    All ops are VRAM-only.
    """
    # BGR → RGB, channel-first
    x = frame_bgr[:, :, [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).half().div(255.0)
    x.sub_(_MEAN).div_(_STD)
    return x


def import_ipc_tensor(ipc_bytes: bytes) -> torch.Tensor | None:
    """
    Reconstruct a PyTorch tensor from a CUDA IPC handle.
    Returns None if the handle has already been freed.
    """
    try:
        ipc_tuple = pickle.loads(ipc_bytes)
        # torch.Storage._new_shared_cuda reconstructs storage from the IPC descriptor.
        # This increments CUDA's ref-count for the underlying allocation.
        storage = torch.Storage._new_shared_cuda(*ipc_tuple)
        # We stored (H, W, 3) uint8 BGRx — but we wrote .contiguous() BGR so it's (H,W,3).
        # The storage is flat; reshape happens via the meta sidecar.
        return torch.tensor(storage, dtype=torch.uint8)
    except Exception as e:
        print(f"[{WORKER_ID}] IPC import failed (frame may have expired): {e}")
        return None


def run(rdb: redis.Redis, pubsub: redis.client.PubSub, model):
    print(f"[{WORKER_ID}] subscribed to {FRAME_CHANNEL}")

    for message in pubsub.listen():
        if message["type"] != "message":
            continue

        # frame_id is the single value published by the reader.
        # All three per-frame Redis keys are derived from it here —
        # this is the fix for the race condition where the old code
        # always queried the static "frame:ipc:{cam_id}" key and
        # could silently read a different frame's IPC handle.
        frame_id = message["data"].decode()
        ipc_key, jpeg_key, meta_key = make_keys(frame_id)
        t0 = time.perf_counter()

        # ── Fetch IPC handle and metadata atomically ──────────────────────────
        # Both keys are addressed by frame_id, so this can never read
        # IPC data from a different frame than the one we were notified about.
        pipe = rdb.pipeline(transaction=False)
        pipe.get(ipc_key)
        pipe.hgetall(meta_key)
        ipc_bytes, meta_raw = pipe.execute()

        if not ipc_bytes:
            print(f"[{WORKER_ID}] IPC handle expired before import — skipped {frame_id}")
            continue

        meta = {k.decode(): v.decode() for k, v in meta_raw.items()}
        h, w = int(meta.get("h", 0)), int(meta.get("w", 0))
        if h == 0 or w == 0:
            print(f"[{WORKER_ID}] missing shape metadata — skipped {frame_id}")
            continue

        # ── Import CUDA IPC handle → tensor (zero PCIe transfer) ─────────────
        flat_tensor = import_ipc_tensor(ipc_bytes)
        if flat_tensor is None:
            continue

        # Reshape from flat storage to (H, W, 3)
        frame_bgr = flat_tensor.reshape(h, w, 3)

        # ── Preprocess on GPU ─────────────────────────────────────────────────
        x = preprocess(frame_bgr)

        # ── Inference ─────────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(pixel_values=x)

        # ── Parse results (model-specific) ───────────────────────────────────
        detections = parse_outputs(outputs, model, h, w)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[{WORKER_ID}] {frame_id}  {len(detections)} detections  {elapsed_ms:.1f}ms")

        # ── Publish detection event if anything found ─────────────────────────
        if detections:
            event = {
                "frame_id":   frame_id,    # archiver uses this to derive jpeg_key / meta_key
                "cam_id":     CAM_ID,
                "worker_id":  WORKER_ID,
                "ts_ms":      int(time.time() * 1000),
                "detections": detections,
            }
            rdb.publish(DETECTION_CHANNEL, json.dumps(event))

            # Store full result payload (longer TTL so archiver can correlate)
            result_key = make_result_key(frame_id)
            rdb.setex(result_key, 60, json.dumps(event))

        # ── Release IPC reference ─────────────────────────────────────────────
        # Explicitly remove our Python reference, then flush PyTorch's cache.
        # CUDA decrements ref-count; allocation freed when frame-reader also drops.
        del flat_tensor, frame_bgr, x, outputs
        torch.cuda.empty_cache()


def parse_outputs(outputs, model, h: int, w: int) -> list[dict]:
    """
    Stub — replace with your model's actual postprocessing.
    Should return a list of dicts with at least {"label", "score", "bbox"}.
    """
    detections = []
    # Example for a detection model with logits + boxes:
    if hasattr(outputs, "logits"):
        scores = outputs.logits.softmax(-1)[0, :, :-1].max(-1)
        keep = scores.values > 0.5
        for score, box in zip(
            scores.values[keep].cpu().tolist(),
            outputs.pred_boxes[0][keep].cpu().tolist() if hasattr(outputs, "pred_boxes") else [],
        ):
            detections.append({"score": round(score, 3), "bbox": box})
    return detections


if __name__ == "__main__":
    rdb    = redis.Redis(host=REDIS_HOST, port=6379, db=0, socket_keepalive=True)
    pubsub = rdb.pubsub()
    pubsub.subscribe(FRAME_CHANNEL)

    model = load_model(MODEL_TYPE)
    print(f"[{WORKER_ID}] model loaded  device=cuda  dtype=fp16")

    run(rdb, pubsub, model)
