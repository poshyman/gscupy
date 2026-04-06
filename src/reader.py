"""
frame-reader/reader.py

Memory lifecycle
────────────────
1. GstCudaMemory   — owned by GStreamer buffer pool; valid only inside on_new_sample()
2. frame_tensor    — new PyTorch VRAM alloc created by .contiguous(); PyTorch owns it
                     GStreamer buffer is safe to reclaim after this point
3. ipc_handle      — serialised CUDA IPC descriptor; CUDA ref-count +1 per importer
4. jpeg_bytes      — CPU-side JPEG encoded from tensor; stored in Redis with TTL
                     used by frame-archiver without needing GPU access
5. Cleanup         — frame_reader holds its own ref until JPEG_TTL expires,
                     then calls torch.cuda.empty_cache(); each worker does the same
                     after inference; CUDA frees when ref-count reaches zero
"""

import os
import time
import struct
import pickle
import ctypes
import threading

import numpy as np
import torch
import cupy as cp
from torch.utils.dlpack import from_dlpack

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstCuda', '1.0')
from gi.repository import Gst, GstCuda, GLib

import redis
import cv2

# ── env ───────────────────────────────────────────────────────────────────────
RTSP_URL    = os.environ["RTSP_URL"]
CAM_ID      = os.environ.get("CAM_ID", "cam0")
REDIS_HOST  = os.environ.get("REDIS_HOST", "localhost")
FRAME_RATE  = int(os.environ.get("FRAME_RATE", "5"))
FRAME_TTL   = int(os.environ.get("FRAME_TTL_MS", "2000"))   # ms
JPEG_TTL    = int(os.environ.get("JPEG_TTL_S", "30"))        # seconds

# ── redis channels & key schema ──────────────────────────────────────────────
from gscupy.keys import make_frame_keys as make_keys, channels

FRAME_CHANNEL = channels.frame_ready(CAM_ID)
JPEG_CHANNEL  = channels.frame_jpeg_ready(CAM_ID)

# ── libgstcuda ctypes bridge ──────────────────────────────────────────────────
_libgstcuda = ctypes.CDLL("libgstcuda-1.0.so.0")
_libgstcuda.gst_cuda_memory_get_device_ptr.restype  = ctypes.c_uint64
_libgstcuda.gst_cuda_memory_get_device_ptr.argtypes = [ctypes.c_void_p]
_libgstcuda.gst_cuda_memory_get_device_id.restype   = ctypes.c_int
_libgstcuda.gst_cuda_memory_get_device_id.argtypes  = [ctypes.c_void_p]

Gst.init(None)

# ── redis connection (pipeline for atomic multi-command writes) ────────────────
rdb = redis.Redis(host=REDIS_HOST, port=6379, db=0, socket_keepalive=True)

# ── in-flight tensor registry ─────────────────────────────────────────────────
# Keeps Python references alive so CUDA doesn't free VRAM while workers hold IPC handles.
# Entries are removed on a background thread after JPEG_TTL seconds.
_live_tensors: dict[str, torch.Tensor] = {}
_live_lock = threading.Lock()


def _expire_tensor(frame_id: str, ipc_key: str, jpeg_key: str, meta_key: str, delay_s: float):
    """Background thread: release VRAM reference and clean up Redis keys.

    Waits delay_s seconds — long enough for all workers to finish inference
    and release their own IPC references — then:
      1. Drops our Python reference to the tensor (PyTorch ref-count -1).
      2. Calls empty_cache() so PyTorch returns the allocation to CUDA.
         CUDA only calls cudaFree when every importer has also released.
      3. Explicitly deletes the three per-frame Redis keys as a belt-and-
         suspenders measure, in case the TTL hasn't fired yet (e.g. if
         JPEG_TTL was set very long for debugging).
    """
    time.sleep(delay_s)
    with _live_lock:
        tensor = _live_tensors.pop(frame_id, None)
    if tensor is not None:
        del tensor
        torch.cuda.empty_cache()
    # Belt-and-suspenders: delete keys even if TTL hasn't fired yet.
    # Safe to call even if keys are already gone (Redis DEL is idempotent).
    rdb.delete(ipc_key, jpeg_key, meta_key)


# ── DLPack zero-copy bridge ───────────────────────────────────────────────────

def gst_cuda_mem_to_torch(gst_mem, h: int, w: int, c: int = 4) -> torch.Tensor:
    """
    Zero-copy: wrap GstCudaMemory device pointer in a PyTorch tensor via CuPy DLPack.

    The returned tensor is a *view* of GStreamer's buffer — it must be made
    contiguous (.contiguous()) before the callback returns so we hold an
    independent PyTorch allocation.
    """
    raw_ptr    = gst_mem.__gpointer__
    device_ptr = _libgstcuda.gst_cuda_memory_get_device_ptr(raw_ptr)
    device_id  = _libgstcuda.gst_cuda_memory_get_device_id(raw_ptr)

    # CuPy zero-copy view of device memory (UnownedMemory = CuPy does NOT free this)
    cupy_arr = cp.ndarray(
        shape=(h, w, c),
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(device_ptr, h * w * c, owner=None),
            offset=0,
        ),
    )

    # DLPack capsule → PyTorch tensor; still zero-copy, still on device_id
    return from_dlpack(cupy_arr.toDlpack())


# ── GStreamer pipeline ────────────────────────────────────────────────────────

PIPELINE_STR = (
    f"rtspsrc location={RTSP_URL} latency=100 protocols=tcp "
    "! rtph264depay ! h264parse "
    "! nvh264dec "
    "! nvvidconv "
    "! video/x-raw(memory:CUDAMemory),format=BGRx "
    f"! videorate max-rate={FRAME_RATE} "
    "! appsink name=sink emit-signals=True max-buffers=1 drop=True sync=False"
)

pipeline = Gst.parse_launch(PIPELINE_STR)
sink = pipeline.get_by_name("sink")


def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    buf    = sample.get_buffer()
    caps   = sample.get_caps().get_structure(0)

    h = caps.get_value("height")
    w = caps.get_value("width")

    gst_mem = buf.peek_memory(0)

    if not GstCuda.is_cuda_memory(gst_mem):
        print("[frame-reader] WARNING: buffer not in CUDA memory — check pipeline caps")
        return Gst.FlowReturn.OK

    # ── Step 1: zero-copy view of GstCudaMemory ──────────────────────────────
    tensor_bgrx = gst_cuda_mem_to_torch(gst_mem, h, w, c=4)

    # ── Step 2: make contiguous PyTorch-owned VRAM copy ──────────────────────
    # .contiguous() triggers an in-VRAM cudaMemcpy2D — no PCIe transfer.
    # After this line, GStreamer's buffer can be recycled safely.
    frame_bgr = tensor_bgrx[:, :, :3].contiguous()   # (H, W, 3) uint8 BGR on GPU

    # frame_id is the single source of truth that ties all three Redis keys
    # together and is the value published on FRAME_CHANNEL.
    frame_id = f"{CAM_ID}:{int(time.time() * 1000)}"
    ipc_key, jpeg_key, meta_key = make_keys(frame_id)

    # Keep Python ref alive while workers may hold IPC handles
    with _live_lock:
        _live_tensors[frame_id] = frame_bgr

    # ── Step 3: export CUDA IPC handle ───────────────────────────────────────
    # share_memory_() marks the tensor's storage as IPC-shareable and
    # increments CUDA's internal ref-count for this allocation.
    # Workers on the same host/GPU can import this handle to get a pointer
    # into the same VRAM region — still no PCIe transfer.
    frame_bgr.share_memory_()
    ipc_handle = frame_bgr.storage()._share_cuda_()   # returns a pickleable tuple

    # ── Step 4: encode JPEG on GPU (stays on GPU until .tobytes()) ────────────
    # We keep an encoded JPEG in Redis so:
    #   a) the frame-archiver has something to write without needing a GPU
    #   b) we have a human-readable artefact if inference triggers an alert
    #
    # torch → CPU numpy for OpenCV encode (one unavoidable CPU copy, done once)
    frame_cpu = frame_bgr.cpu().numpy()          # (H, W, 3) uint8 BGR, PCIe transfer here
    ok, jpeg_buf = cv2.imencode(".jpg", frame_cpu, [cv2.IMWRITE_JPEG_QUALITY, 85])
    jpeg_bytes = jpeg_buf.tobytes() if ok else b""

    # ── Step 5: atomic Redis write ────────────────────────────────────────────
    meta = {
        "frame_id": frame_id,
        "cam_id":   CAM_ID,
        "h":        h,
        "w":        w,
        "ts_ms":    int(time.time() * 1000),
    }

    pipe = rdb.pipeline(transaction=False)

    # IPC handle stored under its per-frame key — workers derive this from frame_id
    pipe.setex(ipc_key,  FRAME_TTL, pickle.dumps(ipc_handle))

    # JPEG + metadata stored under their per-frame keys — archiver derives from frame_id
    if jpeg_bytes:
        pipe.setex(jpeg_key, JPEG_TTL, jpeg_bytes)
    pipe.hset(meta_key, mapping=meta)
    pipe.expire(meta_key, JPEG_TTL)

    # Publish frame_id — all subscribers derive their required keys via make_keys()
    pipe.publish(FRAME_CHANNEL, frame_id)
    if jpeg_bytes:
        pipe.publish(JPEG_CHANNEL, frame_id)

    pipe.execute()

    # ── Step 6: schedule VRAM release ────────────────────────────────────────
    # Give workers JPEG_TTL seconds to finish with the IPC handle, then drop
    # our reference and clean up the Redis keys.
    t = threading.Thread(
        target=_expire_tensor,
        args=(frame_id, ipc_key, jpeg_key, meta_key, JPEG_TTL),
        daemon=True,
    )
    t.start()

    return Gst.FlowReturn.OK


sink.connect("new-sample", on_new_sample)


# ── GStreamer bus error handler ───────────────────────────────────────────────

def on_bus_message(bus, msg):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"[frame-reader] GST error: {err} | {dbg}")
        GLib.timeout_add_seconds(5, lambda: pipeline.set_state(Gst.State.PLAYING) or False)
    elif msg.type == Gst.MessageType.EOS:
        print("[frame-reader] EOS received — reconnecting in 5s")
        pipeline.set_state(Gst.State.NULL)
        GLib.timeout_add_seconds(5, lambda: pipeline.set_state(Gst.State.PLAYING) or False)


bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message)

pipeline.set_state(Gst.State.PLAYING)
print(f"[frame-reader] live  cam={CAM_ID}  fps={FRAME_RATE}  ipc_ttl={FRAME_TTL}ms  jpeg_ttl={JPEG_TTL}s")

GLib.MainLoop().run()
