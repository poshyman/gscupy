"""
frame-reader/reader.py

Responsibilities
────────────────
Decode RTSP/file → GPU tensor → CUDA IPC handle → Redis.
Nothing CPU-side happens here. All byte transfers and JPEG encoding
are owned by frame-archiver, which subscribes to FRAME_CHANNEL.

Memory lifecycle
────────────────
1. GstCudaMemory   — owned by GStreamer buffer pool; valid only inside on_new_sample()
2. frame_tensor    — new PyTorch VRAM alloc created by .contiguous(); PyTorch owns it
                     GStreamer buffer is safe to reclaim after this point
3. ipc_handle      — serialised CUDA IPC descriptor; CUDA ref-count +1 per importer
                     (inference-worker and frame-archiver both import it)
4. Cleanup         — frame_reader holds its own ref until IPC_HOLD_S expires,
                     then calls torch.cuda.empty_cache(); each importer does the same
                     after use; CUDA frees when ref-count reaches zero
"""

import os
import time
import pickle
import ctypes
import threading

import torch
import cupy as cp
from torch.utils.dlpack import from_dlpack

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstCuda", "1.0")
from gi.repository import Gst, GstCuda, GLib

import redis

from cv_pipeline.keys import make_frame_keys as make_keys, channels

# ── env ───────────────────────────────────────────────────────────────────────
RTSP_URL = os.environ["RTSP_URL"]
CAM_ID = os.environ.get("CAM_ID", "cam0")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
FRAME_RATE = int(os.environ.get("FRAME_RATE", "5"))
FRAME_TTL = int(os.environ.get("FRAME_TTL_MS", "2000"))  # IPC key TTL in Redis (ms)
# How long we hold our own Python reference to the tensor.
# Must be >= longest expected inference + archival time so importers
# can finish before CUDA is allowed to free the allocation.
IPC_HOLD_S = int(os.environ.get("IPC_HOLD_S", "30"))

CODEC = os.environ.get("CODEC", "h264")
SOURCE = os.environ.get("SOURCE", "rtsp")
VIDEO_PATH = os.environ.get("VIDEO_PATH", "")
LOOP = os.environ.get("LOOP", "false").lower() == "true"

# ── redis channels ────────────────────────────────────────────────────────────
FRAME_CHANNEL = channels.frame_ready(CAM_ID)

# ── libgstcuda ctypes bridge ──────────────────────────────────────────────────
_libgstcuda = ctypes.CDLL("libgstcuda-1.0.so.0")
_libgstcuda.gst_cuda_memory_get_device_ptr.restype = ctypes.c_uint64
_libgstcuda.gst_cuda_memory_get_device_ptr.argtypes = [ctypes.c_void_p]
_libgstcuda.gst_cuda_memory_get_device_id.restype = ctypes.c_int
_libgstcuda.gst_cuda_memory_get_device_id.argtypes = [ctypes.c_void_p]

Gst.init(None)

rdb = redis.Redis(host=REDIS_HOST, port=6379, db=0, socket_keepalive=True)

# ── in-flight tensor registry ─────────────────────────────────────────────────
_live_tensors: dict[str, torch.Tensor] = {}
_live_lock = threading.Lock()


def _expire_tensor(frame_id: str, ipc_key: str, meta_key: str, delay_s: float):
    """Release our VRAM reference and clean up Redis keys after delay_s seconds.

    CUDA only calls cudaFree when every importer (inference-worker,
    frame-archiver) has also released their reference. This thread just
    drops the reader's own share — it does not race with importers.
    """
    time.sleep(delay_s)
    with _live_lock:
        tensor = _live_tensors.pop(frame_id, None)
    if tensor is not None:
        del tensor
        torch.cuda.empty_cache()
    rdb.delete(ipc_key, meta_key)


# ── DLPack zero-copy bridge ───────────────────────────────────────────────────


def gst_cuda_mem_to_torch(gst_mem, h: int, w: int, c: int = 4) -> torch.Tensor:
    """Zero-copy wrap of GstCudaMemory device pointer as a PyTorch tensor.

    Returns a *view* of GStreamer's buffer — caller must call .contiguous()
    before returning from on_new_sample() to get an independent allocation.
    """
    raw_ptr = gst_mem.__gpointer__
    device_ptr = _libgstcuda.gst_cuda_memory_get_device_ptr(raw_ptr)

    cupy_arr = cp.ndarray(
        shape=(h, w, c),
        dtype=cp.uint8,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(device_ptr, h * w * c, owner=None),
            offset=0,
        ),
    )
    return from_dlpack(cupy_arr.toDlpack())


# ── GStreamer pipeline ────────────────────────────────────────────────────────

_DEPAY = {"h264": "rtph264depay ! h264parse", "h265": "rtph265depay ! h265parse"}
_DECODER = {"h264": "nvh264dec", "h265": "nvh265dec"}

if SOURCE == "rtsp":
    _source = (
        f"rtspsrc location={RTSP_URL} latency=100 protocols=tcp "
        f"! {_DEPAY[CODEC]} ! {_DECODER[CODEC]}"
    )
else:
    _source = (
        f"filesrc location={VIDEO_PATH} "
        f"! qtdemux ! {CODEC}parse ! {_DECODER[CODEC]}"
    )

PIPELINE_STR = (
    f"{_source} "
    "! nvvidconv "
    "! video/x-raw(memory:CUDAMemory),format=BGRx "
    f"! videorate max-rate={FRAME_RATE} "
    "! appsink name=sink emit-signals=True max-buffers=1 drop=True sync=False"
)

pipeline = Gst.parse_launch(PIPELINE_STR)
sink = pipeline.get_by_name("sink")


def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    buf = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)

    h = caps.get_value("height")
    w = caps.get_value("width")

    gst_mem = buf.peek_memory(0)

    if not GstCuda.is_cuda_memory(gst_mem):
        print("[frame-reader] WARNING: buffer not in CUDA memory — check pipeline caps")
        return Gst.FlowReturn.OK

    # ── Step 1: zero-copy view → contiguous PyTorch-owned VRAM allocation ────
    # .contiguous() is an in-VRAM cudaMemcpy2D — no PCIe transfer.
    # After this, GStreamer's buffer can be recycled immediately.
    tensor_bgrx = gst_cuda_mem_to_torch(gst_mem, h, w, c=4)
    frame_bgr = tensor_bgrx[:, :, :3].contiguous()  # (H, W, 3) uint8 BGR

    frame_id = f"{CAM_ID}:{int(time.time() * 1000)}"
    ipc_key, _, meta_key = make_keys(frame_id)  # jpeg_key written by archiver, not us

    with _live_lock:
        _live_tensors[frame_id] = frame_bgr

    # ── Step 2: export CUDA IPC handle ───────────────────────────────────────
    frame_bgr.share_memory_()
    ipc_handle = frame_bgr.storage()._share_cuda_()

    # ── Step 3: write to Redis and notify subscribers ─────────────────────────
    # Both inference-worker and frame-archiver subscribe to FRAME_CHANNEL
    # and each independently import the IPC handle for their own purpose.
    meta = {
        "frame_id": frame_id,
        "cam_id": CAM_ID,
        "h": h,
        "w": w,
        "ts_ms": int(time.time() * 1000),
    }

    pipe = rdb.pipeline(transaction=False)
    pipe.setex(ipc_key, FRAME_TTL, pickle.dumps(ipc_handle))
    pipe.hset(meta_key, mapping=meta)
    pipe.expire(meta_key, IPC_HOLD_S)
    pipe.publish(FRAME_CHANNEL, frame_id)
    pipe.execute()

    # ── Step 4: schedule VRAM release ────────────────────────────────────────
    threading.Thread(
        target=_expire_tensor,
        args=(frame_id, ipc_key, meta_key, IPC_HOLD_S),
        daemon=True,
    ).start()

    return Gst.FlowReturn.OK


sink.connect("new-sample", on_new_sample)


# ── GStreamer bus handler ─────────────────────────────────────────────────────


def on_bus_message(bus, msg):
    if msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        print(f"[frame-reader] GST error: {err} | {dbg}")
        GLib.timeout_add_seconds(
            5, lambda: pipeline.set_state(Gst.State.PLAYING) or False
        )
    elif msg.type == Gst.MessageType.EOS:
        if LOOP:
            pipeline.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                0,
            )
        else:
            print("[frame-reader] EOS — stopping")
            GLib.MainLoop().quit()


bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_bus_message)

pipeline.set_state(Gst.State.PLAYING)
print(
    f"[frame-reader] live  cam={CAM_ID}  source={SOURCE}  "
    f"codec={CODEC}  fps={FRAME_RATE}  "
    f"ipc_ttl={FRAME_TTL}ms  hold={IPC_HOLD_S}s"
)

GLib.MainLoop().run()
