"""
gscupy.keys
~~~~~~~~~~~
Single source of truth for every Redis key and channel name used across
frame-reader, inference-worker, and frame-archiver.

Previously each service duplicated a local `make_keys()` function.
All three services now import from here so a key schema change is one edit.

Usage
-----
    from gscupy.keys import make_frame_keys, channels

    ipc_key, jpeg_key, meta_key = make_frame_keys(frame_id)
    rdb.publish(channels.frame_ready(cam_id), frame_id)
"""

from dataclasses import dataclass


# ── Per-frame keys ────────────────────────────────────────────────────────────
# All three keys are derived from a single frame_id so a worker that receives
# a pub/sub notification can look up any of them without a second round-trip
# to discover the key name.

def make_frame_keys(frame_id: str) -> tuple[str, str, str]:
    """Return (ipc_key, jpeg_key, meta_key) for a given frame_id.

    Args:
        frame_id: Unique frame identifier, format ``"{cam_id}:{ts_ms}"``.
                  e.g. ``"cam0:1704067200123"``

    Returns:
        Tuple of three Redis key strings:
        - ``frame:ipc:{frame_id}``  — pickled CUDA IPC handle, short TTL
        - ``frame:jpeg:{frame_id}`` — JPEG bytes, long TTL
        - ``frame:meta:{frame_id}`` — hash of shape/timestamp metadata, long TTL
    """
    return (
        f"frame:ipc:{frame_id}",
        f"frame:jpeg:{frame_id}",
        f"frame:meta:{frame_id}",
    )


def make_result_key(frame_id: str) -> str:
    """Redis key for inference results associated with a frame.

    Args:
        frame_id: Same frame_id used in :func:`make_frame_keys`.
    """
    return f"result:{frame_id}"


# ── Pub/sub channels ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Channels:
    """Namespaced pub/sub channel name factory.

    All channel names are per-camera so multiple cameras can run on the
    same Redis instance without cross-talk.

    Example::

        ch = channels  # module-level singleton
        rdb.publish(ch.frame_ready("cam0"), frame_id)
        rdb.subscribe(ch.detection("cam0"))
    """

    @staticmethod
    def frame_ready(cam_id: str) -> str:
        """Published by frame-reader; consumed by inference workers."""
        return f"frame_ready:{cam_id}"

    @staticmethod
    def frame_jpeg_ready(cam_id: str) -> str:
        """Published by frame-reader alongside frame_ready; consumed by archiver."""
        return f"frame_jpeg_ready:{cam_id}"

    @staticmethod
    def detection(cam_id: str) -> str:
        """Published by inference workers when detections are found; consumed by archiver."""
        return f"detection:{cam_id}"


channels = _Channels()


# ── TTL constants (seconds unless noted) ──────────────────────────────────────
# Services override these via environment variables; these are the defaults.

DEFAULT_IPC_TTL_MS: int = 2000   # milliseconds — IPC handle expires quickly
DEFAULT_JPEG_TTL_S: int = 30     # seconds      — JPEG kept for archiver
DEFAULT_RESULT_TTL_S: int = 60   # seconds      — inference results
