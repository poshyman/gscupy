"""tests/integration/test_archiver.py

Integration tests for frame_archiver.archiver.
Requires a Redis instance at REDIS_HOST:6379 (provided by the dev container
network — redis is a named service on cv_net).

Run with:
    make test-integration
or:
    pytest tests/integration/ -m integration
"""

import json
import time
import pathlib
import numpy as np
import cv2
import pytest
import redis as redis_lib

from gscupy.keys import make_frame_keys, channels
from frame_archiver.archiver import archive_event, draw_detections


REDIS_HOST = "redis"   # service name on cv_net; use "localhost" if running outside Docker


@pytest.fixture(scope="module")
def rdb():
    r = redis_lib.Redis(host=REDIS_HOST, port=6379, db=15)  # db=15 = test isolation
    yield r
    r.flushdb()   # clean up all test keys after the module


@pytest.fixture
def sample_jpeg() -> bytes:
    """Create a minimal valid JPEG in memory."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:200, 100:300] = (0, 255, 0)   # green rectangle
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def populated_frame(rdb, sample_jpeg) -> str:
    """Write a frame's JPEG + metadata into Redis, return its frame_id."""
    frame_id = f"cam0:{int(time.time() * 1000)}"
    _, jpeg_key, meta_key = make_frame_keys(frame_id)

    pipe = rdb.pipeline(transaction=False)
    pipe.setex(jpeg_key, 30, sample_jpeg)
    pipe.hset(meta_key, mapping={"frame_id": frame_id, "cam_id": "cam0", "h": 480, "w": 640})
    pipe.expire(meta_key, 30)
    pipe.execute()

    return frame_id


@pytest.mark.integration
def test_archive_event_writes_jpeg(rdb, populated_frame, tmp_path):
    event = {
        "frame_id":  populated_frame,
        "cam_id":    "cam0",
        "worker_id": "test",
        "ts_ms":     int(time.time() * 1000),
        "detections": [],
    }
    archive_event(rdb, event, output_dir=tmp_path)

    jpegs = list(tmp_path.glob("*.jpg"))
    assert len(jpegs) == 1
    assert jpegs[0].stat().st_size > 0


@pytest.mark.integration
def test_archive_event_writes_json_sidecar(rdb, populated_frame, tmp_path):
    event = {
        "frame_id":  populated_frame,
        "cam_id":    "cam0",
        "worker_id": "test",
        "ts_ms":     int(time.time() * 1000),
        "detections": [{"label": "person", "score": 0.92, "bbox": [0.5, 0.5, 0.2, 0.3]}],
    }
    archive_event(rdb, event, output_dir=tmp_path)

    jsons = list(tmp_path.glob("*.json"))
    assert len(jsons) == 1
    data = json.loads(jsons[0].read_text())
    assert data["frame_id"] == populated_frame
    assert len(data["detections"]) == 1
    assert data["detections"][0]["label"] == "person"


@pytest.mark.integration
def test_archive_event_handles_expired_jpeg(rdb, tmp_path):
    """If the JPEG has already expired, archive_event should log and return gracefully."""
    event = {
        "frame_id":  "cam0:0",   # frame that was never written
        "cam_id":    "cam0",
        "worker_id": "test",
        "ts_ms":     int(time.time() * 1000),
        "detections": [],
    }
    # Should not raise
    archive_event(rdb, event, output_dir=tmp_path)
    assert list(tmp_path.glob("*")) == []


@pytest.mark.unit
class TestDrawDetections:
    def test_returns_bytes(self, sample_jpeg):
        result = draw_detections(sample_jpeg, [], h=480, w=640)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_draws_bbox(self, sample_jpeg):
        dets = [{"label": "car", "score": 0.8, "bbox": [0.5, 0.5, 0.4, 0.3]}]
        result = draw_detections(sample_jpeg, dets, h=480, w=640)
        # Result should be a valid JPEG
        arr = np.frombuffer(result, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape == (480, 640, 3)

    def test_invalid_jpeg_returns_original(self):
        bad_bytes = b"not a jpeg"
        result = draw_detections(bad_bytes, [], h=480, w=640)
        assert result == bad_bytes
