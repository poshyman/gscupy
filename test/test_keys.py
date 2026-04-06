"""tests/unit/test_keys.py

Unit tests for gscupy.keys — no Redis, no GPU required.
"""

import pytest
from gscupy.keys import make_frame_keys, make_result_key, channels


@pytest.mark.unit
class TestMakeFrameKeys:
    def test_returns_three_keys(self):
        keys = make_frame_keys("cam0:1704067200123")
        assert len(keys) == 3

    def test_ipc_key_format(self):
        ipc, _, _ = make_frame_keys("cam0:1704067200123")
        assert ipc == "frame:ipc:cam0:1704067200123"

    def test_jpeg_key_format(self):
        _, jpeg, _ = make_frame_keys("cam0:1704067200123")
        assert jpeg == "frame:jpeg:cam0:1704067200123"

    def test_meta_key_format(self):
        _, _, meta = make_frame_keys("cam0:1704067200123")
        assert meta == "frame:meta:cam0:1704067200123"

    def test_different_frame_ids_produce_different_keys(self):
        keys_a = make_frame_keys("cam0:1000")
        keys_b = make_frame_keys("cam0:2000")
        assert keys_a != keys_b

    def test_different_cam_ids_produce_different_keys(self):
        keys_a = make_frame_keys("cam0:1000")
        keys_b = make_frame_keys("cam1:1000")
        assert keys_a != keys_b


@pytest.mark.unit
class TestMakeResultKey:
    def test_result_key_format(self):
        key = make_result_key("cam0:1704067200123")
        assert key == "result:cam0:1704067200123"


@pytest.mark.unit
class TestChannels:
    def test_frame_ready_channel(self):
        assert channels.frame_ready("cam0") == "frame_ready:cam0"

    def test_detection_channel(self):
        assert channels.detection("cam0") == "detection:cam0"

    def test_frame_jpeg_ready_channel(self):
        assert channels.frame_jpeg_ready("cam0") == "frame_jpeg_ready:cam0"

    def test_different_cameras_produce_different_channels(self):
        assert channels.frame_ready("cam0") != channels.frame_ready("cam1")
