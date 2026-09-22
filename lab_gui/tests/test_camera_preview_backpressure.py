"""Camera previews must not queue unlimited full-resolution matrices."""

from __future__ import annotations

import threading
import time

from PySide6.QtWidgets import QApplication

from labcontrol.ui.camera_worker import CameraAcquisitionWorker
from labcontrol.ui.virtual_advanced import _VirtualTaskWorker


class _FastCameraController:
    def __init__(self) -> None:
        self.acquired = 0
        self.started = 0
        self.stopped = 0

    def start_camera(self) -> None:
        self.started += 1

    def acquire_frame(self, **_kwargs):
        self.acquired += 1
        return self.acquired

    def stop_camera(self) -> None:
        self.stopped += 1


def test_live_worker_waits_for_gui_acknowledgement() -> None:
    app = QApplication.instance() or QApplication([])
    controller = _FastCameraController()
    worker = CameraAcquisitionWorker(controller, target_fps=100)
    delivered = []
    worker.frame_ready.connect(lambda _source, frame: delivered.append(frame))
    thread = threading.Thread(target=worker.run)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while controller.acquired < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert controller.acquired == 1
        time.sleep(0.15)
        assert controller.acquired == 1, "a queued preview must block further acquisition"
        app.processEvents()
        worker.acknowledge_frame()
        deadline = time.monotonic() + 2
        while controller.acquired < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert controller.acquired == 2
    finally:
        worker.request_stop()
        thread.join(2)
        app.processEvents()
    assert not thread.is_alive()
    assert controller.started == controller.stopped == 1
    assert delivered


def test_virtual_probe_previews_are_coalesced() -> None:
    app = QApplication.instance() or QApplication([])
    delivered = []

    def probe(**callbacks):
        for index in range(100):
            callbacks["frame_callback"](index)
        return "done"

    worker = _VirtualTaskWorker(probe)
    worker.frame.connect(lambda _source, frame: delivered.append(frame))
    worker.run()
    app.processEvents()
    assert delivered == [0]
    worker.acknowledge_frame()
    worker._last_preview_utc = 0.0
    worker._emit_preview(101)
    app.processEvents()
    assert delivered == [0, 101]
