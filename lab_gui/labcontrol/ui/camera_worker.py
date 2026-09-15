"""Interruptible Qt camera acquisition worker."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot

from ..controller import LabController


class CameraAcquisitionWorker(QObject):
    frame_ready = Signal(object)
    error = Signal(str)
    stopped = Signal()

    def __init__(self, controller: LabController, *, target_fps: float = 15.0):
        super().__init__()
        self.controller = controller
        self.target_fps = max(0.5, float(target_fps))
        self._stop = threading.Event()

    @Slot()
    def run(self) -> None:
        self._stop.clear()
        interval = 1.0 / self.target_fps
        try:
            self.controller.start_camera()
            while not self._stop.is_set() and not QThread.currentThread().isInterruptionRequested():
                try:
                    frame = self.controller.acquire_frame(fresh=False, timeout_s=max(0.2, interval * 4))
                    self.frame_ready.emit(frame)
                except Exception as exc:
                    self.error.emit(str(exc))
                    break
                self._stop.wait(interval)
        finally:
            try:
                self.controller.stop_camera()
            except Exception as exc:
                self.error.emit(str(exc))
            self.stopped.emit()

    def request_stop(self) -> None:
        """Thread-safe; may be called directly from the GUI thread."""

        self._stop.set()


def stop_worker_thread(
    thread: QThread | None,
    worker: CameraAcquisitionWorker | None,
    *,
    timeout_ms: int = 3000,
) -> bool:
    if thread is None:
        return True
    if worker is not None:
        worker.request_stop()
    thread.requestInterruption()
    if thread.isRunning() and not thread.wait(timeout_ms):
        return False
    return True
