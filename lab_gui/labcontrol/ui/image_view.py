"""Zoomable quantitative-image display with state-aware overlays."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..devices.camera import CameraFrame
from ..metrics import BeamMetrics


def render_preview(
    raw: np.ndarray,
    *,
    colour: str = "false colour",
    scale: str = "percentile",
    gamma: float = 1.0,
) -> np.ndarray:
    """Create display pixels without modifying or returning analysis data."""

    source = np.asarray(raw, dtype=np.float64)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("Preview requires a finite 2-D matrix.")
    if scale == "full range":
        low, high = float(np.min(source)), float(np.max(source))
    else:
        low, high = (float(v) for v in np.percentile(source, (1.0, 99.8)))
    unit = np.clip((source - low) / max(high - low, 1e-12), 0.0, 1.0)
    if scale == "log":
        unit = np.log1p(100.0 * unit) / np.log(101.0)
    unit = np.power(unit, 1.0 / max(float(gamma), 0.05))
    if colour == "grayscale":
        return np.asarray(np.rint(unit * 255.0), dtype=np.uint8)
    # Compact perceptual-ish blue/cyan/yellow/red map; display only.
    red = np.clip(1.8 * unit - 0.45, 0.0, 1.0)
    green = np.clip(1.8 - np.abs(4.0 * unit - 2.0), 0.0, 1.0)
    blue = np.clip(1.35 - 1.8 * unit, 0.0, 1.0)
    return np.asarray(np.rint(np.stack((red, green, blue), axis=-1) * 255.0), dtype=np.uint8)


class QuantitativeImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#071017"))
        self.setMinimumSize(520, 420)
        self.raw_frame: np.ndarray | None = None
        self.frame: CameraFrame | None = None
        self.metrics: BeamMetrics | None = None
        self._overlay_items = []
        self._first_image = True

    def set_quantitative_frame(
        self,
        frame: CameraFrame,
        metrics: BeamMetrics | None,
        *,
        colour: str,
        scale: str,
        gamma: float,
        show_centre: bool = True,
        show_ring: bool = True,
        show_roi: bool = True,
    ) -> None:
        self.frame = frame
        # CameraFrame owns a stable contiguous matrix.  Retaining that reference
        # avoids another 32 MB copy for every 2048×2048 Beamage frame and view.
        self.raw_frame = frame.data
        self.metrics = metrics
        source = frame.data
        if not frame.metadata.get("quantitative_valid", True):
            stride = max(1, int(np.ceil(max(source.shape) / 1200.0)))
            source = source[::stride, ::stride]
        preview = render_preview(source, colour=colour, scale=scale, gamma=gamma)
        self.set_preview_array(
            preview,
            metrics=metrics,
            show_centre=show_centre,
            show_ring=show_ring,
            show_roi=show_roi,
            clipped=bool(metrics and metrics.values.get("clipped")),
        )

    def set_preview_array(
        self,
        preview: np.ndarray,
        *,
        metrics: BeamMetrics | None = None,
        show_centre: bool = True,
        show_ring: bool = True,
        show_roi: bool = True,
        clipped: bool = False,
    ) -> None:
        array = np.ascontiguousarray(preview)
        if array.ndim == 2:
            height, width = array.shape
            image = QImage(array.data, width, height, width, QImage.Format_Grayscale8).copy()
        elif array.ndim == 3 and array.shape[2] == 3:
            height, width, _ = array.shape
            image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        else:
            raise ValueError("Preview must be grayscale or RGB.")
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(0, 0, width, height)
        for item in self._overlay_items:
            self.scene().removeItem(item)
        self._overlay_items.clear()
        if metrics is not None:
            cy, cx = metrics.centre_yx_px
            radius = float(metrics.values.get("principal_ring_radius_px") or 0.0)
            if show_roi:
                roi_radius = max(radius * 1.7, min(width, height) * 0.12)
                self._overlay_items.append(
                    self.scene().addEllipse(
                        cx - roi_radius,
                        cy - roi_radius,
                        2 * roi_radius,
                        2 * roi_radius,
                        QPen(QColor("#69d2ff"), 1.2, Qt.DashLine),
                    )
                )
            if show_ring and radius > 0:
                self._overlay_items.append(
                    self.scene().addEllipse(
                        cx - radius,
                        cy - radius,
                        2 * radius,
                        2 * radius,
                        QPen(QColor("#f6c85f"), 2.0),
                    )
                )
                core = radius * 0.35
                self._overlay_items.append(
                    self.scene().addEllipse(
                        cx - core,
                        cy - core,
                        2 * core,
                        2 * core,
                        QPen(QColor("#e995ff"), 1.2, Qt.DotLine),
                    )
                )
            if show_centre:
                pen = QPen(QColor("#59f2b1"), 1.6)
                arm = max(8.0, radius * 0.25)
                self._overlay_items.extend(
                    [
                        self.scene().addLine(cx - arm, cy, cx + arm, cy, pen),
                        self.scene().addLine(cx, cy - arm, cx, cy + arm, pen),
                    ]
                )
        if clipped:
            self._overlay_items.append(
                self.scene().addRect(1, 1, width - 2, height - 2, QPen(QColor("#ff5d73"), 4.0))
            )
        if self._first_image:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self._first_image = False

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)

    def reset_zoom(self) -> None:
        self.resetTransform()
        if not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
