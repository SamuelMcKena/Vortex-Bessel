"""Zoomable full-resolution camera-image display with state-aware overlays."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..devices.camera import CameraFrame
from ..metrics import BeamMetrics


_COLOUR_ANCHORS: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    # Smooth scientific display maps.  They affect preview pixels only.
    "inferno": (
        (0.00, (0, 0, 4)),
        (0.18, (40, 11, 84)),
        (0.38, (101, 21, 110)),
        (0.58, (159, 42, 99)),
        (0.76, (219, 92, 50)),
        (0.90, (249, 164, 36)),
        (1.00, (252, 255, 164)),
    ),
    "turbo": (
        (0.00, (48, 18, 59)),
        (0.15, (42, 111, 219)),
        (0.34, (30, 194, 177)),
        (0.52, (103, 248, 87)),
        (0.70, (236, 211, 52)),
        (0.86, (244, 94, 31)),
        (1.00, (122, 4, 3)),
    ),
    "viridis": (
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ),
    "gentec-like": (
        (0.00, (0, 0, 0)),
        (0.15, (0, 0, 125)),
        (0.32, (0, 126, 255)),
        (0.48, (0, 235, 181)),
        (0.64, (92, 255, 45)),
        (0.78, (255, 239, 0)),
        (0.90, (255, 95, 0)),
        (1.00, (255, 255, 255)),
    ),
}


def _apply_colour_map(unit: np.ndarray, colour: str) -> np.ndarray:
    name = colour.strip().lower()
    if name == "grayscale":
        return np.asarray(np.rint(unit * 255.0), dtype=np.uint8)
    # Keep old presets/tests meaningful while replacing the old ad-hoc map.
    if name == "false colour":
        name = "inferno"
    anchors = _COLOUR_ANCHORS.get(name, _COLOUR_ANCHORS["inferno"])
    positions = np.asarray([item[0] for item in anchors], dtype=np.float64)
    colours = np.asarray([item[1] for item in anchors], dtype=np.float64)
    channels = [np.interp(unit.ravel(), positions, colours[:, i]).reshape(unit.shape) for i in range(3)]
    return np.asarray(np.rint(np.stack(channels, axis=-1)), dtype=np.uint8)


def render_preview(
    raw: np.ndarray,
    *,
    colour: str = "inferno",
    scale: str = "percentile",
    gamma: float = 1.0,
    full_scale: float | None = None,
) -> np.ndarray:
    """Create display pixels without cropping, resampling or mutating camera data."""

    source = np.asarray(raw, dtype=np.float64)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("Preview requires a finite 2-D matrix.")

    mode = scale.strip().lower()
    if mode in {"sensor range", "sensor"} and full_scale is not None and full_scale > 0:
        low, high = 0.0, float(full_scale)
    elif mode in {"full range", "min/max"}:
        low, high = float(np.min(source)), float(np.max(source))
    else:
        # Use nearly the complete histogram while rejecting a few hot/dead pixels.
        low, high = (float(v) for v in np.percentile(source, (0.1, 99.95)))
    unit = np.clip((source - low) / max(high - low, 1e-12), 0.0, 1.0)
    if mode == "log":
        unit = np.log1p(300.0 * unit) / np.log(301.0)
    unit = np.power(unit, 1.0 / max(float(gamma), 0.05))
    return _apply_colour_map(unit, colour)


class QuantitativeImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
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
        self.raw_frame = frame.data
        self.metrics = metrics
        # Do not decimate Beamage's 2048x2048 frame.  Zoom is a view transform,
        # so every acquired pixel remains available when inspecting the beam.
        preview = render_preview(
            frame.data,
            colour=colour,
            scale=scale,
            gamma=gamma,
            full_scale=frame.full_scale,
        )
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
            self.fit_full_frame()
            self._first_image = False

    def signal_rect(self, *, padding: float = 0.35) -> QRectF | None:
        """Estimate a display-only beam box; never changes the stored/analyzed frame."""

        if self.raw_frame is None:
            return None
        source = np.asarray(self.raw_frame, dtype=np.float64)
        low, high = (float(v) for v in np.percentile(source, (50.0, 99.95)))
        threshold = low + 0.16 * max(high - low, 0.0)
        ys, xs = np.nonzero(source >= threshold)
        if xs.size < 4:
            return None
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        width = max(8.0, x1 - x0 + 1.0)
        height = max(8.0, y1 - y0 + 1.0)
        extra_x = max(12.0, width * float(padding))
        extra_y = max(12.0, height * float(padding))
        rect = QRectF(x0 - extra_x, y0 - extra_y, width + 2 * extra_x, height + 2 * extra_y)
        return rect.intersected(self.scene().sceneRect())

    def fit_signal(self) -> None:
        rect = self.signal_rect()
        if rect is not None and not rect.isEmpty():
            self.resetTransform()
            self.fitInView(rect, Qt.KeepAspectRatio)

    def fit_full_frame(self) -> None:
        self.resetTransform()
        if not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def zoom_in(self) -> None:
        self.scale(1.35, 1.35)

    def zoom_out(self) -> None:
        self.scale(1.0 / 1.35, 1.0 / 1.35)

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):  # noqa: N802 - Qt API
        self.fit_signal()
        event.accept()

    def reset_zoom(self) -> None:
        self.fit_full_frame()
