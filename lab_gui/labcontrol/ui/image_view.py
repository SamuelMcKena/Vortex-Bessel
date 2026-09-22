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


_LUT_CACHE: dict[str, np.ndarray] = {}


def _colour_lut(name: str) -> np.ndarray:
    cached = _LUT_CACHE.get(name)
    if cached is not None:
        return cached
    anchors = _COLOUR_ANCHORS.get(name, _COLOUR_ANCHORS["inferno"])
    positions = np.asarray([item[0] for item in anchors], dtype=np.float64) * 255.0
    colours = np.asarray([item[1] for item in anchors], dtype=np.float64)
    x = np.arange(256, dtype=np.float64)
    lut = np.stack([np.interp(x, positions, colours[:, i]) for i in range(3)], axis=1)
    lut = np.asarray(np.rint(lut), dtype=np.uint8)
    _LUT_CACHE[name] = lut
    return lut


def _apply_colour_map(unit: np.ndarray, colour: str) -> np.ndarray:
    name = colour.strip().lower()
    indices = np.asarray(np.rint(unit * 255.0), dtype=np.uint8)
    if name == "grayscale":
        return indices
    # Keep old presets/tests meaningful while replacing the old ad-hoc map.
    if name == "false colour":
        name = "inferno"
    return _colour_lut(name)[indices]


def render_preview(
    raw: np.ndarray,
    *,
    colour: str = "inferno",
    scale: str = "percentile",
    gamma: float = 1.0,
    full_scale: float | None = None,
) -> np.ndarray:
    """Create display pixels without cropping, resampling or mutating camera data."""

    # Camera data remains untouched; display arithmetic uses half the memory
    # of float64, important for a full 2048/4096-square virtual preview.
    source = np.asarray(raw, dtype=np.float32)
    if source.ndim != 2 or not np.isfinite(source).all():
        raise ValueError("Preview requires a finite 2-D matrix.")

    mode = scale.strip().lower()
    # Estimate display levels on a sparse view for large sensor frames.  Every
    # source pixel is still mapped into the final full-resolution preview.
    level_sample = source[::4, ::4] if source.size > 1_000_000 else source
    if mode in {"sensor range", "sensor", "sensor log"} and full_scale is not None and full_scale > 0:
        low, high = 0.0, float(full_scale)
    elif mode in {"full range", "min/max"}:
        # Every pixel: a thin ring can fall between the samples of a sparse view.
        low, high = float(np.min(source)), float(np.max(source))
    else:
        # Use nearly the complete histogram while rejecting a few hot/dead pixels.
        # The upper level must come from *every* pixel: a q=5 Bessel core is a
        # few pixels wide, so a sparse sample skips it, the colour scale tops out
        # on the dim envelope and the ring is drawn saturated and invisible.
        low = float(np.percentile(level_sample, 0.1))
        flat = source.ravel()
        # Reject only a handful of hot pixels.  A fixed fraction (0.05 % is
        # ~2000 pixels on a 4M sensor) is larger than a whole q=20 ring, so the
        # ring and its neighbouring fringes all clipped into a fat white disc.
        top = max(3, min(25, int(round(flat.size * 0.0005))))
        high = float(np.partition(flat, flat.size - top)[flat.size - top])
    unit = (source - low) / max(high - low, 1e-12)
    np.clip(unit, 0.0, 1.0, out=unit)
    if mode in {"log", "sensor log"}:
        np.multiply(unit, 300.0, out=unit)
        np.log1p(unit, out=unit)
        unit /= np.log(301.0)
    np.power(unit, 1.0 / max(float(gamma), 0.05), out=unit)
    return _apply_colour_map(unit, colour)


def _cosmetic(pen: QPen) -> QPen:
    """Overlay lines keep their screen width at any zoom instead of covering camera pixels."""
    pen.setCosmetic(True)
    return pen


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
        # Deliberately small: pages that have room give their view a larger
        # minimum themselves.  A large default made the camera pages taller
        # than a laptop screen and forced them to scroll.
        self.setMinimumSize(300, 240)
        self.raw_frame: np.ndarray | None = None
        self.frame: CameraFrame | None = None
        self.metrics: BeamMetrics | None = None
        self._overlay_items = []
        self._first_image = True
        # Display level of detail.  Drawing a 2048-pixel frame into a ~900-pixel
        # view with nearest-neighbour sampling invents a moiré that is not in the
        # camera data, which is easily mistaken for real sensor aliasing.  When
        # zoomed out, screen pixels therefore show the *average* of the camera
        # pixels they cover; at 1:1 or closer every camera pixel is drawn as is.
        self._display_rgb: np.ndarray | None = None
        self._display_factor = 0
        self._pixel_scale = 1.0

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
        display_data: np.ndarray | None = None,
        pixel_scale: float = 1.0,
        display_full_scale: float | None = None,
    ) -> None:
        """Show a camera frame, or other data laid over the same camera-pixel coordinates.

        ``display_data`` (e.g. the noise-free native model intensity) is drawn
        with each of its samples ``pixel_scale`` camera pixels wide, so zoom,
        fit and overlays stay registered to the camera frame.
        """

        self.frame = frame
        self.raw_frame = frame.data
        self.metrics = metrics
        # Do not decimate Beamage's 2048x2048 frame.  Zoom is a view transform,
        # so every acquired pixel remains available when inspecting the beam.
        preview = render_preview(
            frame.data if display_data is None else display_data,
            colour=colour,
            scale=scale,
            gamma=gamma,
            full_scale=frame.full_scale if display_data is None else display_full_scale,
        )
        self.set_preview_array(
            preview,
            metrics=metrics,
            show_centre=show_centre,
            show_ring=show_ring,
            show_roi=show_roi,
            clipped=bool(metrics and metrics.values.get("clipped")),
            pixel_scale=pixel_scale if display_data is not None else 1.0,
        )

    def release_frame(self) -> None:
        """Free a hidden preview's pixels without changing display controls."""

        self.frame = None
        self.raw_frame = None
        self.metrics = None
        self._pixmap_item.setPixmap(QPixmap())
        self._display_rgb = None
        self._display_factor = 0
        for item in self._overlay_items:
            self.scene().removeItem(item)
        self._overlay_items.clear()
        self.scene().setSceneRect(QRectF())

    def set_preview_array(
        self,
        preview: np.ndarray,
        *,
        metrics: BeamMetrics | None = None,
        show_centre: bool = True,
        show_ring: bool = True,
        show_roi: bool = True,
        clipped: bool = False,
        pixel_scale: float = 1.0,
    ) -> None:
        array = np.ascontiguousarray(preview)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if not (array.ndim == 3 and array.shape[2] == 3):
            raise ValueError("Preview must be grayscale or RGB.")
        height, width, _ = array.shape
        self._display_rgb = array
        self._display_factor = 0
        self._pixel_scale = float(pixel_scale)
        self.scene().setSceneRect(0, 0, width * self._pixel_scale, height * self._pixel_scale)
        self._refresh_display_detail()
        for item in self._overlay_items:
            self.scene().removeItem(item)
        self._overlay_items.clear()
        if metrics is not None:
            cy, cx = metrics.centre_yx_px
            radius = float(metrics.values.get("principal_ring_radius_px") or 0.0)
            if show_roi:
                roi_radius = max(radius * 1.7, min(width, height) * self._pixel_scale * 0.12)
                self._overlay_items.append(
                    self.scene().addEllipse(
                        cx - roi_radius,
                        cy - roi_radius,
                        2 * roi_radius,
                        2 * roi_radius,
                        _cosmetic(QPen(QColor("#69d2ff"), 1.2, Qt.DashLine)),
                    )
                )
            if show_ring and radius > 0:
                self._overlay_items.append(
                    self.scene().addEllipse(
                        cx - radius,
                        cy - radius,
                        2 * radius,
                        2 * radius,
                        _cosmetic(QPen(QColor("#f6c85f"), 2.0)),
                    )
                )
                core = radius * 0.35
                self._overlay_items.append(
                    self.scene().addEllipse(
                        cx - core,
                        cy - core,
                        2 * core,
                        2 * core,
                        _cosmetic(QPen(QColor("#e995ff"), 1.2, Qt.DotLine)),
                    )
                )
            if show_centre:
                pen = QPen(QColor("#59f2b1"), 1.6)
                pen.setCosmetic(True)
                # A gapped cross: the markers sit outside the ring so a
                # few-pixel vortex core is never painted over.
                gap = max(3.0, radius * 1.6)
                arm = max(8.0, radius * 1.2)
                self._overlay_items.extend(
                    [
                        self.scene().addLine(cx - gap - arm, cy, cx - gap, cy, pen),
                        self.scene().addLine(cx + gap, cy, cx + gap + arm, cy, pen),
                        self.scene().addLine(cx, cy - gap - arm, cx, cy - gap, pen),
                        self.scene().addLine(cx, cy + gap, cx, cy + gap + arm, pen),
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

    def _refresh_display_detail(self) -> None:
        rgb = self._display_rgb
        if rgb is None:
            return
        # Screen pixels per *displayed sample*: native model samples are a
        # fraction of a camera pixel wide.
        scale = (abs(self.transform().m11()) or 1.0) * self._pixel_scale
        factor = max(1, int(np.floor(1.0 / scale))) if scale < 1.0 else 1
        if factor == self._display_factor:
            return
        self._display_factor = factor
        if factor > 1:
            h = (rgb.shape[0] // factor) * factor
            w = (rgb.shape[1] // factor) * factor
            view = rgb[:h, :w].reshape(h // factor, factor, w // factor, factor, 3)
            shown = np.ascontiguousarray(view.mean(axis=(1, 3)).round().astype(np.uint8))
        else:
            shown = rgb
        height, width, _ = shown.shape
        image = QImage(shown.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._pixmap_item.setScale(float(factor) * self._pixel_scale)

    def scale(self, sx: float, sy: float) -> None:  # noqa: D401 - Qt API override
        super().scale(sx, sy)
        self._refresh_display_detail()

    def resizeEvent(self, event):  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh_display_detail()

    def fit_signal(self) -> None:
        rect = self.signal_rect()
        if rect is not None and not rect.isEmpty():
            self.resetTransform()
            self.fitInView(rect, Qt.KeepAspectRatio)
            self._refresh_display_detail()

    def fit_core(self, *, half_width_px: float | None = None) -> None:
        """Zoom to the beam centre at camera-pixel scale, where the ring and real sensor aliasing are visible."""

        metrics = self.metrics
        if metrics is not None:
            cy, cx = metrics.centre_yx_px
            radius = float(metrics.values.get("principal_ring_radius_px") or 0.0)
        elif self.raw_frame is not None:
            data = np.asarray(self.raw_frame)
            cy, cx = np.unravel_index(int(np.argmax(data)), data.shape)
            radius = 0.0
        else:
            return
        half = float(half_width_px) if half_width_px else max(12.0, 3.0 * radius)
        self.resetTransform()
        self.fitInView(QRectF(cx - half, cy - half, 2 * half, 2 * half), Qt.KeepAspectRatio)
        self._refresh_display_detail()

    def fit_full_frame(self) -> None:
        self.resetTransform()
        if not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._refresh_display_detail()

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
