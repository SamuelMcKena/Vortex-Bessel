from __future__ import annotations

import math
import traceback
from typing import Iterable, Sequence

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets

from .sample import SampleCalibration
from .types import Pose6D


Point2D = tuple[float, float]

TRAVEL_COLOR = "#5c6b78"
LASER_COLOR = "#ffb236"
LASER_GLOW = "#ff7a18"
SAMPLE_EDGE = "#3f8fb8"
SAMPLE_SAFE = "#2f6f92"


class TraceSegment:
    """One contiguous run of motion with a single beam state.

    A long raster accumulates tens of thousands of points, but the widget only
    has a few hundred thousand pixels. Painting every stored point was the
    dominant cost in the update loop, so each segment caches a screen-space
    path that is decimated to at most one point per pixel. A finished segment
    never changes, so its path is rebuilt only when the view itself moves.
    """

    __slots__ = (
        "laser_on",
        "points",
        "_cache_key",
        "_cache_path",
        "_cache_bounds",
        "_length_mm",
    )

    def __init__(self, laser_on: bool, first: Point2D) -> None:
        self.laser_on = bool(laser_on)
        self.points: list[Point2D] = [first]
        self._cache_key: tuple | None = None
        self._cache_path: QtGui.QPainterPath | None = None
        # Kept up to date on append: recomputing it per frame would defeat the
        # point of caching the path.
        self._cache_bounds = (first[0], first[1], first[0], first[1])
        self._length_mm = 0.0

    def append(self, point: Point2D, threshold_mm: float) -> None:
        last = self.points[-1]
        step = math.hypot(point[0] - last[0], point[1] - last[1])
        if step >= threshold_mm:
            self.points.append(point)
            self._length_mm += step
            self._cache_key = None
            x0, y0, x1, y1 = self._cache_bounds
            self._cache_bounds = (
                min(x0, point[0]),
                min(y0, point[1]),
                max(x1, point[0]),
                max(y1, point[1]),
            )

    def length_mm(self) -> float:
        # Accumulated on append; walking every point per frame was a
        # measurable share of the update loop on a long raster.
        return self._length_mm

    def bounds_mm(self) -> tuple[float, float, float, float]:
        return self._cache_bounds

    def painter_path(self, view_key: tuple, transform) -> QtGui.QPainterPath:
        """Screen-space path for this run, decimated to one point per pixel."""

        key = (view_key, len(self.points))
        if self._cache_key == key and self._cache_path is not None:
            return self._cache_path

        pixels = transform(np.asarray(self.points, dtype=float))
        if len(pixels) > 3:
            # Keep the first and last point plus every point that lands on a
            # different pixel from its predecessor. Visually identical, but the
            # path length becomes bounded by the widget size, not the data.
            rounded = np.round(pixels).astype(np.int32)
            keep = np.ones(len(rounded), dtype=bool)
            keep[1:-1] = np.any(np.diff(rounded, axis=0)[:-1] != 0, axis=1)
            pixels = pixels[keep]

        path = QtGui.QPainterPath(
            QtCore.QPointF(float(pixels[0][0]), float(pixels[0][1]))
        )
        for x, y in pixels[1:]:
            path.lineTo(float(x), float(y))
        self._cache_key = key
        self._cache_path = path
        return path


class MovementMap2D(QtWidgets.QWidget):
    """Top-down path map for the standalone hexapod controller.

    The path is stored as alternating segments so a beam-OFF repositioning move
    can never be drawn as though it were written material. Beam-ON runs are
    drawn as thick amber strokes with start/end markers; beam-OFF runs are thin
    dashed grey. The calibrated sample outline is drawn underneath both.
    """

    spanChangeRequested = QtCore.Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self._mode = "sample"
        self._sample_segments: list[TraceSegment] = []
        self._hxp_segments: list[TraceSegment] = []
        self._current_sample: Point2D | None = None
        self._current_hxp: Point2D = (0.0, 0.0)
        self._laser_on = False
        self._span_mm = 40.0
        self._centre_mm: Point2D = (0.0, 0.0)
        self._show_travel = True
        self._show_written = True
        self._show_sample = True
        self._hover_px: QtCore.QPointF | None = None
        self._calibration = SampleCalibration()
        self._paint_failed = False

    # ------------------------------------------------------------- setters
    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"sample", "hxp"}:
            raise ValueError(mode)
        self._mode = mode
        self.update()

    def set_span_mm(self, span_mm: float) -> None:
        self._span_mm = max(0.05, float(span_mm))
        self.update()

    def set_centre_mm(self, x_mm: float, y_mm: float) -> None:
        self._centre_mm = (float(x_mm), float(y_mm))
        self.update()

    def set_show_travel(self, visible: bool) -> None:
        self._show_travel = bool(visible)
        self.update()

    def set_show_written(self, visible: bool) -> None:
        self._show_written = bool(visible)
        self.update()

    def set_show_sample(self, visible: bool) -> None:
        self._show_sample = bool(visible)
        self.update()

    def set_sample_calibration(self, calibration: SampleCalibration) -> None:
        self._calibration = calibration
        self.update()

    def clear(self) -> None:
        self._sample_segments.clear()
        self._hxp_segments.clear()
        self._current_sample = None
        self._current_hxp = (0.0, 0.0)
        self.update()

    # ------------------------------------------------------------ tracking
    #: Beyond this many stored points per view the oldest runs are dropped, so
    #: an all-day session cannot exhaust memory. A 2 mm/s raster sampled at
    #: 20 Hz reaches this only after many hours of continuous motion.
    MAX_TRACE_POINTS = 60_000

    @classmethod
    def _trim(cls, segments: list[TraceSegment]) -> None:
        total = sum(len(segment.points) for segment in segments)
        while total > cls.MAX_TRACE_POINTS and len(segments) > 1:
            total -= len(segments[0].points)
            segments.pop(0)

    @staticmethod
    def _extend(
        segments: list[TraceSegment],
        point: Point2D,
        laser_on: bool,
        threshold_mm: float,
    ) -> None:
        if not segments:
            segments.append(TraceSegment(laser_on, point))
            return
        current = segments[-1]
        if current.laser_on == laser_on:
            current.append(point, threshold_mm)
            return
        # The spot path is continuous: the new segment starts where the beam
        # state actually changed, so written runs cannot appear detached.
        fresh = TraceSegment(laser_on, current.points[-1])
        fresh.append(point, 0.0)
        segments.append(fresh)

    def update_state(
        self,
        pose: Pose6D,
        laser_on: bool,
        sample_beam_xy_mm: Iterable[float] | None,
    ) -> None:
        laser_on = bool(laser_on)
        before = (self._laser_on, self._current_hxp, self._current_sample)
        self._laser_on = laser_on
        threshold = 0.005 if laser_on else 0.02

        self._current_hxp = (float(pose.x), float(pose.y))
        self._extend(self._hxp_segments, self._current_hxp, laser_on, threshold)
        self._trim(self._hxp_segments)

        if sample_beam_xy_mm is not None:
            values = tuple(float(v) for v in sample_beam_xy_mm)
            if len(values) >= 2:
                self._current_sample = (values[0], values[1])
                self._extend(
                    self._sample_segments,
                    self._current_sample,
                    laser_on,
                    threshold,
                )
                self._trim(self._sample_segments)
        # Repainting a stationary map every tick is wasted work.
        if before != (self._laser_on, self._current_hxp, self._current_sample):
            self.update()

    # ------------------------------------------------------------- queries
    def _segments(self) -> list[TraceSegment]:
        return (
            self._sample_segments
            if self._mode == "sample"
            else self._hxp_segments
        )

    def _current(self) -> Point2D | None:
        return (
            self._current_sample
            if self._mode == "sample"
            else self._current_hxp
        )

    def _region(self, *, safe: bool) -> list[Point2D] | None:
        if self._mode == "hxp":
            return (
                self._calibration.safe_region()
                if safe
                else self._calibration.stage_region()
            )
        if safe:
            return None
        return self._calibration.sample_region()

    def written_length_mm(self) -> float:
        return sum(
            segment.length_mm()
            for segment in self._segments()
            if segment.laser_on
        )

    def written_run_count(self) -> int:
        return sum(
            1
            for segment in self._segments()
            if segment.laser_on and len(segment.points) >= 2
        )

    def data_bounds_mm(self) -> tuple[float, float, float, float] | None:
        xs: list[float] = []
        ys: list[float] = []
        for segment in self._segments():
            for x, y in segment.points:
                xs.append(x)
                ys.append(y)
        region = self._region(safe=False)
        if region:
            xs.extend(p[0] for p in region)
            ys.extend(p[1] for p in region)
        current = self._current()
        if current is not None:
            xs.append(current[0])
            ys.append(current[1])
        if not xs:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    def suggested_view(self) -> tuple[float, float, float] | None:
        """Span and centre that frame everything currently drawn."""
        bounds = self.data_bounds_mm()
        if bounds is None:
            return None
        x0, y0, x1, y1 = bounds
        span = max(x1 - x0, y1 - y0) * 1.25
        span = max(0.2, span)
        return (span, (x0 + x1) / 2.0, (y0 + y1) / 2.0)

    # -------------------------------------------------------------- events
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._hover_px = event.position()
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self._hover_px = None
        self.update()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        view = self.suggested_view()
        if view is not None:
            self.set_centre_mm(view[1], view[2])
            self.spanChangeRequested.emit(view[0])
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        if steps:
            self.spanChangeRequested.emit(
                max(0.1, self._span_mm * (0.85 ** steps))
            )
            event.accept()
            return
        super().wheelEvent(event)

    # ------------------------------------------------------------ painting
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        # PySide6 aborts the process when an exception escapes a virtual, and a
        # map that cannot draw must not take the controller down with it.
        try:
            self._paint(event)
        except Exception:
            if not self._paint_failed:
                self._paint_failed = True
                traceback.print_exc()

    def _paint(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#0a1016"))

        plot = self.rect().adjusted(52, 30, -16, -34)
        if plot.width() <= 20 or plot.height() <= 20:
            return

        title = (
            "LASER PATH ON SAMPLE"
            if self._mode == "sample"
            else "HXP XY CARRIAGE PATH"
        )
        x_label = "SAMPLE X" if self._mode == "sample" else "HXP X"
        y_label = "SAMPLE Y" if self._mode == "sample" else "HXP Y"

        half = self._span_mm / 2.0
        cx, cy = self._centre_mm
        # Equal millimetres per pixel on both axes, so a written line keeps its
        # true shape; the longer axis simply shows more of the stage.
        pixels_per_mm = min(plot.width(), plot.height()) / (2.0 * half)
        half_x = (plot.width() / 2.0) / pixels_per_mm
        half_y = (plot.height() / 2.0) / pixels_per_mm

        def to_px(point: Sequence[float]) -> QtCore.QPointF:
            return QtCore.QPointF(
                plot.center().x() + (float(point[0]) - cx) * pixels_per_mm,
                plot.center().y() - (float(point[1]) - cy) * pixels_per_mm,
            )

        def to_mm(pixel: QtCore.QPointF) -> Point2D:
            return (
                cx + (pixel.x() - plot.center().x()) / pixels_per_mm,
                cy - (pixel.y() - plot.center().y()) / pixels_per_mm,
            )

        centre_px = plot.center()

        def to_px_array(values: "np.ndarray") -> "np.ndarray":
            pixels = np.empty_like(values, dtype=float)
            pixels[:, 0] = centre_px.x() + (values[:, 0] - cx) * pixels_per_mm
            pixels[:, 1] = centre_px.y() - (values[:, 1] - cy) * pixels_per_mm
            return pixels

        view_key = (
            round(cx, 9),
            round(cy, 9),
            round(pixels_per_mm, 6),
            plot.width(),
            plot.height(),
        )
        visible_mm = (
            cx - half_x,
            cy - half_y,
            cx + half_x,
            cy + half_y,
        )

        painter.setClipRect(plot.adjusted(-1, -1, 1, 1))
        self._paint_grid(painter, to_px, half, half_x, half_y, cx, cy)
        if self._show_sample:
            self._paint_sample(painter, to_px)
        self._paint_traces(painter, to_px, view_key, to_px_array, visible_mm)
        self._paint_current(painter, to_px)
        painter.setClipping(False)

        # The trace/marker passes leave a fill brush set; the frame must not
        # flood the plot with it.
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2c3945"), 1))
        painter.drawRoundedRect(plot, 5, 5)
        self._paint_labels(painter, plot, title, x_label, y_label)
        self._paint_legend(painter, plot)
        self._paint_hover(painter, plot, to_mm, to_px)

    def _paint_grid(self, painter, to_px, half, half_x, half_y, cx, cy) -> None:
        raw = (half * 2.0) / 8.0
        exponent = math.floor(math.log10(raw)) if raw > 0 else 0
        base = 10.0 ** exponent
        major = base
        for factor in (1.0, 2.0, 5.0, 10.0):
            major = base * factor
            if major >= raw:
                break
        minor = major / 5.0

        def ticks(centre: float, reach: float, step: float) -> list[float]:
            values: list[float] = []
            value = math.floor((centre - reach) / step) * step
            limit = centre + reach
            while value <= limit + 1e-9 and len(values) < 3000:
                values.append(value)
                value += step
            return values

        def draw(step: float) -> tuple[list[float], list[float]]:
            xs = ticks(cx, half_x, step)
            ys = ticks(cy, half_y, step)
            for value in xs:
                painter.drawLine(
                    to_px((value, cy - half_y)),
                    to_px((value, cy + half_y)),
                )
            for value in ys:
                painter.drawLine(
                    to_px((cx - half_x, value)),
                    to_px((cx + half_x, value)),
                )
            return xs, ys

        painter.setPen(QtGui.QPen(QtGui.QColor("#141d25"), 1))
        draw(minor)
        painter.setPen(QtGui.QPen(QtGui.QColor("#22303c"), 1))
        self._major_x, self._major_y = draw(major)
        self._major_step = major

        painter.setPen(QtGui.QPen(QtGui.QColor("#4c5c6b"), 1.4))
        painter.drawLine(to_px((cx - half_x, 0.0)), to_px((cx + half_x, 0.0)))
        painter.drawLine(to_px((0.0, cy - half_y)), to_px((0.0, cy + half_y)))
        self._to_px = to_px

    def _paint_sample(self, painter, to_px) -> None:
        region = self._region(safe=False)
        if not region:
            return
        polygon = QtGui.QPolygonF([to_px(point) for point in region])
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(63, 143, 184, 26))
        painter.drawPolygon(polygon)
        pen = QtGui.QPen(QtGui.QColor(SAMPLE_EDGE), 1.6)
        pen.setStyle(QtCore.Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)

        safe = self._region(safe=True)
        if safe and self._calibration.margin_mm > 0:
            pen = QtGui.QPen(QtGui.QColor(SAMPLE_SAFE), 1.1)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawPolygon(
                QtGui.QPolygonF([to_px(point) for point in safe])
            )

        if self._mode == "hxp":
            painter.setPen(QtGui.QPen(QtGui.QColor("#6fb6d8"), 1))
            painter.setBrush(QtGui.QColor("#1b3a4b"))
            font = QtGui.QFont("Segoe UI", 7, QtGui.QFont.Weight.DemiBold)
            painter.setFont(font)
            for index, point in enumerate(self._calibration.points):
                pixel = to_px((point.pose.x, point.pose.y))
                painter.drawEllipse(pixel, 4.5, 4.5)
                painter.drawText(
                    QtCore.QRectF(pixel.x() + 6, pixel.y() - 14, 60, 12),
                    QtCore.Qt.AlignmentFlag.AlignLeft,
                    f"P{index + 1}",
                )

    @staticmethod
    def _intersects(bounds, visible) -> bool:
        return not (
            bounds[2] < visible[0]
            or bounds[0] > visible[2]
            or bounds[3] < visible[1]
            or bounds[1] > visible[3]
        )

    def _paint_traces(self, painter, to_px, view_key, to_px_array, visible) -> None:
        segments = self._segments()
        if not segments:
            return

        def drawable(want_laser: bool) -> list[TraceSegment]:
            chosen = []
            for segment in segments:
                if segment.laser_on is not want_laser or len(segment.points) < 2:
                    continue
                # Runs entirely outside the view cost nothing to skip.
                if not self._intersects(segment.bounds_mm(), visible):
                    continue
                chosen.append(segment)
            return chosen

        if self._show_travel:
            pen = QtGui.QPen(QtGui.QColor(TRAVEL_COLOR), 1.4)
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
            pen.setDashPattern([5.0, 4.0])
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for segment in drawable(False):
                painter.drawPath(segment.painter_path(view_key, to_px_array))

        if not self._show_written:
            return

        glow = QtGui.QPen(QtGui.QColor(255, 122, 24, 70), 9.0)
        glow.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        glow.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        core = QtGui.QPen(QtGui.QColor(LASER_COLOR), 3.4)
        core.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        core.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        written = drawable(True)
        for segment in written:
            path = segment.painter_path(view_key, to_px_array)
            painter.setPen(glow)
            painter.drawPath(path)
            painter.setPen(core)
            painter.drawPath(path)

        painter.setFont(QtGui.QFont("Consolas", 7))
        for index, segment in enumerate(written, start=1):
            start = to_px(segment.points[0])
            end = to_px(segment.points[-1])
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffe0a8"), 1.4))
            painter.setBrush(QtGui.QColor("#0a1016"))
            painter.drawEllipse(start, 3.2, 3.2)
            painter.setBrush(QtGui.QColor(LASER_COLOR))
            painter.drawRect(QtCore.QRectF(end.x() - 2.6, end.y() - 2.6, 5.2, 5.2))
            if len(written) <= 24:
                painter.setPen(QtGui.QColor("#c9a45e"))
                painter.drawText(
                    QtCore.QRectF(start.x() - 26, start.y() - 15, 22, 12),
                    QtCore.Qt.AlignmentFlag.AlignRight,
                    str(index),
                )

    def _paint_current(self, painter, to_px) -> None:
        current = self._current()
        if current is None:
            return
        pixel = to_px(current)
        if self._laser_on:
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffd489"), 2))
            painter.setBrush(QtGui.QColor(255, 178, 54, 55))
            painter.drawEllipse(pixel, 10, 10)
            painter.setBrush(QtGui.QColor(LASER_COLOR))
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor("#93a4b2"), 1.4))
            painter.setBrush(QtGui.QColor("#dbe3e9"))
        painter.drawEllipse(pixel, 4, 4)

    def _paint_labels(self, painter, plot, title, x_label, y_label) -> None:
        painter.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.DemiBold))
        painter.setPen(QtGui.QColor("#b6c3ce"))
        painter.drawText(plot.left() + 2, plot.top() - 10, title)

        painter.setFont(QtGui.QFont("Consolas", 7))
        painter.setPen(QtGui.QColor("#6f7d89"))

        major_x = getattr(self, "_major_x", [])
        major_y = getattr(self, "_major_y", [])
        step = getattr(self, "_major_step", 0.0)

        def stride(values, axis_index: int, minimum_px: float) -> int:
            """Label every nth gridline so the numbers never collide."""
            if len(values) < 2 or step <= 0:
                return 1
            first = self._to_px((values[0], values[0]))
            second = self._to_px((values[1], values[1]))
            spacing = abs(
                (second.x() - first.x())
                if axis_index == 0
                else (second.y() - first.y())
            )
            if spacing <= 0.5:
                return len(values) + 1
            return max(1, math.ceil(minimum_px / spacing))

        stride_x = stride(major_x, 0, 56.0)
        for value in major_x:
            if step > 0 and round(value / step) % stride_x:
                continue
            pixel = self._to_px((value, 0.0))
            if not (plot.left() - 1 <= pixel.x() <= plot.right() + 1):
                continue
            painter.drawText(
                QtCore.QRectF(pixel.x() - 28, plot.bottom() + 4, 56, 13),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"{value:g}",
            )

        stride_y = stride(major_y, 1, 22.0)
        for value in major_y:
            if step > 0 and round(value / step) % stride_y:
                continue
            pixel = self._to_px((0.0, value))
            if not (plot.top() - 1 <= pixel.y() <= plot.bottom() + 1):
                continue
            painter.drawText(
                QtCore.QRectF(2, pixel.y() - 7, 46, 14),
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"{value:g}",
            )

        painter.setFont(QtGui.QFont("Segoe UI", 7, QtGui.QFont.Weight.DemiBold))
        painter.setPen(QtGui.QColor("#82909d"))
        painter.drawText(
            QtCore.QRectF(plot.right() - 96, plot.bottom() + 18, 96, 13),
            QtCore.Qt.AlignmentFlag.AlignRight,
            f"{x_label} / mm",
        )
        painter.save()
        painter.translate(13, plot.top() + 96)
        painter.rotate(-90)
        painter.drawText(0, 0, f"{y_label} / mm")
        painter.restore()

    def _paint_legend(self, painter, plot) -> None:
        painter.setFont(QtGui.QFont("Segoe UI", 7, QtGui.QFont.Weight.DemiBold))
        x = plot.left() + 8
        y = plot.top() + 12

        pen = QtGui.QPen(QtGui.QColor(TRAVEL_COLOR), 1.6)
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(int(x), int(y), int(x) + 20, int(y))
        painter.setPen(QtGui.QColor("#8b98a4"))
        painter.drawText(int(x) + 26, int(y) + 4, "LASER OFF")

        x2 = x + 96
        painter.setPen(QtGui.QPen(QtGui.QColor(LASER_COLOR), 3.4))
        painter.drawLine(int(x2), int(y), int(x2) + 20, int(y))
        painter.setPen(QtGui.QColor("#e0b978"))
        painter.drawText(int(x2) + 26, int(y) + 4, "LASER ON")

        if self._calibration.stage_region() is not None and self._show_sample:
            x3 = x2 + 92
            painter.setPen(QtGui.QPen(QtGui.QColor(SAMPLE_EDGE), 1.6))
            painter.drawLine(int(x3), int(y), int(x3) + 20, int(y))
            painter.setPen(QtGui.QColor("#7fb6d2"))
            painter.drawText(int(x3) + 26, int(y) + 4, "SAMPLE")

        runs = self.written_run_count()
        if runs:
            painter.setFont(QtGui.QFont("Consolas", 7))
            painter.setPen(QtGui.QColor("#c9a45e"))
            painter.drawText(
                QtCore.QRectF(plot.right() - 210, plot.top() + 4, 200, 13),
                QtCore.Qt.AlignmentFlag.AlignRight,
                f"{runs} written run(s)   {self.written_length_mm():.3f} mm",
            )

    def _paint_hover(self, painter, plot, to_mm, to_px) -> None:
        if self._hover_px is None or not plot.contains(self._hover_px.toPoint()):
            return
        x_mm, y_mm = to_mm(self._hover_px)
        pen = QtGui.QPen(QtGui.QColor(120, 150, 175, 130), 1)
        pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        painter.setPen(pen)
        painter.drawLine(
            QtCore.QPointF(plot.left(), self._hover_px.y()),
            QtCore.QPointF(plot.right(), self._hover_px.y()),
        )
        painter.drawLine(
            QtCore.QPointF(self._hover_px.x(), plot.top()),
            QtCore.QPointF(self._hover_px.x(), plot.bottom()),
        )

        text = f"X {x_mm:+.4f}   Y {y_mm:+.4f} mm"
        current = self._current()
        if current is not None:
            dx = x_mm - current[0]
            dy = y_mm - current[1]
            text += f"    d {math.hypot(dx, dy):.4f} mm"
        painter.setFont(QtGui.QFont("Consolas", 7))
        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 12
        box = QtCore.QRectF(
            min(self._hover_px.x() + 10, plot.right() - width),
            max(plot.top() + 2, self._hover_px.y() - 24),
            width,
            17,
        )
        painter.setPen(QtGui.QPen(QtGui.QColor("#3a4a57"), 1))
        painter.setBrush(QtGui.QColor(12, 19, 26, 225))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QtGui.QColor("#cfdae2"))
        painter.drawText(box, QtCore.Qt.AlignmentFlag.AlignCenter, text)
