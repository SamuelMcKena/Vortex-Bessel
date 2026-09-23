from __future__ import annotations

import math
from typing import Iterable

from PySide6 import QtCore, QtGui, QtWidgets

from .types import Pose6D


class MovementMap2D(QtWidgets.QWidget):
    """Top-down path map for the standalone hexapod controller.

    SAMPLE BEAM expresses the fixed laser footprint in moving sample coordinates.
    HXP XY displays the measured carriage X/Y motion. Full travel is grey and
    laser-enabled sections are highlighted in amber.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._mode = "sample"
        self._travel_sample: list[tuple[float, float]] = []
        self._process_sample: list[tuple[float, float]] = []
        self._travel_hxp: list[tuple[float, float]] = []
        self._process_hxp: list[tuple[float, float]] = []
        self._current_sample: tuple[float, float] | None = None
        self._current_hxp: tuple[float, float] = (0.0, 0.0)
        self._laser_on = False
        self._span_mm = 40.0

    def set_mode(self, mode: str) -> None:
        mode = str(mode).lower().strip()
        if mode not in {"sample", "hxp"}:
            raise ValueError(mode)
        self._mode = mode
        self.update()

    def set_span_mm(self, span_mm: float) -> None:
        self._span_mm = max(2.0, float(span_mm))
        self.update()

    def clear(self) -> None:
        self._travel_sample.clear()
        self._process_sample.clear()
        self._travel_hxp.clear()
        self._process_hxp.clear()
        self._current_sample = None
        self._current_hxp = (0.0, 0.0)
        self.update()

    @staticmethod
    def _append_if_moved(
        points: list[tuple[float, float]],
        point: tuple[float, float],
        threshold_mm: float = 0.02,
    ) -> None:
        if not points:
            points.append(point)
            return
        dx = point[0] - points[-1][0]
        dy = point[1] - points[-1][1]
        if math.hypot(dx, dy) >= threshold_mm:
            points.append(point)

    def update_state(
        self,
        pose: Pose6D,
        laser_on: bool,
        sample_beam_xy_mm: Iterable[float] | None,
    ) -> None:
        self._laser_on = bool(laser_on)
        self._current_hxp = (float(pose.x), float(pose.y))
        self._append_if_moved(self._travel_hxp, self._current_hxp)
        if laser_on:
            self._append_if_moved(
                self._process_hxp,
                self._current_hxp,
                threshold_mm=0.01,
            )

        if sample_beam_xy_mm is not None:
            vals = tuple(float(v) for v in sample_beam_xy_mm)
            if len(vals) >= 2:
                self._current_sample = (vals[0], vals[1])
                self._append_if_moved(self._travel_sample, self._current_sample)
                if laser_on:
                    self._append_if_moved(
                        self._process_sample,
                        self._current_sample,
                        threshold_mm=0.01,
                    )
        self.update()

    def _datasets(self):
        if self._mode == "sample":
            return (
                self._travel_sample,
                self._process_sample,
                self._current_sample,
                "SAMPLE X",
                "SAMPLE Y",
                "LASER PATH ON SAMPLE",
            )
        return (
            self._travel_hxp,
            self._process_hxp,
            self._current_hxp,
            "HXP X",
            "HXP Y",
            "HXP XY CARRIAGE PATH",
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#0b1117"))

        travel, process, current, x_label, y_label, title = self._datasets()
        rect = self.rect().adjusted(48, 26, -18, -32)
        if rect.width() <= 10 or rect.height() <= 10:
            return

        painter.setPen(QtGui.QPen(QtGui.QColor("#2c3945"), 1))
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(QtGui.QColor("#aab7c3"))
        painter.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Weight.DemiBold))
        painter.drawText(50, 17, title)

        half = self._span_mm / 2.0
        raw = self._span_mm / 8.0
        candidates = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
        major = min(candidates, key=lambda x: abs(x - raw))
        minor = major / 5.0

        def to_px(pt: tuple[float, float]) -> QtCore.QPointF:
            x, y = pt
            px = rect.center().x() + (x / half) * (rect.width() / 2.0)
            py = rect.center().y() - (y / half) * (rect.height() / 2.0)
            return QtCore.QPointF(px, py)

        painter.setPen(QtGui.QPen(QtGui.QColor("#17212b"), 1))
        v = math.floor(-half / minor) * minor
        while v <= half + 1e-9:
            painter.drawLine(to_px((v, -half)), to_px((v, half)))
            painter.drawLine(to_px((-half, v)), to_px((half, v)))
            v += minor

        painter.setFont(QtGui.QFont("Consolas", 7))
        painter.setPen(QtGui.QPen(QtGui.QColor("#273643"), 1))
        v = math.ceil(-half / major) * major
        while v <= half + 1e-9:
            painter.drawLine(to_px((v, -half)), to_px((v, half)))
            painter.drawLine(to_px((-half, v)), to_px((half, v)))
            if abs(v) > 1e-9:
                p = to_px((v, 0.0))
                painter.setPen(QtGui.QColor("#6f7d89"))
                painter.drawText(
                    int(p.x() - 14),
                    rect.bottom() + 17,
                    28,
                    12,
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    f"{v:g}",
                )
                p = to_px((0.0, v))
                painter.drawText(
                    4,
                    int(p.y() - 7),
                    40,
                    14,
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter,
                    f"{v:g}",
                )
                painter.setPen(QtGui.QPen(QtGui.QColor("#273643"), 1))
            v += major

        painter.setPen(QtGui.QPen(QtGui.QColor("#526272"), 1.4))
        painter.drawLine(to_px((-half, 0.0)), to_px((half, 0.0)))
        painter.drawLine(to_px((0.0, -half)), to_px((0.0, half)))

        if len(travel) >= 2:
            path = QtGui.QPainterPath(to_px(travel[0]))
            for pt in travel[1:]:
                path.lineTo(to_px(pt))
            pen = QtGui.QPen(QtGui.QColor("#6f7c88"), 2.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)

        if len(process) >= 2:
            path = QtGui.QPainterPath(to_px(process[0]))
            for pt in process[1:]:
                path.lineTo(to_px(pt))
            pen = QtGui.QPen(QtGui.QColor("#ffc45e"), 4.0)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)

        if current is not None:
            p = to_px(current)
            if self._laser_on:
                painter.setPen(QtGui.QPen(QtGui.QColor("#ffcf75"), 2))
                painter.setBrush(QtGui.QColor(255, 185, 70, 48))
                painter.drawEllipse(p, 9, 9)
                painter.setBrush(QtGui.QColor("#ffc45e"))
            else:
                painter.setPen(QtGui.QPen(QtGui.QColor("#9eb0bf"), 1.5))
                painter.setBrush(QtGui.QColor("#d8e0e6"))
            painter.drawEllipse(p, 4, 4)

        painter.setFont(QtGui.QFont("Segoe UI", 7, QtGui.QFont.Weight.DemiBold))
        painter.setPen(QtGui.QColor("#82909d"))
        painter.drawText(
            rect.right() - 75,
            rect.bottom() + 26,
            75,
            12,
            QtCore.Qt.AlignmentFlag.AlignRight,
            f"{x_label} / mm",
        )
        painter.save()
        painter.translate(13, rect.top() + 80)
        painter.rotate(-90)
        painter.drawText(0, 0, f"{y_label} / mm")
        painter.restore()

        lx = rect.right() - 170
        ly = rect.top() + 10
        painter.setPen(QtGui.QPen(QtGui.QColor("#6f7c88"), 2))
        painter.drawLine(lx, ly, lx + 22, ly)
        painter.setPen(QtGui.QColor("#8b98a4"))
        painter.drawText(lx + 28, ly + 4, "travel")
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffc45e"), 4))
        painter.drawLine(lx + 78, ly, lx + 100, ly)
        painter.setPen(QtGui.QColor("#d7b36c"))
        painter.drawText(lx + 106, ly + 4, "laser on")
