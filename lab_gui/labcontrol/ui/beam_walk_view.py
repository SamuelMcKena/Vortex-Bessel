"""Dependency-light Qt propagation plot for accumulated formal captures."""

from __future__ import annotations

from PySide6.QtCore import QLineF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..beam_walk import BeamWalkResult


class BeamWalkPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.result: BeamWalkResult | None = None
        self.setMinimumHeight(230)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API
        return QSize(900, 240)

    def set_result(self, result: BeamWalkResult | None) -> None:
        self.result = result
        self.update()

    def paintEvent(self, _event):  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0b151c"))
        if self.result is None:
            painter.setPen(QColor("#8fa2ae"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Capture at least two physical z planes to fit propagation.")
            return
        planes = list(self.result.per_plane)
        z = [float(row["z_mm"]) for row in planes]
        series = [
            ("x centre / px", [float(row["mean_x_px"]) for row in planes], QColor("#59f2b1")),
            ("y centre / px", [float(row["mean_y_px"]) for row in planes], QColor("#f6c85f")),
            (
                "ring radius / px",
                [row.get("mean_ring_radius_px") for row in planes],
                QColor("#e995ff"),
            ),
        ]
        margin_left, margin_right, top, gap = 98.0, 18.0, 20.0, 12.0
        panel_height = (self.height() - top - 22.0 - 2 * gap) / 3.0
        for index, (label, raw_values, colour) in enumerate(series):
            values = [float(value) if value is not None else float("nan") for value in raw_values]
            valid = [(zz, vv) for zz, vv in zip(z, values) if vv == vv]
            rect = QRectF(
                margin_left,
                top + index * (panel_height + gap),
                max(10.0, self.width() - margin_left - margin_right),
                panel_height,
            )
            painter.setPen(QPen(QColor("#263a47"), 1.0))
            painter.drawRect(rect)
            painter.setPen(QColor("#aebdc7"))
            painter.drawText(QRectF(6, rect.top(), margin_left - 12, rect.height()), Qt.AlignVCenter | Qt.AlignRight, label)
            if len(valid) < 2:
                painter.setPen(QColor("#617783"))
                painter.drawText(rect, Qt.AlignCenter, "not available")
                continue
            z_values = [item[0] for item in valid]
            y_values = [item[1] for item in valid]
            z_min, z_max = min(z_values), max(z_values)
            y_min, y_max = min(y_values), max(y_values)
            z_span = max(z_max - z_min, 1e-12)
            padding = max((y_max - y_min) * 0.15, 0.25)
            y_min, y_max = y_min - padding, y_max + padding
            y_span = y_max - y_min

            def point(zz: float, value: float):
                return (
                    rect.left() + (zz - z_min) / z_span * rect.width(),
                    rect.bottom() - (value - y_min) / y_span * rect.height(),
                )

            painter.setPen(QPen(colour, 2.0))
            previous = None
            for zz, value in valid:
                px, py = point(zz, value)
                if previous is not None:
                    painter.drawLine(QLineF(previous[0], previous[1], px, py))
                painter.setBrush(colour)
                painter.drawEllipse(QRectF(px - 3, py - 3, 6, 6))
                previous = (px, py)
            painter.setPen(QColor("#748995"))
            painter.setFont(QFont(painter.font().family(), 8))
            painter.drawText(rect.adjusted(4, 2, -4, -2), Qt.AlignTop | Qt.AlignLeft, f"{y_max:.3g}")
            painter.drawText(rect.adjusted(4, 2, -4, -2), Qt.AlignBottom | Qt.AlignLeft, f"{y_min:.3g}")
        painter.setPen(QColor("#8fa2ae"))
        painter.drawText(
            QRectF(margin_left, self.height() - 20, self.width() - margin_left - margin_right, 18),
            Qt.AlignCenter,
            f"z / mm  •  {self.result.fit['label']}  •  total {float(self.result.fit['total_mrad']):.4g} mrad",
        )
