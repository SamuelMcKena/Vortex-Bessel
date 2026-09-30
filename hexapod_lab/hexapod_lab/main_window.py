from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import datetime
import json
import sys
import time
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from .hardware_profiles import (
    LegacyHardwareProfile,
    PockelsCandidate,
    load_legacy_hardware_profile,
)
from .controller_profile import (
    HXPControllerProfile,
    load_hxp_controller_profile,
)
from .kinematics import RigKinematics, RigProfile
from .movement_map import MovementMap2D
from .providers import (
    AttenuatorProvider,
    HXPDigitalLaserConfig,
    HXPDigitalLaserGate,
    HXPProvider,
    HXPProviderConfig,
    HXPAnalogAttenuatorConfig,
    HXPAnalogAttenuatorProvider,
    HexapodProvider,
    LaserGateProvider,
    UnconfiguredAttenuatorProvider,
    VirtualAttenuatorProvider,
    VirtualHexapodProvider,
    VirtualLaserGate,
)
from .recipe import (
    MOTION_KINDS,
    POCKELS_KINDS,
    Recipe,
    RecipeStep,
    StepKind,
    preflight_recipe,
)
from .sample import SampleCalibration, SamplePoint, next_point_label
from .sweeps import RasterSweepSpec, build_raster_sweep
from .types import (
    AttenuatorSnapshot,
    HexapodSnapshot,
    LaserSnapshot,
    MotionState,
    Pose6D,
)
from .viewer import Hexapod3DViewer
from .workspace import (
    AXES,
    RecipeWorkspaceIssue,
    WorkspaceLimits,
    validate_recipe_workspace,
)


PACKAGE_DIR = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = APP_ROOT / "assets"
DEFAULT_PROFILE = ASSETS_DIR / "cad_profile.json"
DEFAULT_CONFIG = APP_ROOT / "hardware_config.example.json"
LEGACY_EVIDENCE = ASSETS_DIR / "legacy_hardware_evidence.json"
CONTROLLER_PROFILE = ASSETS_DIR / "hxp_controller_profile_2026-09-30.json"
CRASH_LOG = APP_ROOT / "hexapod_lab_crash.log"
RECIPE_MODULE_MIME = "application/x-hexapod-recipe-module"


MODULE_PRESENTATION = {
    StepKind.MOVE_ABSOLUTE: ("ABS MOVE", "#4f9bc4"),
    StepKind.MOVE_INCREMENTAL: ("REL MOVE", "#668fd1"),
    StepKind.MOVE_LINE_VELOCITY: ("LINE", "#8a79d6"),
    StepKind.WRITE_LINE: ("MOVE WHILE WRITE", "#e0663f"),
    StepKind.POCKELS_CELL: ("BEAM", "#d08a3a"),
    StepKind.ATTENUATOR_SET: ("ATTENUATOR", "#42a777"),
    StepKind.WAIT: ("WAIT", "#8c98a3"),
}

MODULE_CATALOGUE = (
    ("move_absolute", "ABSOLUTE MOVE", "Move to an XYZUVW pose", "#4f9bc4"),
    ("move_incremental", "RELATIVE MOVE", "Offset the current XYZUVW pose", "#668fd1"),
    ("move_line_velocity", "LINE AT VELOCITY", "XYZ line using target velocity", "#8a79d6"),
    ("write_line", "MOVE WHILE WRITE", "Line with the beam open for the move only", "#e0663f"),
    ("pockels_open", "POCKELS OPEN", "Enable the process beam", "#d08a3a"),
    ("pockels_closed", "POCKELS CLOSED", "Disable the process beam", "#d08a3a"),
    ("attenuator_set", "ATTENUATOR", "Set requested transmission", "#42a777"),
    ("wait", "WAIT", "Pause before the next module", "#8c98a3"),
)


class ModuleLibraryDelegate(QtWidgets.QStyledItemDelegate):
    def sizeHint(self, option, index) -> QtCore.QSize:
        return QtCore.QSize(option.rect.width(), 52)

    def paint(self, painter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(2, 2, -2, -2)
        selected = bool(
            option.state & QtWidgets.QStyle.StateFlag.State_Selected
        )
        color = QtGui.QColor(
            str(index.data(QtCore.Qt.ItemDataRole.UserRole + 1))
        )
        painter.setPen(
            QtGui.QPen(color if selected else QtGui.QColor("#33414d"))
        )
        painter.setBrush(QtGui.QColor("#20303c" if selected else "#151f28"))
        painter.drawRoundedRect(rect, 6, 6)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(
            QtCore.QRectF(rect.left(), rect.top(), 5, rect.height()),
            2,
            2,
        )

        lines = str(index.data(QtCore.Qt.ItemDataRole.DisplayRole)).strip().split("\n")
        title = lines[0].strip() if lines else "MODULE"
        detail = lines[1].strip() if len(lines) > 1 else ""
        title_font = QtGui.QFont(option.font)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(color.lighter(135))
        painter.drawText(
            rect.adjusted(15, 5, -10, -25),
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
            title,
        )
        detail_font = QtGui.QFont(option.font)
        detail_font.setPointSizeF(max(8.0, option.font.pointSizeF() - 1.0))
        painter.setFont(detail_font)
        painter.setPen(QtGui.QColor("#a9b7c2"))
        detail = QtGui.QFontMetrics(detail_font).elidedText(
            detail,
            QtCore.Qt.TextElideMode.ElideRight,
            max(30, rect.width() - 28),
        )
        painter.drawText(
            rect.adjusted(15, 26, -10, -4),
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
            detail,
        )
        painter.restore()


class ModuleLibrary(QtWidgets.QListWidget):
    """Drag source for recipe modules; double-click also inserts."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.DragOnly
        )
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.setSpacing(3)
        self.setWordWrap(True)
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setItemDelegate(ModuleLibraryDelegate(self))

        for key, title, description, color in MODULE_CATALOGUE:
            item = QtWidgets.QListWidgetItem(f"  {title}\n  {description}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, color)
            item.setForeground(QtGui.QColor(color).lighter(135))
            item.setBackground(QtGui.QColor("#151f28"))
            item.setToolTip(
                "Drag into the sequence, or double-click to insert it after "
                "the selected block"
            )
            item.setSizeHint(QtCore.QSize(240, 52))
            self.addItem(item)
        # The palette already scrolls as a page; a nested scroll area here only
        # makes the modules harder to reach.
        self.setFixedHeight(self.count() * 58 + 8)

    def _drag_pixmap(self, item: QtWidgets.QListWidgetItem) -> QtGui.QPixmap:
        color = QtGui.QColor(str(item.data(QtCore.Qt.ItemDataRole.UserRole + 1)))
        title = str(item.text()).strip().split("\n")[0].strip()
        ratio = self.devicePixelRatioF()
        width, height = 232, 40
        pixmap = QtGui.QPixmap(int(width * ratio), int(height * ratio))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(color, 1.6))
        painter.setBrush(QtGui.QColor(20, 30, 39, 242))
        painter.drawRoundedRect(QtCore.QRectF(1, 1, width - 2, height - 2), 6, 6)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(QtCore.QRectF(2, 2, 5, height - 4), 2, 2)
        font = QtGui.QFont(self.font())
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(color.lighter(140))
        painter.drawText(
            QtCore.QRectF(16, 0, width - 24, height),
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
            title,
        )
        painter.end()
        return pixmap

    def startDrag(self, supported_actions) -> None:
        item = self.currentItem()
        if item is None:
            return
        mime = QtCore.QMimeData()
        mime.setData(
            RECIPE_MODULE_MIME,
            str(item.data(QtCore.Qt.ItemDataRole.UserRole)).encode("utf-8"),
        )
        drag = QtGui.QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(self._drag_pixmap(item))
        # Hot spot is in logical pixels, independent of the device ratio.
        drag.setHotSpot(QtCore.QPoint(18, 20))
        drag.exec(QtCore.Qt.DropAction.CopyAction)


class RecipeBlockDelegate(QtWidgets.QStyledItemDelegate):
    """Paint recipe rows as clear, colour-coded experiment modules."""

    def sizeHint(self, option, index) -> QtCore.QSize:
        return QtCore.QSize(option.rect.width(), 72)

    def paint(self, painter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(2, 2, -2, -2)
        selected = bool(
            option.state
            & QtWidgets.QStyle.StateFlag.State_Selected
        )
        background = QtGui.QColor("#243847" if selected else "#18212a")
        border = QtGui.QColor("#6aa2c8" if selected else "#35424f")
        painter.setPen(QtGui.QPen(border, 1.5 if selected else 1.0))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 7, 7)

        raw = index.data(QtCore.Qt.ItemDataRole.UserRole) or {}
        try:
            kind = StepKind(str(raw.get("kind", "wait")))
        except Exception:
            kind = StepKind.WAIT
        tag, color_text = MODULE_PRESENTATION[kind]
        accent = QtGui.QColor(color_text)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(
            QtCore.QRectF(rect.left(), rect.top(), 6, rect.height()),
            3,
            3,
        )

        number = index.row() + 1
        title_font = QtGui.QFont(option.font)
        title_font.setBold(True)
        title_font.setPointSizeF(max(8.5, option.font.pointSizeF() - 0.5))
        painter.setFont(title_font)
        painter.setPen(accent.lighter(130))
        painter.drawText(
            rect.adjusted(18, 8, -50, -36),
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
            f"{number:02d}   {tag}",
        )

        lines = str(index.data(QtCore.Qt.ItemDataRole.DisplayRole)).split("\n")
        detail = lines[-1] if lines else ""
        detail_font = QtGui.QFont(option.font)
        detail_font.setPointSizeF(max(8.0, option.font.pointSizeF() - 1.0))
        painter.setFont(detail_font)
        painter.setPen(QtGui.QColor("#d7e0e7"))
        metrics = QtGui.QFontMetrics(detail_font)
        detail = metrics.elidedText(
            detail,
            QtCore.Qt.TextElideMode.ElideRight,
            max(40, rect.width() - 76),
        )
        painter.drawText(
            rect.adjusted(18, 34, -46, -7),
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter,
            detail,
        )

        painter.setPen(QtGui.QColor("#8493a0"))
        painter.drawText(
            rect.adjusted(rect.width() - 39, 0, -10, 0),
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "⋮⋮",
        )
        painter.restore()


class RecipeStepEditorDialog(QtWidgets.QDialog):
    """Edit one recipe module without rebuilding the whole sequence."""

    def __init__(
        self,
        step: RecipeStep,
        limits: WorkspaceLimits | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._original = step
        self._boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.setWindowTitle(f"Edit module — {step.label}")
        self.setModal(True)
        self.setMinimumWidth(440)

        layout = QtWidgets.QVBoxLayout(self)
        tag, _ = MODULE_PRESENTATION[step.kind]
        heading = QtWidgets.QLabel(tag)
        heading.setObjectName("section")
        layout.addWidget(heading)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        if step.kind in (StepKind.MOVE_ABSOLUTE, StepKind.MOVE_INCREMENTAL):
            key = "pose" if step.kind == StepKind.MOVE_ABSOLUTE else "delta"
            values = Pose6D.from_iterable(step.payload[key]).as_tuple()
            for index, (axis, value) in enumerate(zip("XYZUVW", values)):
                if limits is None:
                    lo, hi = (
                        (-1000.0, 1000.0)
                        if index < 3
                        else (-180.0, 180.0)
                    )
                else:
                    axis_lo, axis_hi = limits.axis_bounds(axis)
                    if step.kind == StepKind.MOVE_ABSOLUTE:
                        lo, hi = axis_lo, axis_hi
                    else:
                        span = axis_hi - axis_lo
                        lo, hi = -span, span
                box = self._number_box(
                    value,
                    lo,
                    hi,
                    4,
                    " mm" if index < 3 else " °",
                )
                self._boxes[axis] = box
                form.addRow(axis, box)
        elif step.kind in (StepKind.MOVE_LINE_VELOCITY, StepKind.WRITE_LINE):
            delta = list(step.payload["delta_xyz_mm"])
            for axis, value in zip("XYZ", delta):
                span = (
                    2000.0
                    if limits is None
                    else limits.axis_bounds(axis)[1]
                    - limits.axis_bounds(axis)[0]
                )
                box = self._number_box(value, -span, span, 4, " mm")
                self._boxes[axis] = box
                form.addRow(f"d{axis}", box)
            velocity = self._number_box(
                float(step.payload["velocity_mm_s"]),
                0.001,
                100.0
                if limits is None
                else limits.maximum_line_velocity_mm_s,
                3,
                " mm/s",
            )
            self._boxes["velocity"] = velocity
            form.addRow("Target velocity", velocity)
            if step.kind == StepKind.WRITE_LINE:
                note = QtWidgets.QLabel(
                    "The Pockels cell is opened before this move and closed "
                    "again when it finishes, fails or is stopped."
                )
                note.setObjectName("muted")
                note.setWordWrap(True)
                form.addRow(note)
        elif step.kind == StepKind.POCKELS_CELL:
            self.state_combo = QtWidgets.QComboBox()
            self.state_combo.addItems(["CLOSED — laser off", "OPEN — laser on"])
            self.state_combo.setCurrentIndex(
                1 if bool(step.payload.get("open")) else 0
            )
            form.addRow("Pockels state", self.state_combo)
        elif step.kind == StepKind.ATTENUATOR_SET:
            box = self._number_box(
                float(step.payload["transmission_percent"]),
                0.0,
                100.0,
                1,
                " %",
            )
            self._boxes["attenuator"] = box
            form.addRow("Transmission", box)
        elif step.kind == StepKind.WAIT:
            box = self._number_box(
                float(step.payload.get("seconds", 0.0)),
                0.0,
                3600.0,
                3,
                " s",
            )
            self._boxes["wait"] = box
            form.addRow("Duration", box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if parent is not None:
            for widget in (
                *self.findChildren(QtWidgets.QAbstractSpinBox),
                *self.findChildren(QtWidgets.QComboBox),
            ):
                widget.installEventFilter(parent)

    @staticmethod
    def _number_box(
        value: float,
        minimum: float,
        maximum: float,
        decimals: int,
        suffix: str,
    ) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSuffix(suffix)
        box.setValue(value)
        return box

    def edited_step(self) -> RecipeStep:
        kind = self._original.kind
        if kind in (StepKind.MOVE_ABSOLUTE, StepKind.MOVE_INCREMENTAL):
            pose = Pose6D(*(self._boxes[a].value() for a in "XYZUVW"))
            return (
                RecipeStep.move_absolute(pose)
                if kind == StepKind.MOVE_ABSOLUTE
                else RecipeStep.move_incremental(pose)
            )
        if kind in (StepKind.MOVE_LINE_VELOCITY, StepKind.WRITE_LINE):
            builder = (
                RecipeStep.move_line_velocity
                if kind == StepKind.MOVE_LINE_VELOCITY
                else RecipeStep.write_line
            )
            return builder(
                self._boxes["X"].value(),
                self._boxes["Y"].value(),
                self._boxes["Z"].value(),
                self._boxes["velocity"].value(),
            )
        if kind == StepKind.POCKELS_CELL:
            return RecipeStep.pockels_cell(self.state_combo.currentIndex() == 1)
        if kind == StepKind.ATTENUATOR_SET:
            return RecipeStep.attenuator_set(self._boxes["attenuator"].value())
        return RecipeStep.wait(self._boxes["wait"].value())


class RecipeList(QtWidgets.QListWidget):
    """Sequence list with drag autoscroll and a visible insertion point.

    Qt's own autoscroll only runs for drags it recognises, so dragging a module
    in from the palette used to pin the view in place. The list therefore drives
    its own edge scrolling for every drag type and paints its own insertion
    marker, which also makes the drop position unambiguous.
    """

    orderChanged = QtCore.Signal()
    moduleDropped = QtCore.Signal(str, int)
    editRequested = QtCore.Signal()
    duplicateRequested = QtCore.Signal()
    deleteRequested = QtCore.Signal()
    moveRequested = QtCore.Signal(int)

    AUTOSCROLL_MARGIN_PX = 52
    AUTOSCROLL_MAX_PX = 26

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.setAlternatingRowColors(True)
        self.setSpacing(4)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(False)
        self.setAutoScroll(False)
        self.setVerticalScrollMode(
            QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setItemDelegate(RecipeBlockDelegate(self))

        self._drop_row: int | None = None
        self._scroll_speed = 0
        self._scroll_timer = QtCore.QTimer(self)
        self._scroll_timer.setInterval(30)
        self._scroll_timer.timeout.connect(self._auto_scroll_step)

    # ------------------------------------------------------------ autoscroll
    def _auto_scroll_step(self) -> None:
        bar = self.verticalScrollBar()
        before = bar.value()
        bar.setValue(before + self._scroll_speed)
        if bar.value() == before:
            self._scroll_timer.stop()

    def _update_auto_scroll(self, point: QtCore.QPoint) -> None:
        height = self.viewport().height()
        margin = min(self.AUTOSCROLL_MARGIN_PX, max(16, height // 4))
        speed = 0
        if point.y() < margin:
            depth = (margin - point.y()) / margin
            speed = -max(2, int(depth * self.AUTOSCROLL_MAX_PX))
        elif point.y() > height - margin:
            depth = (point.y() - (height - margin)) / margin
            speed = max(2, int(depth * self.AUTOSCROLL_MAX_PX))
        self._scroll_speed = speed
        if speed and not self._scroll_timer.isActive():
            self._scroll_timer.start()
        elif not speed:
            self._scroll_timer.stop()

    def _stop_auto_scroll(self) -> None:
        self._scroll_timer.stop()
        self._scroll_speed = 0

    # ---------------------------------------------------------- drag & drop
    @staticmethod
    def _is_module_drop(event) -> bool:
        return event.mimeData().hasFormat(RECIPE_MODULE_MIME)

    def _row_at(self, point: QtCore.QPoint) -> int:
        index = self.indexAt(point)
        if index.isValid():
            rect = self.visualRect(index)
            return index.row() + (1 if point.y() > rect.center().y() else 0)
        # The pointer is in the spacing between two blocks or in empty space.
        # Falling through to "append" here is what made precise drops feel
        # unreliable, so resolve the nearest insertion point instead.
        for row in range(self.count()):
            rect = self.visualRect(self.model().index(row, 0))
            if rect.isEmpty():
                continue
            if point.y() < rect.center().y():
                return row
        return self.count()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self._is_module_drop(event):
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        point = event.position().toPoint()
        self._update_auto_scroll(point)
        self._drop_row = self._row_at(point)
        self.viewport().update()
        if self._is_module_drop(event):
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dragMoveEvent(event)
        event.accept()

    def dragLeaveEvent(self, event: QtGui.QDragLeaveEvent) -> None:
        self._stop_auto_scroll()
        self._drop_row = None
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        self._stop_auto_scroll()
        self._drop_row = None
        self.viewport().update()
        if self._is_module_drop(event):
            key = bytes(
                event.mimeData().data(RECIPE_MODULE_MIME)
            ).decode("utf-8")
            row = self._row_at(event.position().toPoint())
            self.moduleDropped.emit(key, row)
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()
            return
        super().dropEvent(event)
        self.orderChanged.emit()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        try:
            self._paint_drop_indicator()
        except Exception:
            self._drop_row = None

    def _paint_drop_indicator(self) -> None:
        if self._drop_row is None:
            return
        painter = QtGui.QPainter(self.viewport())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.count() == 0:
            y = 12
        elif self._drop_row >= self.count():
            y = self.visualRect(self.model().index(self.count() - 1, 0)).bottom() + 3
        else:
            y = self.visualRect(self.model().index(self._drop_row, 0)).top() - 3
        y = max(3, min(self.viewport().height() - 3, y))
        pen = QtGui.QPen(QtGui.QColor("#6ec1f0"), 3)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(8, y, self.viewport().width() - 8, y)
        painter.setBrush(QtGui.QColor("#6ec1f0"))
        painter.drawEllipse(QtCore.QPointF(8, y), 3.5, 3.5)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        modifiers = event.modifiers()
        if event.key() in (
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
        ):
            self.editRequested.emit()
            return
        if event.key() in (
            QtCore.Qt.Key.Key_Delete,
            QtCore.Qt.Key.Key_Backspace,
        ):
            self.deleteRequested.emit()
            return
        if (
            event.key() == QtCore.Qt.Key.Key_D
            and modifiers & QtCore.Qt.KeyboardModifier.ControlModifier
        ):
            self.duplicateRequested.emit()
            return
        if (
            event.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down)
            and modifiers
            & (
                QtCore.Qt.KeyboardModifier.AltModifier
                | QtCore.Qt.KeyboardModifier.ControlModifier
            )
        ):
            self.moveRequested.emit(
                -1 if event.key() == QtCore.Qt.Key.Key_Up else 1
            )
            return
        super().keyPressEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    """Standalone manual controller + script builder for HXP and process beam."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hexapod + Laser Lab — Standalone Controller")
        self.setMinimumSize(1040, 600)
        self._fit_to_screen(preferred=(1720, 1020))

        self.profile = RigProfile.from_json(DEFAULT_PROFILE)
        self.kinematics = RigKinematics(self.profile)
        self.workspace_limits = WorkspaceLimits.cad_conservative(
            self.kinematics
        )
        self.legacy_profile: LegacyHardwareProfile = (
            load_legacy_hardware_profile(LEGACY_EVIDENCE)
        )
        self.controller_profile: HXPControllerProfile = (
            load_hxp_controller_profile(CONTROLLER_PROFILE)
        )
        self._selected_legacy_candidate_key = "labview_v3"

        self.virtual_stage = VirtualHexapodProvider(
            motion_validator=self._validate_motion_path
        )
        self.virtual_stage.connect()
        self.real_stage: HXPProvider | None = None

        self.virtual_laser = VirtualLaserGate()
        self.virtual_laser.connect()
        self.real_laser: HXPDigitalLaserGate | None = None

        self.virtual_attenuator = VirtualAttenuatorProvider(0.0)
        self.virtual_attenuator.connect()
        self.real_attenuator: AttenuatorProvider = (
            UnconfiguredAttenuatorProvider()
        )

        self._last_stage_snapshot = self.virtual_stage.snapshot()
        self._last_tick_monotonic = time.monotonic()
        self._tick_failures = 0
        self._poll_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="hxp-poll",
        )
        self._poll_future: Future | None = None
        self._selftest_future: Future | None = None
        self._last_real_poll = 0.0
        self._real_poll_failures = 0
        self._real_frames_match_profile = False
        self._beam_close_failed = False

        # Measured sample extent. Nothing is assumed until the operator drives
        # the stage to real sample edges and logs them.
        self.sample_calibration = SampleCalibration()

        self.recipe = Recipe()
        self._recipe_running = False
        self._recipe_index = 0
        self._recipe_step_issued = False
        self._recipe_started_monotonic = 0.0
        self._wait_until = 0.0
        self._recipe_write_block_open = False

        # Manual "LINE Move_While Write" mirrors the recovered LabVIEW workflow.
        self._manual_write_line_active = False
        self._manual_write_line_started = False
        self._manual_write_line_description = ""

        self._build_ui()
        self._apply_workspace_limits_to_widgets()
        self._install_wheel_guards()
        self._apply_style()
        self._load_default_config()
        self._apply_mock_profile()
        self._autoload_step_if_present()
        self._refresh_sample_views()
        self._update_sequence_controls()
        self._update_recipe_preflight_view()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.statusBar().showMessage(
            "Virtual stage, Pockels cell and attenuator ready"
        )
        keys = QtWidgets.QLabel("KEYS")
        keys.setObjectName("headerLabel")
        keys.setToolTip(
            "<br>".join(
                f"<b>{sequence}</b> &nbsp; {tip}"
                for sequence, tip in self._shortcut_help
            )
        )
        self.statusBar().addPermanentWidget(keys)

    def _fit_to_screen(
        self,
        *,
        preferred: tuple[int, int],
        margin_px: int = 8,
    ) -> None:
        """Open at the preferred size, but never larger than the screen.

        Laboratory machines are often 1280x800 (or a scaled panel that Qt
        reports as such), which is smaller than the comfortable layout size.
        Clamping here keeps every column and the run bar reachable instead of
        letting the window extend past the desktop.
        """

        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(*preferred)
            return
        available = screen.availableGeometry()
        # Leave room for the title bar and borders, which cannot be measured
        # until the window is mapped.
        frame_allowance = 56
        width = min(preferred[0], max(0, available.width() - 2 * margin_px))
        height = min(
            preferred[1],
            max(0, available.height() - frame_allowance - margin_px),
        )
        width = max(width, self.minimumWidth())
        height = max(height, self.minimumHeight())
        self.resize(width, height)
        self.move(
            available.left() + max(0, (available.width() - width) // 2),
            available.top() + max(0, (available.height() - height) // 2),
        )


    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(7)

        root.addLayout(self._build_header())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._build_manual_tab(), "CONTROL")
        self.tabs.addTab(self._build_script_tab(), "SCRIPT BUILDER")
        self.tabs.addTab(self._build_commissioning_tab(), "COMMISSIONING")
        self.tabs.addTab(self._build_setup_tab(), "SETUP + DIAGNOSTICS")
        root.addWidget(self.tabs, 1)
        self._install_shortcuts()

    def _install_shortcuts(self) -> None:
        """Keyboard access for the actions an operator repeats all day."""

        self._shortcut_help: list[tuple[str, str]] = []

        def bind(sequence: str, slot, tip: str) -> None:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(slot)
            self._shortcut_help.append((sequence, tip))

        for index, name in enumerate(
            ("CONTROL", "SCRIPT BUILDER", "COMMISSIONING", "SETUP")
        ):
            bind(
                f"Ctrl+{index + 1}",
                lambda i=index: self.tabs.setCurrentIndex(i),
                f"Show {name}",
            )
        bind("Ctrl+B", self._close_all_pockels, "Close the process beam")
        bind("Esc", self._abort_all, "Stop motion and close the beam")
        bind("Ctrl+R", self._run_recipe, "Run the loaded script")
        bind("Ctrl+.", self._stop_recipe, "Stop the running script")
        bind("Ctrl+E", self._capture_sample_point, "Log a sample edge point")

    def _build_header(self) -> QtWidgets.QVBoxLayout:
        """Build a two-row header that remains readable on scaled displays."""

        header = QtWidgets.QVBoxLayout()
        header.setSpacing(5)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        title_box = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("HEXAPOD + LASER LAB")
        title.setObjectName("title")
        subtitle = QtWidgets.QLabel(
            "Manual control • digital twin • process-beam control"
        )
        subtitle.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        controls.addLayout(title_box)

        controls.addStretch(1)

        mode_label = QtWidgets.QLabel("Lab mode")
        mode_label.setObjectName("headerLabel")
        controls.addWidget(mode_label)
        self.lab_mode = QtWidgets.QComboBox()
        self.lab_mode.addItems(["MOCK LAB", "REAL LAB"])
        self.lab_mode.setMinimumWidth(105)
        self.lab_mode.currentIndexChanged.connect(self._lab_mode_changed)
        controls.addWidget(self.lab_mode)

        self.mode_chip = QtWidgets.QLabel("MOCK • SAFE")
        self.mode_chip.setObjectName("chipSafe")

        profile_label = QtWidgets.QLabel("Legacy profile")
        profile_label.setObjectName("headerLabel")
        controls.addWidget(profile_label)
        self.legacy_profile_combo = QtWidgets.QComboBox()
        for candidate in self.legacy_profile.pockels_candidates:
            self.legacy_profile_combo.addItem(candidate.label, candidate.key)
        self.legacy_profile_combo.currentIndexChanged.connect(
            self._legacy_profile_changed
        )
        self.legacy_profile_combo.setMinimumWidth(190)
        self.legacy_profile_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        controls.addWidget(self.legacy_profile_combo)

        mock_speed_label = QtWidgets.QLabel("Mock speed")
        mock_speed_label.setObjectName("headerLabel")
        controls.addWidget(mock_speed_label)
        self.mock_time_scale = QtWidgets.QDoubleSpinBox()
        self.mock_time_scale.setRange(0.1, 50.0)
        self.mock_time_scale.setDecimals(1)
        self.mock_time_scale.setSingleStep(0.5)
        self.mock_time_scale.setValue(1.0)
        self.mock_time_scale.setPrefix("×")
        self.mock_time_scale.setToolTip(
            "Accelerates the virtual simulation clock only. "
            "It never changes real HXP motion."
        )
        self.mock_time_scale.setMinimumWidth(100)
        self.mock_time_scale.setMaximumWidth(110)
        controls.addWidget(self.mock_time_scale)

        header.addLayout(controls)

        status = QtWidgets.QHBoxLayout()
        status.setSpacing(8)
        status_label = QtWidgets.QLabel("LIVE STATUS")
        status_label.setObjectName("headerLabel")
        status.addWidget(status_label)
        status.addStretch(1)

        # Internal stage-provider selector is retained for routing/config
        # compatibility, but ordinary users operate through the single MOCK/REAL
        # lab-mode selector above. Hiding the duplicate selector keeps the header
        # unambiguous.
        stage_label = QtWidgets.QLabel("Stage")
        stage_label.setObjectName("headerLabel")
        stage_label.setVisible(False)
        status.addWidget(stage_label)
        self.stage_mode = QtWidgets.QComboBox()
        self.stage_mode.addItems(["Virtual", "Real Newport HXP"])
        self.stage_mode.currentIndexChanged.connect(
            self._stage_mode_changed
        )
        self.stage_mode.setVisible(False)
        status.addWidget(self.stage_mode)

        self.stage_chip = QtWidgets.QLabel("STAGE IDLE")
        self.stage_chip.setObjectName("chipNeutral")
        status.addWidget(self.mode_chip)
        status.addWidget(self.stage_chip)

        self.beam_chip = QtWidgets.QLabel("BEAM CLOSED")
        self.beam_chip.setObjectName("chipSafe")
        status.addWidget(self.beam_chip)

        self.attenuator_chip = QtWidgets.QLabel("ATT 0.0 %")
        self.attenuator_chip.setObjectName("chipNeutral")
        status.addWidget(self.attenuator_chip)

        self.sample_chip = QtWidgets.QLabel("SAMPLE UNCALIBRATED")
        self.sample_chip.setObjectName("chipNeutral")
        self.sample_chip.setToolTip(
            "Sample edge calibration and bound state (CONTROL tab)"
        )
        status.addWidget(self.sample_chip)

        for chip in (
            self.mode_chip,
            self.stage_chip,
            self.beam_chip,
            self.attenuator_chip,
            self.sample_chip,
        ):
            chip.setMinimumWidth(96)
            chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.close_beam_header = QtWidgets.QPushButton("CLOSE BEAM")
        self.close_beam_header.setObjectName("safeAction")
        self.close_beam_header.setToolTip(
            "Request Pockels cell CLOSED. This does not power off the PHAROS."
        )
        self.close_beam_header.clicked.connect(self._close_all_pockels)
        self.close_beam_header.setMinimumWidth(110)
        status.addWidget(self.close_beam_header)

        self.stop_header = QtWidgets.QPushButton(
            "STOP MOTION + CLOSE BEAM"
        )
        self.stop_header.setObjectName("danger")
        self.stop_header.setToolTip(
            "Software abort: request Pockels CLOSED and send HXP GroupMoveAbort. "
            "This is not a hardware emergency stop."
        )
        self.stop_header.clicked.connect(self._abort_all)
        self.stop_header.setMinimumWidth(190)
        status.addWidget(self.stop_header)

        header.addLayout(status)

        return header

    def _build_manual_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        motion_scroll = self._scroll_wrap(self._build_motion_panel())
        motion_scroll.setMinimumWidth(300)
        split.addWidget(motion_scroll)
        split.addWidget(self._build_visual_panel())
        process_scroll = self._scroll_wrap(self._build_process_panel())
        process_scroll.setMinimumWidth(290)
        split.addWidget(process_scroll)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([330, 600, 310])
        layout.addWidget(split, 1)
        layout.addWidget(self._build_home_script_bar())
        return tab

    def _build_home_script_bar(self) -> QtWidgets.QWidget:
        """Run the loaded script without leaving the control screen."""

        bar = QtWidgets.QFrame()
        bar.setObjectName("runBar")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(9)

        heading = QtWidgets.QLabel("SCRIPT")
        heading.setObjectName("headerLabel")
        row.addWidget(heading)

        self.home_script_name = QtWidgets.QLabel("No script loaded")
        self.home_script_name.setObjectName("mono")
        self.home_script_name.setMinimumWidth(200)
        row.addWidget(self.home_script_name)

        self.home_preflight_chip = QtWidgets.QLabel("EMPTY")
        self.home_preflight_chip.setObjectName("chipNeutral")
        self.home_preflight_chip.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.home_preflight_chip.setMinimumWidth(128)
        row.addWidget(self.home_preflight_chip)

        self.home_script_progress = QtWidgets.QProgressBar()
        self.home_script_progress.setRange(0, 100)
        self.home_script_progress.setValue(0)
        self.home_script_progress.setTextVisible(True)
        self.home_script_progress.setFormat("idle")
        self.home_script_progress.setMinimumWidth(220)
        row.addWidget(self.home_script_progress, 1)

        load = QtWidgets.QPushButton("Load…")
        load.setToolTip("Load a saved recipe JSON")
        load.clicked.connect(self._load_recipe)
        row.addWidget(load)

        builder = QtWidgets.QPushButton("Open builder")
        builder.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        row.addWidget(builder)

        self.home_run_btn = QtWidgets.QPushButton("▶  RUN SCRIPT")
        self.home_run_btn.setObjectName("primary")
        self.home_run_btn.setMinimumHeight(38)
        self.home_run_btn.setMinimumWidth(170)
        self.home_run_btn.setToolTip(
            "Run the loaded script on the current providers (Ctrl+R)"
        )
        self.home_run_btn.clicked.connect(self._run_recipe)
        row.addWidget(self.home_run_btn)

        self.home_stop_btn = QtWidgets.QPushButton("■  STOP SCRIPT")
        self.home_stop_btn.setObjectName("danger")
        self.home_stop_btn.setMinimumHeight(38)
        self.home_stop_btn.clicked.connect(self._stop_recipe)
        row.addWidget(self.home_stop_btn)
        return bar

    def _build_motion_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(9)

        stage_box = QtWidgets.QGroupBox("STAGE CONNECTION")
        stage_layout = QtWidgets.QVBoxLayout(stage_box)
        self.manual_stage_status = QtWidgets.QLabel(
            "Virtual stage connected"
        )
        self.manual_stage_status.setObjectName("statusPill")
        stage_layout.addWidget(self.manual_stage_status)
        row = QtWidgets.QHBoxLayout()
        self.connect_stage_btn = QtWidgets.QPushButton("Connect")
        self.connect_stage_btn.clicked.connect(self._connect_stage)
        self.disconnect_stage_btn = QtWidgets.QPushButton("Disconnect")
        self.disconnect_stage_btn.clicked.connect(self._disconnect_stage)
        row.addWidget(self.connect_stage_btn)
        row.addWidget(self.disconnect_stage_btn)
        stage_layout.addLayout(row)
        hint = QtWidgets.QLabel(
            "Real-HXP network settings live in Setup + Diagnostics."
        )
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        stage_layout.addWidget(hint)
        layout.addWidget(stage_box)

        target_box = QtWidgets.QGroupBox("TARGET POSE")
        target_layout = QtWidgets.QVBoxLayout(target_box)
        pose_grid = QtWidgets.QGridLayout()
        self.pose_boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for i, axis in enumerate("XYZUVW"):
            box = self._pose_spinbox(i)
            self.pose_boxes[axis] = box
            r = i % 3
            c = (i // 3) * 2
            pose_grid.addWidget(QtWidgets.QLabel(axis), r, c)
            pose_grid.addWidget(box, r, c + 1)
        target_layout.addLayout(pose_grid)

        self.move_abs_btn = QtWidgets.QPushButton("MOVE TO TARGET")
        self.move_abs_btn.setObjectName("primary")
        self.move_abs_btn.setMinimumHeight(34)
        self.move_abs_btn.clicked.connect(self._move_absolute)
        copy_btn = QtWidgets.QPushButton("Target ← actual")
        copy_btn.setToolTip("Copy the live pose into the target boxes")
        copy_btn.clicked.connect(self._target_from_actual)
        target_layout.addWidget(self.move_abs_btn)
        target_layout.addWidget(copy_btn)
        layout.addWidget(target_box)

        jog_box = QtWidgets.QGroupBox("JOG")
        jog_layout = QtWidgets.QVBoxLayout(jog_box)
        steps = QtWidgets.QFormLayout()
        self.jog_mm = QtWidgets.QDoubleSpinBox()
        self.jog_mm.setRange(0.0001, 10.0)
        self.jog_mm.setDecimals(4)
        self.jog_mm.setValue(0.1)
        self.jog_mm.setSuffix(" mm")
        self.jog_deg = QtWidgets.QDoubleSpinBox()
        self.jog_deg.setRange(0.0001, 5.0)
        self.jog_deg.setDecimals(4)
        self.jog_deg.setValue(0.1)
        self.jog_deg.setSuffix(" °")
        steps.addRow("XYZ step", self.jog_mm)
        steps.addRow("UVW step", self.jog_deg)
        jog_layout.addLayout(steps)

        jog_grid = QtWidgets.QGridLayout()
        jog_grid.setSpacing(4)
        self.jog_buttons: list[QtWidgets.QPushButton] = []
        for row, axis in enumerate("XYZUVW"):
            label = QtWidgets.QLabel(axis)
            label.setObjectName("mono")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            minus = QtWidgets.QPushButton(f"− {axis}")
            plus = QtWidgets.QPushButton(f"{axis} +")
            for button, sign in ((minus, -1.0), (plus, +1.0)):
                button.setMinimumHeight(30)
                # Hold to keep stepping, as on the physical jog pendant.
                button.setAutoRepeat(True)
                button.setAutoRepeatDelay(450)
                button.setAutoRepeatInterval(180)
                button.clicked.connect(
                    lambda _=False, a=axis, sg=sign: self._jog(a, sg)
                )
                self.jog_buttons.append(button)
            jog_grid.addWidget(minus, row, 0)
            jog_grid.addWidget(label, row, 1)
            jog_grid.addWidget(plus, row, 2)
        jog_grid.setColumnStretch(0, 1)
        jog_grid.setColumnStretch(2, 1)
        jog_layout.addLayout(jog_grid)
        jog_hint = QtWidgets.QLabel(
            "Hold a jog button to keep stepping."
        )
        jog_hint.setObjectName("muted")
        jog_layout.addWidget(jog_hint)
        layout.addWidget(jog_box)
        layout.addWidget(self._build_sample_box())

        quick_box = QtWidgets.QGroupBox("QUICK LINE / WRITING MOVE")
        quick_layout = QtWidgets.QVBoxLayout(quick_box)
        quick_note = QtWidgets.QLabel(
            "Mirrors the recovered LabVIEW LINE Move / LINE Move_While Write "
            "workflow using native HXP Line + target velocity."
        )
        quick_note.setObjectName("muted")
        quick_note.setWordWrap(True)
        quick_layout.addWidget(quick_note)

        quick_form = QtWidgets.QFormLayout()
        self.quick_dx = QtWidgets.QDoubleSpinBox()
        x_span = self.workspace_limits.maximum_pose.x - self.workspace_limits.minimum_pose.x
        y_span = self.workspace_limits.maximum_pose.y - self.workspace_limits.minimum_pose.y
        z_span = self.workspace_limits.maximum_pose.z - self.workspace_limits.minimum_pose.z
        self.quick_dx.setRange(-x_span, x_span)
        self.quick_dx.setDecimals(4)
        self.quick_dx.setValue(-7.0)
        self.quick_dx.setSuffix(" mm")
        self.quick_dy = QtWidgets.QDoubleSpinBox()
        self.quick_dy.setRange(-y_span, y_span)
        self.quick_dy.setDecimals(4)
        self.quick_dy.setValue(0.0)
        self.quick_dy.setSuffix(" mm")
        self.quick_dz = QtWidgets.QDoubleSpinBox()
        self.quick_dz.setRange(-z_span, z_span)
        self.quick_dz.setDecimals(4)
        self.quick_dz.setValue(0.0)
        self.quick_dz.setSuffix(" mm")
        self.quick_velocity = QtWidgets.QDoubleSpinBox()
        self.quick_velocity.setRange(
            0.001,
            self.workspace_limits.maximum_line_velocity_mm_s,
        )
        self.quick_velocity.setDecimals(3)
        self.quick_velocity.setValue(1.0)
        self.quick_velocity.setSuffix(" mm/s")
        self.quick_row_pitch = QtWidgets.QDoubleSpinBox()
        self.quick_row_pitch.setRange(-y_span, y_span)
        self.quick_row_pitch.setDecimals(4)
        self.quick_row_pitch.setValue(0.02)
        self.quick_row_pitch.setSuffix(" mm")
        quick_form.addRow("dX", self.quick_dx)
        quick_form.addRow("dY", self.quick_dy)
        quick_form.addRow("dZ", self.quick_dz)
        quick_form.addRow("Velocity", self.quick_velocity)
        quick_form.addRow("Return row pitch", self.quick_row_pitch)
        quick_layout.addLayout(quick_form)

        quick_buttons = QtWidgets.QGridLayout()
        self.quick_move_btn = QtWidgets.QPushButton("MOVE LINE")
        self.quick_move_btn.clicked.connect(
            lambda: self._start_quick_line(write=False)
        )
        self.quick_write_btn = QtWidgets.QPushButton(
            "WRITE LINE\nPOCKELS OPEN DURING MOVE"
        )
        self.quick_write_btn.setObjectName("beamOnSmall")
        self.quick_write_btn.clicked.connect(
            lambda: self._start_quick_line(write=True)
        )
        self.quick_return_btn = QtWidgets.QPushButton(
            "RETURN + ROW\nBEAM CLOSED"
        )
        self.quick_return_btn.setObjectName("safeAction")
        self.quick_return_btn.clicked.connect(self._quick_return_row)
        quick_buttons.addWidget(self.quick_move_btn, 0, 0)
        quick_buttons.addWidget(self.quick_write_btn, 0, 1)
        quick_buttons.addWidget(self.quick_return_btn, 1, 0, 1, 2)
        quick_layout.addLayout(quick_buttons)

        self.quick_line_status = QtWidgets.QLabel("Ready")
        self.quick_line_status.setObjectName("statusPill")
        self.quick_line_status.setWordWrap(True)
        quick_layout.addWidget(self.quick_line_status)
        layout.addWidget(quick_box)

        actions = QtWidgets.QGroupBox("STAGE ACTIONS")
        actions_layout = QtWidgets.QHBoxLayout(actions)
        self.home_btn = QtWidgets.QPushButton("HOME")
        self.home_btn.clicked.connect(self._home)
        self.stop_motion_btn = QtWidgets.QPushButton(
            "STOP + CLOSE BEAM"
        )
        self.stop_motion_btn.setObjectName("danger")
        self.stop_motion_btn.clicked.connect(self._abort_all)
        actions_layout.addWidget(self.home_btn)
        actions_layout.addWidget(self.stop_motion_btn)
        layout.addWidget(actions)
        layout.addStretch(1)
        return panel

    def _build_sample_box(self) -> QtWidgets.QWidget:
        """Log real sample edges and bind motion to the measured area."""

        box = QtWidgets.QGroupBox("SAMPLE EDGE CALIBRATION")
        layout = QtWidgets.QVBoxLayout(box)
        layout.setSpacing(6)

        note = QtWidgets.QLabel(
            "Jog until the beam sits on a physical sample edge/corner, then "
            "CAPTURE. Two opposite corners give a rectangle; three or more give "
            "the measured outline."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.sample_name = QtWidgets.QLineEdit("Sample")
        self.sample_name.setPlaceholderText("Sample / chip identifier")
        self.sample_name.editingFinished.connect(self._apply_sample_settings)
        layout.addWidget(self.sample_name)

        self.capture_sample_btn = QtWidgets.QPushButton(
            "⊕  CAPTURE CORNER"
        )
        self.capture_sample_btn.setObjectName("primary")
        self.capture_sample_btn.setMinimumHeight(36)
        self.capture_sample_btn.setToolTip(
            "Log the current hexapod pose as a sample edge point (Ctrl+E)"
        )
        self.capture_sample_btn.clicked.connect(self._capture_sample_point)
        layout.addWidget(self.capture_sample_btn)

        self.sample_points_list = QtWidgets.QListWidget()
        self.sample_points_list.setObjectName("sampleList")
        self.sample_points_list.setMaximumHeight(118)
        self.sample_points_list.setToolTip(
            "Captured edge points, in capture order"
        )
        layout.addWidget(self.sample_points_list)

        point_row = QtWidgets.QHBoxLayout()
        remove = QtWidgets.QPushButton("Remove selected")
        remove.clicked.connect(self._remove_sample_point)
        clear = QtWidgets.QPushButton("Clear all")
        clear.clicked.connect(self._clear_sample_points)
        goto = QtWidgets.QPushButton("Go to centre")
        goto.setToolTip(
            "Move to the centre of the calibrated area at the fitted surface Z"
        )
        goto.clicked.connect(self._goto_sample_centre)
        point_row.addWidget(remove)
        point_row.addWidget(clear)
        point_row.addWidget(goto)
        layout.addLayout(point_row)

        form = QtWidgets.QFormLayout()
        self.sample_margin = QtWidgets.QDoubleSpinBox()
        self.sample_margin.setRange(0.0, 50.0)
        self.sample_margin.setDecimals(3)
        self.sample_margin.setSingleStep(0.05)
        self.sample_margin.setValue(0.25)
        self.sample_margin.setSuffix(" mm")
        self.sample_margin.setToolTip(
            "Keep-out distance held inside the measured sample edge"
        )
        self.sample_margin.valueChanged.connect(self._apply_sample_settings)
        form.addRow("Edge margin", self.sample_margin)

        z_row = QtWidgets.QHBoxLayout()
        self.sample_z_below = QtWidgets.QDoubleSpinBox()
        self.sample_z_below.setRange(0.0, 50.0)
        self.sample_z_below.setDecimals(3)
        self.sample_z_below.setValue(0.5)
        self.sample_z_below.setPrefix("−")
        self.sample_z_below.setSuffix(" mm")
        self.sample_z_above = QtWidgets.QDoubleSpinBox()
        self.sample_z_above.setRange(0.0, 50.0)
        self.sample_z_above.setDecimals(3)
        self.sample_z_above.setValue(0.5)
        self.sample_z_above.setPrefix("+")
        self.sample_z_above.setSuffix(" mm")
        for widget in (self.sample_z_below, self.sample_z_above):
            widget.valueChanged.connect(self._apply_sample_settings)
            z_row.addWidget(widget)
        form.addRow("Z band about surface", z_row)
        layout.addLayout(form)

        self.sample_bound_xy = QtWidgets.QCheckBox(
            "BIND MOTION TO SAMPLE"
        )
        self.sample_bound_xy.setToolTip(
            "Blocks any XY move whose path leaves the measured outline"
        )
        self.sample_bound_xy.toggled.connect(self._apply_sample_settings)
        layout.addWidget(self.sample_bound_xy)

        self.sample_bound_z = QtWidgets.QCheckBox(
            "Also bind Z to the surface"
        )
        self.sample_bound_z.toggled.connect(self._apply_sample_settings)
        layout.addWidget(self.sample_bound_z)

        self.sample_status = QtWidgets.QLabel(
            "No sample edge points captured"
        )
        self.sample_status.setObjectName("statusPill")
        self.sample_status.setWordWrap(True)
        layout.addWidget(self.sample_status)

        file_row = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save sample…")
        save.clicked.connect(self._save_sample_calibration)
        load = QtWidgets.QPushButton("Load sample…")
        load.clicked.connect(self._load_sample_calibration)
        file_row.addWidget(save)
        file_row.addWidget(load)
        layout.addLayout(file_row)
        return box

    def _build_visual_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(5)

        topbar = QtWidgets.QHBoxLayout()
        topbar.setSpacing(5)
        topbar.addWidget(self._section("LIVE DIGITAL TWIN"))
        for name, label in (
            ("iso", "Iso"),
            ("top", "Top"),
            ("front", "Front"),
            ("side", "Side"),
        ):
            button = QtWidgets.QPushButton(label)
            button.setObjectName("chipButton")
            button.setMaximumWidth(56)
            button.setToolTip(f"Snap the 3D camera to the {label} view")
            button.clicked.connect(lambda _=False, n=name: self.viewer.set_view(n))
            topbar.addWidget(button)
        topbar.addStretch(1)
        self.cad_status = QtWidgets.QLabel("CAD-derived rig")
        self.cad_status.setObjectName("muted")
        topbar.addWidget(self.cad_status)
        load_cad = QtWidgets.QPushButton("Load STEP…")
        load_cad.clicked.connect(self._load_step_dialog)
        topbar.addWidget(load_cad)
        clear = QtWidgets.QPushButton("Clear paths")
        clear.setToolTip("Erase the recorded travel and written traces")
        clear.clicked.connect(self._clear_visual_traces)
        topbar.addWidget(clear)
        layout.addLayout(topbar)

        self.viewer = Hexapod3DViewer(panel, self.profile)
        self.viewer.set_status_callback(self._cad_message)

        map_container = QtWidgets.QWidget()
        map_layout = QtWidgets.QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.setSpacing(3)
        map_bar = QtWidgets.QHBoxLayout()
        map_bar.setSpacing(5)
        map_bar.addWidget(self._section("2D PATH MAP"))

        self.show_travel_check = QtWidgets.QCheckBox("Travel")
        self.show_travel_check.setChecked(True)
        self.show_travel_check.setToolTip(
            "Show beam-OFF repositioning moves (dashed grey)"
        )
        self.show_written_check = QtWidgets.QCheckBox("Written")
        self.show_written_check.setChecked(True)
        self.show_written_check.setToolTip(
            "Show beam-ON written runs (solid amber)"
        )
        self.show_sample_check = QtWidgets.QCheckBox("Sample")
        self.show_sample_check.setChecked(True)
        self.show_sample_check.setToolTip(
            "Show the calibrated sample outline"
        )
        for check in (
            self.show_travel_check,
            self.show_written_check,
            self.show_sample_check,
        ):
            map_bar.addWidget(check)
        map_bar.addStretch(1)

        self.map_mode = QtWidgets.QComboBox()
        self.map_mode.addItems(
            ["Laser path on sample", "HXP XY carriage path"]
        )
        self.map_span = QtWidgets.QDoubleSpinBox()
        self.map_span.setRange(0.05, 500.0)
        self.map_span.setDecimals(2)
        self.map_span.setValue(40.0)
        self.map_span.setSuffix(" mm span")
        self.map_span.setToolTip(
            "Map width. Scroll over the map to zoom; double-click to fit."
        )
        fit = QtWidgets.QPushButton("Fit")
        fit.setObjectName("chipButton")
        fit.setMaximumWidth(48)
        fit.setToolTip("Frame every recorded point and the sample outline")
        fit.clicked.connect(self._fit_map_view)
        map_bar.addWidget(self.map_mode)
        map_bar.addWidget(self.map_span)
        map_bar.addWidget(fit)
        map_layout.addLayout(map_bar)

        self.movement_map = MovementMap2D(map_container)
        self.map_mode.currentIndexChanged.connect(
            lambda i: self.movement_map.set_mode(
                "sample" if i == 0 else "hxp"
            )
        )
        self.map_span.valueChanged.connect(
            self.movement_map.set_span_mm
        )
        self.movement_map.spanChangeRequested.connect(
            lambda span: self.map_span.setValue(
                max(self.map_span.minimum(), min(self.map_span.maximum(), span))
            )
        )
        self.show_travel_check.toggled.connect(self._apply_trace_visibility)
        self.show_written_check.toggled.connect(self._apply_trace_visibility)
        self.show_sample_check.toggled.connect(self._apply_trace_visibility)
        map_layout.addWidget(self.movement_map, 1)

        vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vertical.addWidget(self.viewer)
        vertical.addWidget(map_container)
        vertical.setStretchFactor(0, 3)
        vertical.setStretchFactor(1, 2)
        vertical.setSizes([460, 300])
        layout.addWidget(vertical, 1)
        return panel

    def _build_process_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(9)

        beam_box = QtWidgets.QGroupBox("PROCESS BEAM / POCKELS CELL")
        beam_layout = QtWidgets.QVBoxLayout(beam_box)
        self.manual_beam_status = QtWidgets.QLabel(
            "LASER OFF • POCKELS CLOSED"
        )
        self.manual_beam_status.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.manual_beam_status.setObjectName("laserOff")
        beam_layout.addWidget(self.manual_beam_status)

        explanation = QtWidgets.QLabel(
            "LASER ON/OFF here means opening/closing the process-beam "
            "Pockels cell. It does not power-cycle the PHAROS source."
        )
        explanation.setObjectName("muted")
        explanation.setWordWrap(True)
        beam_layout.addWidget(explanation)

        self.manual_beam_arm = QtWidgets.QCheckBox(
            "ARM REAL-BEAM CONTROL"
        )
        self.manual_beam_arm.setObjectName("dangerCheck")
        self.manual_beam_arm.setToolTip(
            "Required only when the real LX13/Pockels provider is selected."
        )
        beam_layout.addWidget(self.manual_beam_arm)

        self.laser_on_btn = QtWidgets.QPushButton(
            "LASER ON\nOPEN POCKELS CELL"
        )
        self.laser_on_btn.setObjectName("beamOn")
        self.laser_on_btn.setMinimumHeight(62)
        self.laser_on_btn.clicked.connect(
            lambda: self._set_pockels(True, manual=True)
        )
        beam_layout.addWidget(self.laser_on_btn)

        self.laser_off_btn = QtWidgets.QPushButton(
            "LASER OFF\nCLOSE POCKELS CELL"
        )
        self.laser_off_btn.setObjectName("beamOff")
        self.laser_off_btn.setMinimumHeight(58)
        self.laser_off_btn.clicked.connect(
            lambda: self._set_pockels(False, manual=True)
        )
        beam_layout.addWidget(self.laser_off_btn)

        provider_note = QtWidgets.QLabel(
            "Pockels provider is configured in Setup + Diagnostics."
        )
        provider_note.setObjectName("muted")
        provider_note.setWordWrap(True)
        beam_layout.addWidget(provider_note)
        layout.addWidget(beam_box)

        att_box = QtWidgets.QGroupBox("ATTENUATOR")
        att_layout = QtWidgets.QVBoxLayout(att_box)
        self.attenuator_status = QtWidgets.QLabel(
            "Transmission setpoint: 0.0 %"
        )
        self.attenuator_status.setObjectName("statusPill")
        att_layout.addWidget(self.attenuator_status)

        slider_row = QtWidgets.QHBoxLayout()
        self.attenuator_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal
        )
        self.attenuator_slider.setRange(0, 1000)
        self.attenuator_slider.setValue(0)
        self.attenuator_spin = QtWidgets.QDoubleSpinBox()
        self.attenuator_spin.setRange(0.0, 100.0)
        self.attenuator_spin.setDecimals(1)
        self.attenuator_spin.setSuffix(" %")
        self.attenuator_spin.setValue(0.0)
        self.attenuator_slider.valueChanged.connect(
            lambda v: self.attenuator_spin.setValue(v / 10.0)
        )
        self.attenuator_spin.valueChanged.connect(
            lambda v: self.attenuator_slider.setValue(round(v * 10.0))
        )
        slider_row.addWidget(self.attenuator_slider, 1)
        slider_row.addWidget(self.attenuator_spin)
        att_layout.addLayout(slider_row)

        presets = QtWidgets.QHBoxLayout()
        for value in (0, 25, 50, 75, 100):
            b = QtWidgets.QPushButton(f"{value}%")
            b.clicked.connect(
                lambda _=False, v=value: self._attenuator_preset(v)
            )
            presets.addWidget(b)
        att_layout.addLayout(presets)

        self.set_attenuator_btn = QtWidgets.QPushButton(
            "SET ATTENUATOR"
        )
        self.set_attenuator_btn.setObjectName("primary")
        self.set_attenuator_btn.clicked.connect(
            self._set_attenuator_manual
        )
        att_layout.addWidget(self.set_attenuator_btn)
        att_note = QtWidgets.QLabel(
            "The GUI uses normalized transmission %. A real attenuator "
            "driver/calibration will map this to the device's native control."
        )
        att_note.setObjectName("muted")
        att_note.setWordWrap(True)
        att_layout.addWidget(att_note)
        layout.addWidget(att_box)

        state_box = QtWidgets.QGroupBox("LIVE POSITION")
        state_layout = QtWidgets.QVBoxLayout(state_box)
        self.state_labels: dict[str, QtWidgets.QLabel] = {}
        self.target_labels: dict[str, QtWidgets.QLabel] = {}
        self.error_labels: dict[str, QtWidgets.QLabel] = {}
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel(""), 0, 0)
        grid.addWidget(QtWidgets.QLabel("Actual"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Target"), 0, 2)
        grid.addWidget(QtWidgets.QLabel("Δ"), 0, 3)
        for i, axis in enumerate("XYZUVW", start=1):
            grid.addWidget(QtWidgets.QLabel(axis), i, 0)

            actual = QtWidgets.QLabel("0.0000")
            actual.setObjectName("mono")
            self.state_labels[axis] = actual
            grid.addWidget(actual, i, 1)

            target = QtWidgets.QLabel("0.0000")
            target.setObjectName("mono")
            self.target_labels[axis] = target
            grid.addWidget(target, i, 2)

            error = QtWidgets.QLabel("0.0000")
            error.setObjectName("mono")
            self.error_labels[axis] = error
            grid.addWidget(error, i, 3)
        state_layout.addLayout(grid)
        self.motion_status = QtWidgets.QLabel("IDLE")
        self.motion_status.setObjectName("statusPill")
        state_layout.addWidget(self.motion_status)
        layout.addWidget(state_box)

        leg_box = QtWidgets.QGroupBox("ACTUATOR GEOMETRY")
        leg_layout = QtWidgets.QGridLayout(leg_box)
        self.leg_labels: list[QtWidgets.QLabel] = []
        for i in range(6):
            leg_layout.addWidget(QtWidgets.QLabel(f"Strut {i + 1}"), i, 0)
            val = QtWidgets.QLabel("— mm")
            val.setObjectName("mono")
            self.leg_labels.append(val)
            leg_layout.addWidget(val, i, 1)
        layout.addWidget(leg_box)
        layout.addStretch(1)
        return panel

    def _build_script_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        palette_scroll = self._scroll_wrap(self._build_script_palette())
        palette_scroll.setMinimumWidth(270)
        split.addWidget(palette_scroll)
        split.addWidget(self._build_script_sequence())
        run_panel = self._build_script_run_panel()
        run_panel.setMinimumWidth(280)
        split.addWidget(run_panel)
        split.setSizes([300, 570, 310])
        layout.addWidget(split)
        return tab

    def _build_script_palette(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        library_box = QtWidgets.QGroupBox(
            "MODULE LIBRARY — DRAG INTO SEQUENCE"
        )
        library_layout = QtWidgets.QVBoxLayout(library_box)
        library_hint = QtWidgets.QLabel(
            "Drag a module into any position. Its current defaults can be "
            "edited after insertion by double-clicking the block."
        )
        library_hint.setObjectName("muted")
        library_hint.setWordWrap(True)
        library_layout.addWidget(library_hint)
        self.module_library = ModuleLibrary()
        self.module_library.itemDoubleClicked.connect(
            lambda item: self._recipe_module_dropped(
                str(item.data(QtCore.Qt.ItemDataRole.UserRole)),
                self.recipe_list.count(),
            )
        )
        library_layout.addWidget(self.module_library)
        layout.addWidget(library_box)

        move_box = QtWidgets.QGroupBox("MOTION DEFAULTS / QUICK ADD")
        move_layout = QtWidgets.QVBoxLayout(move_box)
        self.script_move_kind = QtWidgets.QComboBox()
        self.script_move_kind.addItems(
            [
                "Absolute pose",
                "Relative move",
                "Line move at target velocity",
                "Move while write (beam open for the move)",
            ]
        )
        self.script_move_kind.currentIndexChanged.connect(
            self._script_move_kind_changed
        )
        move_layout.addWidget(self.script_move_kind)
        pose_grid = QtWidgets.QGridLayout()
        self.script_pose_boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for i, axis in enumerate("XYZUVW"):
            box = self._pose_spinbox(i)
            self.script_pose_boxes[axis] = box
            r = i % 3
            c = (i // 3) * 2
            pose_grid.addWidget(QtWidgets.QLabel(axis), r, c)
            pose_grid.addWidget(box, r, c + 1)
        move_layout.addLayout(pose_grid)

        speed_row = QtWidgets.QFormLayout()
        self.script_line_velocity = QtWidgets.QDoubleSpinBox()
        self.script_line_velocity.setRange(
            0.001,
            self.workspace_limits.maximum_line_velocity_mm_s,
        )
        self.script_line_velocity.setDecimals(3)
        self.script_line_velocity.setValue(1.0)
        self.script_line_velocity.setSuffix(" mm/s")
        self.script_line_velocity.setToolTip(
            "Used only for 'Line move at target velocity'. "
            "This maps to the legacy HXP "
            "HexapodMoveIncrementalControlWithTargetVelocity command."
        )
        speed_row.addRow("Line velocity", self.script_line_velocity)
        move_layout.addLayout(speed_row)

        row = QtWidgets.QHBoxLayout()
        add_move = QtWidgets.QPushButton("+ ADD MOVE")
        add_move.setObjectName("primary")
        add_move.clicked.connect(self._recipe_add_move)
        copy_live = QtWidgets.QPushButton("Pose ← live")
        copy_live.setToolTip("Copy the live stage pose into these boxes")
        copy_live.clicked.connect(self._script_pose_from_live)
        row.addWidget(add_move, 2)
        row.addWidget(copy_live, 1)
        move_layout.addLayout(row)

        quick_write = QtWidgets.QPushButton(
            "+ MOVE WHILE WRITE  —  dX/dY/dZ at line velocity"
        )
        quick_write.setObjectName("beamOnSmall")
        quick_write.setToolTip(
            "One block that opens the Pockels cell, runs the line and closes "
            "the cell again when the move ends"
        )
        quick_write.clicked.connect(
            lambda: self._append_recipe_step(
                self._default_step_for_module("write_line")
            )
        )
        move_layout.addWidget(quick_write)
        layout.addWidget(move_box)

        beam_box = QtWidgets.QGroupBox("ADD BEAM STEP")
        beam_layout = QtWidgets.QVBoxLayout(beam_box)
        on = QtWidgets.QPushButton("+ LASER ON / POCKELS OPEN")
        on.setObjectName("beamOnSmall")
        on.clicked.connect(
            lambda: self._append_recipe_step(
                RecipeStep.pockels_cell(True)
            )
        )
        off = QtWidgets.QPushButton("+ LASER OFF / POCKELS CLOSED")
        off.setObjectName("beamOffSmall")
        off.clicked.connect(
            lambda: self._append_recipe_step(
                RecipeStep.pockels_cell(False)
            )
        )
        beam_layout.addWidget(on)
        beam_layout.addWidget(off)
        layout.addWidget(beam_box)

        att_box = QtWidgets.QGroupBox("ADD ATTENUATOR STEP")
        att_layout = QtWidgets.QHBoxLayout(att_box)
        self.script_attenuator = QtWidgets.QDoubleSpinBox()
        self.script_attenuator.setRange(0.0, 100.0)
        self.script_attenuator.setDecimals(1)
        self.script_attenuator.setSuffix(" %")
        add_att = QtWidgets.QPushButton("+ ADD")
        add_att.clicked.connect(self._recipe_add_attenuator)
        att_layout.addWidget(self.script_attenuator)
        att_layout.addWidget(add_att)
        layout.addWidget(att_box)

        wait_box = QtWidgets.QGroupBox("ADD WAIT")
        wait_layout = QtWidgets.QHBoxLayout(wait_box)
        self.script_wait = QtWidgets.QDoubleSpinBox()
        self.script_wait.setRange(0.0, 3600.0)
        self.script_wait.setDecimals(3)
        self.script_wait.setValue(1.0)
        self.script_wait.setSuffix(" s")
        add_wait = QtWidgets.QPushButton("+ ADD")
        add_wait.clicked.connect(self._recipe_add_wait)
        wait_layout.addWidget(self.script_wait)
        wait_layout.addWidget(add_wait)
        layout.addWidget(wait_box)

        sweep_box = QtWidgets.QGroupBox("RASTER / PARAMETER SWEEP")
        sweep_layout = QtWidgets.QFormLayout(sweep_box)

        self.sweep_write_dx = QtWidgets.QDoubleSpinBox()
        x_span = self.workspace_limits.maximum_pose.x - self.workspace_limits.minimum_pose.x
        y_span = self.workspace_limits.maximum_pose.y - self.workspace_limits.minimum_pose.y
        self.sweep_write_dx.setRange(-x_span, x_span)
        self.sweep_write_dx.setDecimals(3)
        self.sweep_write_dx.setValue(-7.0)
        self.sweep_write_dx.setSuffix(" mm")

        self.sweep_row_pitch = QtWidgets.QDoubleSpinBox()
        self.sweep_row_pitch.setRange(-y_span, y_span)
        self.sweep_row_pitch.setDecimals(4)
        self.sweep_row_pitch.setValue(0.02)
        self.sweep_row_pitch.setSuffix(" mm")

        self.sweep_series_spacing = QtWidgets.QDoubleSpinBox()
        self.sweep_series_spacing.setRange(0.0, y_span)
        self.sweep_series_spacing.setDecimals(4)
        self.sweep_series_spacing.setValue(0.10)
        self.sweep_series_spacing.setSuffix(" mm")

        self.sweep_v_start = QtWidgets.QDoubleSpinBox()
        self.sweep_v_start.setRange(
            0.001,
            self.workspace_limits.maximum_line_velocity_mm_s,
        )
        self.sweep_v_start.setDecimals(3)
        self.sweep_v_start.setValue(0.2)
        self.sweep_v_start.setSuffix(" mm/s")

        self.sweep_v_stop = QtWidgets.QDoubleSpinBox()
        self.sweep_v_stop.setRange(
            0.001,
            self.workspace_limits.maximum_line_velocity_mm_s,
        )
        self.sweep_v_stop.setDecimals(3)
        self.sweep_v_stop.setValue(2.1)
        self.sweep_v_stop.setSuffix(" mm/s")

        self.sweep_v_step = QtWidgets.QDoubleSpinBox()
        self.sweep_v_step.setRange(
            -self.workspace_limits.maximum_line_velocity_mm_s,
            self.workspace_limits.maximum_line_velocity_mm_s,
        )
        self.sweep_v_step.setDecimals(3)
        self.sweep_v_step.setValue(0.1)
        self.sweep_v_step.setSuffix(" mm/s")

        self.sweep_return_velocity = QtWidgets.QDoubleSpinBox()
        self.sweep_return_velocity.setRange(
            0.001,
            self.workspace_limits.maximum_line_velocity_mm_s,
        )
        self.sweep_return_velocity.setDecimals(3)
        self.sweep_return_velocity.setValue(1.0)
        self.sweep_return_velocity.setSuffix(" mm/s")

        self.sweep_write_blocks = QtWidgets.QCheckBox(
            "Use MOVE WHILE WRITE blocks (half the modules)"
        )
        self.sweep_write_blocks.setChecked(True)
        self.sweep_write_blocks.setToolTip(
            "Emit one combined write block per line instead of separate "
            "OPEN / LINE / CLOSED modules"
        )

        self.sweep_use_attenuation = QtWidgets.QCheckBox(
            "Sweep attenuator transmission"
        )
        self.sweep_att_start = QtWidgets.QDoubleSpinBox()
        self.sweep_att_start.setRange(0.0, 100.0)
        self.sweep_att_start.setValue(25.0)
        self.sweep_att_start.setSuffix(" %")
        self.sweep_att_stop = QtWidgets.QDoubleSpinBox()
        self.sweep_att_stop.setRange(0.0, 100.0)
        self.sweep_att_stop.setValue(25.0)
        self.sweep_att_stop.setSuffix(" %")
        self.sweep_att_step = QtWidgets.QDoubleSpinBox()
        self.sweep_att_step.setRange(-100.0, 100.0)
        self.sweep_att_step.setValue(5.0)
        self.sweep_att_step.setSuffix(" %")

        sweep_layout.addRow("Write dX", self.sweep_write_dx)
        sweep_layout.addRow("Row pitch", self.sweep_row_pitch)
        sweep_layout.addRow("Series spacing", self.sweep_series_spacing)
        sweep_layout.addRow("Velocity start", self.sweep_v_start)
        sweep_layout.addRow("Velocity stop", self.sweep_v_stop)
        sweep_layout.addRow("Velocity step", self.sweep_v_step)
        sweep_layout.addRow("Return velocity", self.sweep_return_velocity)
        sweep_layout.addRow(self.sweep_write_blocks)
        sweep_layout.addRow(self.sweep_use_attenuation)
        sweep_layout.addRow("Atten start", self.sweep_att_start)
        sweep_layout.addRow("Atten stop", self.sweep_att_stop)
        sweep_layout.addRow("Atten step", self.sweep_att_step)

        self.sweep_summary = QtWidgets.QLabel(
            "Legacy-compatible raster generator: Pockels OPEN only for "
            "the writing line; return moves are CLOSED."
        )
        self.sweep_summary.setObjectName("muted")
        self.sweep_summary.setWordWrap(True)
        sweep_layout.addRow(self.sweep_summary)

        build_sweep = QtWidgets.QPushButton(
            "GENERATE SWEEP RECIPE"
        )
        build_sweep.setObjectName("primary")
        build_sweep.clicked.connect(self._generate_sweep_recipe)
        sweep_layout.addRow(build_sweep)
        layout.addWidget(sweep_box)

        tip = QtWidgets.QLabel(
            "Build the recipe here; ordinary stage/laser control remains on "
            "the Control tab. Drag steps in the sequence to reorder them."
        )
        tip.setObjectName("muted")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        layout.addStretch(1)
        return panel

    def _build_script_sequence(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(self._section("SEQUENCE"))
        self.recipe_name = QtWidgets.QLineEdit("Untitled recipe")
        self.recipe_name.setPlaceholderText("Recipe name")
        header.addWidget(self.recipe_name, 1)
        layout.addLayout(header)

        self.sequence_hint = QtWidgets.QLabel(
            "DRAG MODULES HERE  •  the list scrolls while you drag past an "
            "edge  •  double-click a block to edit  •  Alt+↑/↓ reorders"
        )
        self.sequence_hint.setObjectName("dropHint")
        self.sequence_hint.setWordWrap(True)
        self.sequence_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sequence_hint)

        self.recipe_list = RecipeList()
        self.recipe_list.setMinimumHeight(180)
        self.recipe_list.orderChanged.connect(
            self._update_recipe_preflight_view
        )
        self.recipe_list.moduleDropped.connect(
            self._recipe_module_dropped
        )
        self.recipe_list.editRequested.connect(
            self._recipe_edit_selected
        )
        self.recipe_list.duplicateRequested.connect(
            self._recipe_duplicate
        )
        self.recipe_list.deleteRequested.connect(
            self._recipe_remove
        )
        self.recipe_list.itemDoubleClicked.connect(
            lambda _item: self._recipe_edit_selected()
        )
        self.recipe_list.moveRequested.connect(self._recipe_move_selected)
        self.recipe_list.currentRowChanged.connect(
            lambda _row: self._update_sequence_controls()
        )
        layout.addWidget(self.recipe_list, 1)

        edit_row = QtWidgets.QHBoxLayout()
        edit_row.setSpacing(5)
        up = QtWidgets.QPushButton("↑")
        up.setObjectName("chipButton")
        up.setMaximumWidth(36)
        up.setToolTip("Move the selected block up (Alt+Up)")
        up.clicked.connect(lambda: self._recipe_move_selected(-1))
        down = QtWidgets.QPushButton("↓")
        down.setObjectName("chipButton")
        down.setMaximumWidth(36)
        down.setToolTip("Move the selected block down (Alt+Down)")
        down.clicked.connect(lambda: self._recipe_move_selected(1))
        edit = QtWidgets.QPushButton("Edit")
        edit.setObjectName("primary")
        edit.setToolTip("Edit the selected block (Enter)")
        edit.clicked.connect(self._recipe_edit_selected)
        remove = QtWidgets.QPushButton("Remove")
        remove.setToolTip("Remove the selected block (Delete)")
        remove.clicked.connect(self._recipe_remove)
        duplicate = QtWidgets.QPushButton("Duplicate")
        duplicate.setToolTip("Duplicate the selected block (Ctrl+D)")
        duplicate.clicked.connect(self._recipe_duplicate)
        clear = QtWidgets.QPushButton("Clear recipe")
        clear.clicked.connect(self._recipe_clear)
        for widget in (up, down, edit, remove, duplicate, clear):
            edit_row.addWidget(widget)
        edit_row.addStretch(1)
        self.sequence_buttons = (up, down, edit, remove, duplicate)
        layout.addLayout(edit_row)

        file_row = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save recipe…")
        save.clicked.connect(self._save_recipe)
        load = QtWidgets.QPushButton("Load recipe…")
        load.clicked.connect(self._load_recipe)
        file_row.addWidget(save)
        file_row.addWidget(load)
        file_row.addStretch(1)
        layout.addLayout(file_row)
        return panel

    def _build_script_run_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        summary_box = QtWidgets.QGroupBox("EXECUTION TARGET")
        summary_layout = QtWidgets.QVBoxLayout(summary_box)
        self.execution_summary = QtWidgets.QLabel("")
        self.execution_summary.setWordWrap(True)
        self.execution_summary.setObjectName("statusPill")
        summary_layout.addWidget(self.execution_summary)
        layout.addWidget(summary_box)

        preflight_box = QtWidgets.QGroupBox("PREFLIGHT")
        preflight_layout = QtWidgets.QVBoxLayout(preflight_box)
        self.preflight_text = QtWidgets.QPlainTextEdit()
        self.preflight_text.setReadOnly(True)
        self.preflight_text.setMinimumHeight(220)
        preflight_layout.addWidget(self.preflight_text)
        refresh = QtWidgets.QPushButton("REFRESH PREFLIGHT")
        refresh.clicked.connect(self._update_recipe_preflight_view)
        preflight_layout.addWidget(refresh)
        layout.addWidget(preflight_box, 1)

        self.real_script_arm = QtWidgets.QCheckBox(
            "ARM REAL SCRIPT RUN"
        )
        self.real_script_arm.setObjectName("dangerCheck")
        self.real_script_arm.setToolTip(
            "Required if any step will command real hardware."
        )
        layout.addWidget(self.real_script_arm)

        self.run_recipe_btn = QtWidgets.QPushButton(
            "▶ RUN SCRIPT ON CURRENT PROVIDERS"
        )
        self.run_recipe_btn.setObjectName("primary")
        self.run_recipe_btn.setMinimumHeight(44)
        self.run_recipe_btn.clicked.connect(self._run_recipe)
        layout.addWidget(self.run_recipe_btn)

        self.stop_recipe_btn = QtWidgets.QPushButton(
            "■ STOP SCRIPT + CLOSE BEAM"
        )
        self.stop_recipe_btn.setObjectName("danger")
        self.stop_recipe_btn.clicked.connect(self._stop_recipe)
        layout.addWidget(self.stop_recipe_btn)

        self.script_progress_bar = QtWidgets.QProgressBar()
        self.script_progress_bar.setRange(0, 100)
        self.script_progress_bar.setValue(0)
        self.script_progress_bar.setFormat("idle")
        layout.addWidget(self.script_progress_bar)

        self.recipe_progress = QtWidgets.QLabel("Ready")
        self.recipe_progress.setObjectName("statusPill")
        self.recipe_progress.setWordWrap(True)
        layout.addWidget(self.recipe_progress)

        note = QtWidgets.QLabel(
            "A script always starts by requesting the Pockels cell CLOSED. "
            "Real-hardware execution is blocked unless all required providers "
            "are connected and the real-script arm box is checked."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        return panel

    def _build_commissioning_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        evidence_box = QtWidgets.QGroupBox(
            "RECOVERED LEGACY HARDWARE PROFILE"
        )
        evidence = QtWidgets.QVBoxLayout(evidence_box)
        self.commission_profile = QtWidgets.QComboBox()
        for candidate in self.legacy_profile.pockels_candidates:
            self.commission_profile.addItem(
                candidate.label,
                candidate.key,
            )
        self.commission_profile.currentIndexChanged.connect(
            self._commission_candidate_changed
        )
        evidence.addWidget(self.commission_profile)

        self.commission_evidence_text = QtWidgets.QLabel("")
        self.commission_evidence_text.setWordWrap(True)
        self.commission_evidence_text.setObjectName("statusPill")
        evidence.addWidget(self.commission_evidence_text)

        use_candidate = QtWidgets.QPushButton(
            "LOAD CANDIDATE INTO SETUP (DOES NOT ARM)"
        )
        use_candidate.clicked.connect(
            self._apply_commission_candidate_to_setup
        )
        evidence.addWidget(use_candidate)
        left_layout.addWidget(evidence_box)

        mock_box = QtWidgets.QGroupBox("MOCK LAB TESTING")
        mock_layout = QtWidgets.QFormLayout(mock_box)
        self.mock_scenario = QtWidgets.QComboBox()
        self.mock_scenario.addItems(
            [
                "Nominal — all virtual devices healthy",
                "Stage disconnected",
                "Pockels provider disconnected",
                "Attenuator unavailable",
            ]
        )
        self.mock_scenario.currentIndexChanged.connect(
            self._mock_scenario_changed
        )
        mock_layout.addRow("Scenario", self.mock_scenario)
        mock_note = QtWidgets.QLabel(
            "Mock mode uses the same control paths, line trajectories, "
            "Pockels state model, CAD animation and recipe engine as real mode. "
            "These fault presets let the GUI's failure handling be tested "
            "without touching hardware."
        )
        mock_note.setWordWrap(True)
        mock_note.setObjectName("muted")
        mock_layout.addRow(mock_note)
        left_layout.addWidget(mock_box)

        verify_box = QtWidgets.QGroupBox(
            "REAL POCKELS COMMISSIONING CHECKLIST"
        )
        verify_layout = QtWidgets.QVBoxLayout(verify_box)
        self.commission_checks: list[QtWidgets.QCheckBox] = []
        for label in (
            "Physical HXP output wire traced to the current LX13 interface",
            "OPEN/CLOSED polarity confirmed on the present wiring",
            "Electrical level / compatibility checked",
            "Beam safely intercepted for commissioning",
            "Physical interlock and hardware E-stop are active",
        ):
            cb = QtWidgets.QCheckBox(label)
            self.commission_checks.append(cb)
            verify_layout.addWidget(cb)

        verify_note = QtWidgets.QLabel(
            "The software evidence can pre-fill a historical candidate, but "
            "this checklist is deliberately about the present physical wiring."
        )
        verify_note.setWordWrap(True)
        verify_note.setObjectName("muted")
        verify_layout.addWidget(verify_note)

        mark_verified = QtWidgets.QPushButton(
            "MARK CURRENT MAPPING VERIFIED"
        )
        mark_verified.setObjectName("danger")
        mark_verified.clicked.connect(
            self._mark_current_mapping_verified
        )
        verify_layout.addWidget(mark_verified)

        clear_verified = QtWidgets.QPushButton(
            "CLEAR HARDWARE VERIFICATION"
        )
        clear_verified.clicked.connect(
            self._clear_hardware_verification
        )
        verify_layout.addWidget(clear_verified)
        left_layout.addWidget(verify_box)

        analog_box = QtWidgets.QGroupBox(
            "LEGACY ANALOGUE POWER / ATTENUATION PATH"
        )
        analog_layout = QtWidgets.QFormLayout(analog_box)
        self.raw_analog_gpio = QtWidgets.QLineEdit(
            self.legacy_profile.analog_monitor_gpio
        )
        self.raw_analog_value = QtWidgets.QDoubleSpinBox()
        self.raw_analog_value.setRange(0.0, 10.0)
        self.raw_analog_value.setDecimals(3)
        self.raw_analog_value.setValue(1.0)
        self.raw_analog_value.setToolTip(
            "Current lab mapping: GPIO2.DAC1 is inverted: "
            "raw 0 = 100 % transmission and raw 10 = 0 %."
        )
        self.raw_analog_arm = QtWidgets.QCheckBox(
            "I confirm this GPIO is the present analogue power/attenuation path"
        )
        analog_layout.addRow("HXP analogue GPIO", self.raw_analog_gpio)
        analog_layout.addRow("Raw legacy value", self.raw_analog_value)
        analog_layout.addRow(self.raw_analog_arm)
        raw_buttons = QtWidgets.QHBoxLayout()
        read_raw = QtWidgets.QPushButton("READ RAW")
        read_raw.clicked.connect(self._read_raw_analog)
        write_raw = QtWidgets.QPushButton("WRITE RAW DAC 0–10")
        write_raw.setObjectName("danger")
        write_raw.clicked.connect(self._write_raw_analog)
        raw_buttons.addWidget(read_raw)
        raw_buttons.addWidget(write_raw)
        analog_layout.addRow(raw_buttons)
        raw_note = QtWidgets.QLabel(
            "Raw commissioning access mirrors the same GPIO2.DAC1 path used by "
            "the normal REAL LAB attenuator. Current mapping: 0 = 100 %, 10 = 0 % transmission."
        )
        raw_note.setWordWrap(True)
        raw_note.setObjectName("muted")
        analog_layout.addRow(raw_note)
        left_layout.addWidget(analog_box)
        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        selftest_box = QtWidgets.QGroupBox("READ-ONLY SELF TEST")
        selftest_layout = QtWidgets.QVBoxLayout(selftest_box)
        selftest_note = QtWidgets.QLabel(
            "This test performs no motion and no GPIO writes. In REAL LAB it "
            "reads firmware, pose, GPIO1.DO, GPIO3.DO, GPIO4.DO and "
            "GPIO2.DAC1 from the connected HXP."
        )
        selftest_note.setWordWrap(True)
        selftest_note.setObjectName("muted")
        selftest_layout.addWidget(selftest_note)
        run_selftest = QtWidgets.QPushButton(
            "RUN READ-ONLY HARDWARE SELF TEST"
        )
        run_selftest.setObjectName("primary")
        run_selftest.clicked.connect(
            self._run_read_only_self_test
        )
        selftest_layout.addWidget(run_selftest)

        self.commission_log = QtWidgets.QPlainTextEdit()
        self.commission_log.setReadOnly(True)
        self.commission_log.setMinimumHeight(420)
        selftest_layout.addWidget(self.commission_log, 1)
        right_layout.addWidget(selftest_box, 1)

        recovered = QtWidgets.QGroupBox("WHAT THE OLD FILES ESTABLISH")
        recovered_layout = QtWidgets.QVBoxLayout(recovered)
        recovered_text = QtWidgets.QLabel(
            "HXP 192.168.0.254:5001 • 10 s timeout • HEXAPOD • Work frame\n"
            "LabVIEW v3 Pockels candidate: GPIO3.DO, mask 1, states 0/1 "
            "(polarity unresolved)\n"
            "TCL writing map: GPIO4.DO, mask 1, write=1, non-write=0\n"
            "Gate/writing marker: GPIO1.DO, mask 4, states 0/4; "
            "MotionStart/MotionEnd toggle\n"
            "Power/attenuator: GPIO2.DAC1 • CURRENT LAB 0 = 100 %, 10 = 0 % transmission; "
            "inverted linear DAC mapping\n"
            "Controller backup: RRPS geometry, Work Z=209 mm, Base/Tool Z=25 mm; "
            "six LTA-HX actuator limits loaded as reference\n"
            "STEP CAD is VISUAL ONLY in REAL LAB; HXP native Line-limit preflight "
            "is authoritative for writing trajectories"
        )
        recovered_text.setWordWrap(True)
        recovered_text.setObjectName("statusPill")
        recovered_layout.addWidget(recovered_text)
        right_layout.addWidget(recovered)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._scroll_wrap(left))
        split.addWidget(right)
        split.setSizes([610, 900])
        layout.addWidget(split)

        self._commission_candidate_changed(0)
        return tab

    def _build_setup_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)

        hxp_box = QtWidgets.QGroupBox("NEWPORT HXP")
        hxp = QtWidgets.QFormLayout(hxp_box)
        self.hxp_host = QtWidgets.QLineEdit("192.168.0.254")
        self.hxp_port = QtWidgets.QSpinBox()
        self.hxp_port.setRange(1, 65535)
        self.hxp_port.setValue(5001)
        self.hxp_timeout = QtWidgets.QDoubleSpinBox()
        self.hxp_timeout.setRange(0.1, 60.0)
        self.hxp_timeout.setDecimals(1)
        self.hxp_timeout.setValue(10.0)
        self.hxp_timeout.setSuffix(" s")
        self.hxp_group = QtWidgets.QLineEdit("HEXAPOD")
        self.hxp_coords = QtWidgets.QComboBox()
        self.hxp_coords.addItems(["Work", "Tool"])
        hxp.addRow("IP address", self.hxp_host)
        hxp.addRow("Port", self.hxp_port)
        hxp.addRow("Timeout", self.hxp_timeout)
        hxp.addRow("Group", self.hxp_group)
        hxp.addRow("Move frame", self.hxp_coords)
        hxp_buttons = QtWidgets.QHBoxLayout()
        cb = QtWidgets.QPushButton("Connect HXP")
        cb.clicked.connect(self._connect_real_hxp)
        db = QtWidgets.QPushButton("Disconnect")
        db.clicked.connect(self._disconnect_real_hxp)
        hxp_buttons.addWidget(cb)
        hxp_buttons.addWidget(db)
        hxp.addRow(hxp_buttons)
        left_layout.addWidget(hxp_box)

        limits_box = QtWidgets.QGroupBox("WORKSPACE / CONTROLLER LIMITS")
        limits_layout = QtWidgets.QVBoxLayout(limits_box)
        self.workspace_limit_status = QtWidgets.QLabel()
        self.workspace_limit_status.setWordWrap(True)
        self.workspace_limit_status.setObjectName("statusPill")
        limits_layout.addWidget(self.workspace_limit_status)

        axis_grid = QtWidgets.QGridLayout()
        axis_grid.addWidget(QtWidgets.QLabel("Axis"), 0, 0)
        axis_grid.addWidget(QtWidgets.QLabel("Minimum"), 0, 1)
        axis_grid.addWidget(QtWidgets.QLabel("Maximum"), 0, 2)
        self.limit_min_boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self.limit_max_boxes: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for row, axis in enumerate(AXES, start=1):
            minimum = QtWidgets.QDoubleSpinBox()
            maximum = QtWidgets.QDoubleSpinBox()
            for box in (minimum, maximum):
                box.setRange(-100000.0, 100000.0)
                box.setDecimals(4)
                box.setSuffix(" mm" if axis in "XYZ" else " °")
            self.limit_min_boxes[axis] = minimum
            self.limit_max_boxes[axis] = maximum
            axis_grid.addWidget(QtWidgets.QLabel(axis), row, 0)
            axis_grid.addWidget(minimum, row, 1)
            axis_grid.addWidget(maximum, row, 2)
        limits_layout.addLayout(axis_grid)

        strut_grid = QtWidgets.QGridLayout()
        strut_grid.addWidget(QtWidgets.QLabel("Mock CAD strut"), 0, 0)
        strut_grid.addWidget(QtWidgets.QLabel("Min length"), 0, 1)
        strut_grid.addWidget(QtWidgets.QLabel("Max length"), 0, 2)
        self.limit_leg_min_boxes: list[QtWidgets.QDoubleSpinBox] = []
        self.limit_leg_max_boxes: list[QtWidgets.QDoubleSpinBox] = []
        for index in range(6):
            minimum = QtWidgets.QDoubleSpinBox()
            maximum = QtWidgets.QDoubleSpinBox()
            for box in (minimum, maximum):
                box.setRange(0.0, 10000.0)
                box.setDecimals(3)
                box.setSuffix(" mm")
            self.limit_leg_min_boxes.append(minimum)
            self.limit_leg_max_boxes.append(maximum)
            strut_grid.addWidget(QtWidgets.QLabel(str(index + 1)), index + 1, 0)
            strut_grid.addWidget(minimum, index + 1, 1)
            strut_grid.addWidget(maximum, index + 1, 2)
        limits_layout.addLayout(strut_grid)

        limit_form = QtWidgets.QFormLayout()
        self.limit_tilt = QtWidgets.QDoubleSpinBox()
        self.limit_tilt.setRange(0.001, 90.0)
        self.limit_tilt.setDecimals(3)
        self.limit_tilt.setSuffix(" °")
        self.limit_velocity = QtWidgets.QDoubleSpinBox()
        self.limit_velocity.setRange(0.001, 1000.0)
        self.limit_velocity.setDecimals(3)
        self.limit_velocity.setSuffix(" mm/s")
        self.limit_samples = QtWidgets.QSpinBox()
        self.limit_samples.setRange(2, 501)
        limit_form.addRow("Max strut direction change", self.limit_tilt)
        limit_form.addRow("Max line velocity", self.limit_velocity)
        limit_form.addRow("Path samples", self.limit_samples)
        limits_layout.addLayout(limit_form)

        self.limits_verified = QtWidgets.QCheckBox(
            "I verified the complete envelope against the controller/manual"
        )
        limits_layout.addWidget(self.limits_verified)
        limit_buttons = QtWidgets.QHBoxLayout()
        apply_limits = QtWidgets.QPushButton("Apply limits")
        apply_limits.clicked.connect(self._apply_workspace_limits_from_ui)
        read_limits = QtWidgets.QPushButton("Read actuator limits + load Cartesian reference")
        read_limits.clicked.connect(self._read_workspace_limits_from_hxp)
        limit_buttons.addWidget(apply_limits)
        limit_buttons.addWidget(read_limits)
        limits_layout.addLayout(limit_buttons)
        warning = QtWidgets.QLabel(
            "The HXP exposes live user-travel limits for the six physical "
            "actuators (HEXAPOD.1…6), not for the virtual Cartesian channels "
            "HEXAPOD.X…W. REAL translation Lines/jogs therefore use the HXP's "
            "native HexapodMoveIncrementalControlLimitGet preflight. The Cartesian "
            "boxes below are a nominal HXP100-family reference only and are NOT "
            "automatically marked as a verified coupled workspace."
        )
        warning.setWordWrap(True)
        warning.setObjectName("muted")
        limits_layout.addWidget(warning)
        left_layout.addWidget(limits_box)

        pockels_box = QtWidgets.QGroupBox("POCKELS CELL / PHAROS LX13")
        pockels = QtWidgets.QFormLayout(pockels_box)
        self.laser_mode = QtWidgets.QComboBox()
        self.laser_mode.addItems(
            ["Virtual Pockels cell", "Real HXP GPIO → PHAROS LX13"]
        )
        self.laser_mode.currentIndexChanged.connect(
            self._laser_mode_changed
        )
        self.gpio_name = QtWidgets.QLineEdit("")
        self.gpio_mask = QtWidgets.QSpinBox()
        self.gpio_mask.setRange(0, 65535)
        self.gpio_open = QtWidgets.QSpinBox()
        self.gpio_open.setRange(0, 65535)
        self.gpio_closed = QtWidgets.QSpinBox()
        self.gpio_closed.setRange(0, 65535)
        pockels.addRow("Provider", self.laser_mode)
        pockels.addRow("HXP GPIO name", self.gpio_name)
        pockels.addRow("Mask", self.gpio_mask)
        pockels.addRow("OPEN value", self.gpio_open)
        pockels.addRow("CLOSED value", self.gpio_closed)
        self.wiring_verified = QtWidgets.QCheckBox(
            "I verified LX13 pinout, active level and HXP electrical compatibility"
        )
        pockels.addRow(self.wiring_verified)
        pockels_buttons = QtWidgets.QHBoxLayout()
        arm = QtWidgets.QPushButton("Connect / arm provider")
        arm.clicked.connect(self._connect_laser)
        close = QtWidgets.QPushButton("Request CLOSED")
        close.setObjectName("safeAction")
        close.clicked.connect(self._close_all_pockels)
        pockels_buttons.addWidget(arm)
        pockels_buttons.addWidget(close)
        pockels.addRow(pockels_buttons)
        left_layout.addWidget(pockels_box)

        att_box = QtWidgets.QGroupBox("ATTENUATOR HARDWARE")
        att = QtWidgets.QFormLayout(att_box)
        self.attenuator_mode = QtWidgets.QComboBox()
        self.attenuator_mode.addItems(
            [
                "Virtual attenuator",
                "Real HXP GPIO2.DAC1 attenuator",
            ]
        )
        self.attenuator_mode.currentIndexChanged.connect(
            self._attenuator_mode_changed
        )
        self.attenuator_model = QtWidgets.QLineEdit("")
        self.attenuator_model.setPlaceholderText(
            "Enter make/model/controller when known"
        )
        self.attenuator_driver_status = QtWidgets.QLabel(
            "Current lab mapping: HXP GPIO2.DAC1, raw 0–10 is inverted. "
            "Operator-confirmed endpoints: 0 = 100 % transmission; 10 = 0 % transmission."
        )
        self.attenuator_driver_status.setWordWrap(True)
        self.attenuator_driver_status.setObjectName("muted")
        att.addRow("Provider", self.attenuator_mode)
        att.addRow("Hardware", self.attenuator_model)
        att.addRow(self.attenuator_driver_status)
        connect_att = QtWidgets.QPushButton("Connect attenuator")
        connect_att.clicked.connect(self._connect_attenuator)
        att.addRow(connect_att)
        left_layout.addWidget(att_box)

        config_box = QtWidgets.QGroupBox("CONFIGURATION")
        config_layout = QtWidgets.QHBoxLayout(config_box)
        load_cfg = QtWidgets.QPushButton("Load config…")
        load_cfg.clicked.connect(self._load_hardware_config_dialog)
        save_cfg = QtWidgets.QPushButton("Save config…")
        save_cfg.clicked.connect(self._save_hardware_config_dialog)
        config_layout.addWidget(load_cfg)
        config_layout.addWidget(save_cfg)
        left_layout.addWidget(config_box)
        left_layout.addStretch(1)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        cad_box = QtWidgets.QGroupBox("CAD / DIGITAL TWIN")
        cad_layout = QtWidgets.QVBoxLayout(cad_box)
        self.setup_cad_status = QtWidgets.QLabel(
            "Using CAD-derived rig profile. Load the supplied STEP for exact surfaces."
        )
        self.setup_cad_status.setWordWrap(True)
        self.setup_cad_status.setObjectName("muted")
        cad_layout.addWidget(self.setup_cad_status)
        load_step = QtWidgets.QPushButton(
            "Load Stewart Platform STEP…"
        )
        load_step.clicked.connect(self._load_step_dialog)
        cad_layout.addWidget(load_step)
        right_layout.addWidget(cad_box)

        diagnostics = QtWidgets.QGroupBox("DIAGNOSTICS")
        diag_layout = QtWidgets.QVBoxLayout(diagnostics)
        diag_buttons = QtWidgets.QHBoxLayout()
        firmware = QtWidgets.QPushButton("Read HXP firmware")
        firmware.clicked.connect(self._diagnostic_firmware)
        poll = QtWidgets.QPushButton("Poll HXP pose")
        poll.clicked.connect(self._diagnostic_pose)
        clear_log = QtWidgets.QPushButton("Clear log")
        clear_log.clicked.connect(lambda: self.diag_log.clear())
        diag_buttons.addWidget(firmware)
        diag_buttons.addWidget(poll)
        diag_buttons.addWidget(clear_log)
        diag_layout.addLayout(diag_buttons)
        self.diag_log = QtWidgets.QPlainTextEdit()
        self.diag_log.setReadOnly(True)
        diag_layout.addWidget(self.diag_log, 1)
        right_layout.addWidget(diagnostics, 1)

        safety = QtWidgets.QGroupBox("SAFETY BOUNDARY")
        safety_layout = QtWidgets.QVBoxLayout(safety)
        safety_text = QtWidgets.QLabel(
            "This GUI can request motion and process-beam state, but it is not "
            "a safety PLC, laser interlock, or hardware emergency stop. "
            "The physical PHAROS interlock/shutter chain and the hardware E-stop "
            "remain authoritative. Real LX13 control is intentionally impossible "
            "until its wiring values are explicitly verified."
        )
        safety_text.setWordWrap(True)
        safety_text.setObjectName("muted")
        safety_layout.addWidget(safety_text)
        right_layout.addWidget(safety)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self._scroll_wrap(left))
        split.addWidget(right)
        split.setSizes([650, 850])
        layout.addWidget(split)
        return tab

    def _scroll_wrap(self, widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        widget.setMinimumWidth(0)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        area.setWidget(widget)
        return area

    def _install_wheel_guards(self) -> None:
        """Keep page scrolling from changing laboratory setpoints.

        Spin boxes, sliders and closed combo boxes are intentionally adjusted
        by keyboard/click/drag only. A wheel event over one of them scrolls its
        enclosing panel instead of silently changing a value.
        """

        guarded = [
            *self.findChildren(QtWidgets.QAbstractSpinBox),
            *self.findChildren(QtWidgets.QComboBox),
            *self.findChildren(QtWidgets.QSlider),
        ]
        for widget in guarded:
            widget.installEventFilter(self)

    def eventFilter(
        self,
        watched: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        if (
            event.type() == QtCore.QEvent.Type.Wheel
            and isinstance(
                watched,
                (
                    QtWidgets.QAbstractSpinBox,
                    QtWidgets.QComboBox,
                    QtWidgets.QSlider,
                ),
            )
        ):
            parent = (
                watched.parentWidget()
                if isinstance(watched, QtWidgets.QWidget)
                else None
            )
            while parent is not None:
                if isinstance(parent, QtWidgets.QAbstractScrollArea):
                    delta = event.pixelDelta()
                    if delta.isNull():
                        delta = event.angleDelta()
                    if abs(delta.y()) >= abs(delta.x()):
                        bar = parent.verticalScrollBar()
                        bar.setValue(bar.value() - delta.y())
                    else:
                        bar = parent.horizontalScrollBar()
                        bar.setValue(bar.value() - delta.x())
                    event.accept()
                    return True
                parent = parent.parentWidget()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def _section(self, text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("section")
        return label

    def _pose_spinbox(self, index: int) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        # Two XYZ/UVW columns must fit in the narrow control/script palettes at
        # 1250 logical pixels and common Windows 150% display scaling.
        box.setMinimumWidth(118)
        box.setMaximumWidth(160)
        box.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        box.setDecimals(4)
        axis = AXES[index]
        minimum, maximum = self.workspace_limits.axis_bounds(axis)
        if index < 3:
            box.setRange(minimum, maximum)
            box.setSingleStep(0.1)
            box.setSuffix(" mm")
        else:
            box.setRange(minimum, maximum)
            box.setSingleStep(0.1)
            box.setSuffix(" °")
        return box

    def _validate_motion_path(self, start: Pose6D, target: Pose6D) -> None:
        """Shared coupled-envelope check used by the virtual stage."""
        self.workspace_limits.require_path(self.kinematics, start, target)
        self.sample_calibration.require_path(start, target)

    def _planned_stage_pose(self) -> Pose6D:
        snap = self._last_stage_snapshot
        if snap.state == MotionState.MOVING and snap.target is not None:
            return snap.target
        return snap.actual

    def _require_requested_motion(
        self,
        target: Pose6D,
        *,
        velocity_mm_s: float | None = None,
    ) -> None:
        if not all(
            isinstance(v, (int, float)) and abs(float(v)) < float("inf")
            for v in target.as_tuple()
        ):
            raise ValueError("requested pose must contain six finite values")

        start = self._planned_stage_pose()

        if self.stage_mode.currentIndex() == 1:
            if self.real_stage is None or not self.real_stage.client.connected:
                raise RuntimeError("Real HXP is not connected")
            if not self._real_frames_match_profile:
                raise RuntimeError(
                    "Real motion is blocked because the live HXP Work/Tool "
                    "frames do not match the commissioned profile"
                )
            if not self._real_readback_is_fresh():
                raise RuntimeError(
                    "Real motion is blocked because the HXP pose/status "
                    "readback is stale or not ready"
                )
            self.real_stage.client.require_ready_for_motion(
                self.real_stage.config.group
            )

            if velocity_mm_s is not None:
                # Translation-only processing lines have an authoritative HXP
                # coupled-kinematics preflight. Never use the decorative STEP
                # strut geometry as a safety decision for the real machine.
                dx = target.x - start.x
                dy = target.y - start.y
                dz = target.z - start.z
                max_velocity, trajectory_percent = (
                    self.real_stage.client.line_incremental_control_limits(
                        dx,
                        dy,
                        dz,
                        group=self.real_stage.config.group,
                        coordinate_system=self.real_stage.config.coordinate_system,
                    )
                )
                # HXP reports the executable trajectory as a FRACTION
                # in [0, 1], not a percentage in [0, 100]. Newport's own
                # examples use 1.0 for a fully executable trajectory.
                if trajectory_percent < 0.999999:
                    raise RuntimeError(
                        "HXP rejected the complete Line trajectory: "
                        f"only {trajectory_percent * 100.0:.3f}% is executable"
                    )
                if float(velocity_mm_s) > max_velocity + 1e-9:
                    raise RuntimeError(
                        f"requested {float(velocity_mm_s):.3f} mm/s exceeds "
                        f"the HXP carriage limit {max_velocity:.3f} mm/s for this Line"
                    )
            else:
                # Absolute/relative XYZUVW moves still require a deliberately
                # operator-verified Cartesian envelope. Only Cartesian bounds are
                # checked here; CAD strut lengths are visual-only.
                if not self.workspace_limits.controller_verified:
                    raise RuntimeError(
                        "Real general motion is blocked until the Cartesian "
                        "workspace envelope is explicitly verified in Setup + Diagnostics."
                    )
                self.workspace_limits.require_cartesian_pose(target)
        else:
            if velocity_mm_s is not None:
                self.workspace_limits.require_line_velocity(velocity_mm_s)
            self.workspace_limits.require_path(self.kinematics, start, target)

        # The measured sample extent is a second, independent bound in both modes.
        self.sample_calibration.require_path(start, target)

    @staticmethod
    def _set_spin_range_preserving_value(
        box: QtWidgets.QDoubleSpinBox,
        minimum: float,
        maximum: float,
    ) -> None:
        value = box.value()
        box.setRange(float(minimum), float(maximum))
        box.setValue(max(float(minimum), min(float(maximum), value)))

    def _apply_workspace_limits_to_widgets(self) -> None:
        limits = self.workspace_limits
        for boxes_name in ("pose_boxes", "script_pose_boxes"):
            boxes = getattr(self, boxes_name, {})
            for axis, box in boxes.items():
                self._set_spin_range_preserving_value(
                    box,
                    *limits.axis_bounds(axis),
                )

        spans = {
            axis: limits.axis_bounds(axis)[1] - limits.axis_bounds(axis)[0]
            for axis in AXES
        }
        for name, span in (
            ("quick_dx", spans["X"]),
            ("quick_dy", spans["Y"]),
            ("quick_dz", spans["Z"]),
            ("quick_row_pitch", spans["Y"]),
            ("sweep_write_dx", spans["X"]),
            ("sweep_row_pitch", spans["Y"]),
        ):
            box = getattr(self, name, None)
            if box is not None:
                self._set_spin_range_preserving_value(box, -span, span)
        if hasattr(self, "sweep_series_spacing"):
            self._set_spin_range_preserving_value(
                self.sweep_series_spacing,
                0.0,
                spans["Y"],
            )
        for name in (
            "quick_velocity",
            "script_line_velocity",
            "sweep_v_start",
            "sweep_v_stop",
            "sweep_return_velocity",
        ):
            box = getattr(self, name, None)
            if box is not None:
                self._set_spin_range_preserving_value(
                    box,
                    0.001,
                    limits.maximum_line_velocity_mm_s,
                )
        if hasattr(self, "sweep_v_step"):
            vmax = limits.maximum_line_velocity_mm_s
            self._set_spin_range_preserving_value(
                self.sweep_v_step,
                -vmax,
                vmax,
            )
        if hasattr(self, "jog_mm"):
            self._set_spin_range_preserving_value(
                self.jog_mm,
                0.0001,
                max(spans[a] for a in "XYZ"),
            )
            self._set_spin_range_preserving_value(
                self.jog_deg,
                0.0001,
                max(spans[a] for a in "UVW"),
            )

        if not hasattr(self, "limit_min_boxes"):
            self._script_move_kind_changed()
            return
        for axis in AXES:
            lo, hi = limits.axis_bounds(axis)
            self.limit_min_boxes[axis].setValue(lo)
            self.limit_max_boxes[axis].setValue(hi)
        for box, value in zip(
            self.limit_leg_min_boxes,
            limits.minimum_leg_lengths_mm,
        ):
            box.setValue(value)
        for box, value in zip(
            self.limit_leg_max_boxes,
            limits.maximum_leg_lengths_mm,
        ):
            box.setValue(value)
        self.limit_tilt.setValue(limits.maximum_leg_tilt_deg)
        self.limit_velocity.setValue(limits.maximum_line_velocity_mm_s)
        self.limit_samples.setValue(limits.path_samples)
        self.limits_verified.setChecked(limits.controller_verified)
        state = "VERIFIED" if limits.controller_verified else "PROVISIONAL"
        self.workspace_limit_status.setText(
            f"{state} • {limits.source}"
        )
        self._script_move_kind_changed()

    def _script_move_kind_changed(self, _index: int | None = None) -> None:
        if not hasattr(self, "script_pose_boxes"):
            return
        kind = self.script_move_kind.currentIndex()
        for axis, box in self.script_pose_boxes.items():
            lo, hi = self.workspace_limits.axis_bounds(axis)
            if kind == 0:
                minimum, maximum = lo, hi
            else:
                span = hi - lo
                minimum, maximum = -span, span
            self._set_spin_range_preserving_value(box, minimum, maximum)
            box.setEnabled(kind < 2 or axis in "XYZ")
        if hasattr(self, "script_line_velocity"):
            self.script_line_velocity.setEnabled(kind >= 2)

    def _apply_workspace_limits_from_ui(self) -> None:
        try:
            minimum = Pose6D(
                *(self.limit_min_boxes[a].value() for a in AXES)
            )
            maximum = Pose6D(
                *(self.limit_max_boxes[a].value() for a in AXES)
            )
            limits = WorkspaceLimits.from_dict(
                {
                    "source": getattr(
                        self,
                        "_workspace_pending_source",
                        "operator-configured envelope",
                    ),
                    "controller_verified": self.limits_verified.isChecked(),
                    "axes": {
                        axis: [
                            minimum.as_tuple()[index],
                            maximum.as_tuple()[index],
                        ]
                        for index, axis in enumerate(AXES)
                    },
                    "struts": {
                        "minimum_length_mm": [
                            box.value() for box in self.limit_leg_min_boxes
                        ],
                        "maximum_length_mm": [
                            box.value() for box in self.limit_leg_max_boxes
                        ],
                        "maximum_tilt_from_home_deg": self.limit_tilt.value(),
                    },
                    "maximum_line_velocity_mm_s": self.limit_velocity.value(),
                    "path_samples": self.limit_samples.value(),
                },
                self.kinematics,
            )
            current_issues = limits.violations(
                self.kinematics,
                self._last_stage_snapshot.actual,
            )
            if current_issues:
                raise ValueError(
                    "Current pose is outside the proposed envelope: "
                    + current_issues[0].message
                )
            self.workspace_limits = limits
            self._workspace_pending_source = limits.source
            self._apply_workspace_limits_to_widgets()
            self._update_recipe_preflight_view()
            self.statusBar().showMessage("Workspace limits applied", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid workspace limits",
                str(exc),
            )

    def _read_workspace_limits_from_hxp(self) -> None:
        """Read live physical actuator limits and load a Cartesian reference.

        Newport's PositionerUserTravelLimitsGet applies to the six physical
        positioners (HEXAPOD.1…6). The virtual Cartesian channels HEXAPOD.X…W
        are valid for gathered/current Cartesian data but are not positioners
        for that API on this controller. Translation moves are therefore
        authorised with the HXP native Line control-limit preflight instead of
        pretending a rectangular Cartesian envelope came from the controller.
        """
        if self.real_stage is None or not self.real_stage.client.connected:
            QtWidgets.QMessageBox.warning(
                self,
                "HXP not connected",
                "Connect the real HXP before reading actuator limits.",
            )
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )

            actuator_lines = []
            live_actuator_limits: list[tuple[float, float]] = []
            for item in self.controller_profile.actuators:
                try:
                    live_lo, live_hi = (
                        self.real_stage.client.positioner_user_travel_limits(
                            item.positioner
                        )
                    )
                    current = self.real_stage.client.positioner_current_position(
                        item.positioner
                    )
                    live_actuator_limits.append((live_lo, live_hi))
                    actuator_lines.append(
                        f"{item.positioner}: {current:.6f} mm in "
                        f"[{live_lo:.6f}, {live_hi:.6f}] mm"
                    )
                except Exception as exc:
                    live_actuator_limits.append(
                        (
                            item.minimum_target_position_mm,
                            item.maximum_target_position_mm,
                        )
                    )
                    actuator_lines.append(
                        f"{item.positioner}: live read failed ({exc}); backup "
                        f"[{item.minimum_target_position_mm:.6f}, "
                        f"{item.maximum_target_position_mm:.6f}] mm"
                    )

            # Reference travel from Newport HXP100-family specifications. These
            # are independent nominal axis ranges, not a guaranteed rectangular
            # coupled workspace. Keep controller_verified=False.
            reference = {
                "X": (-27.5, 27.5),
                "Y": (-25.0, 25.0),
                "Z": (-14.0, 14.0),
                "U": (-11.5, 11.5),
                "V": (-10.5, 10.5),
                "W": (-19.0, 19.0),
            }
            for axis in AXES:
                self.limit_min_boxes[axis].setValue(reference[axis][0])
                self.limit_max_boxes[axis].setValue(reference[axis][1])

            self.limits_verified.setChecked(False)
            self._workspace_pending_source = (
                "LIVE HXP physical actuator limits + Newport HXP100-family "
                "nominal Cartesian reference; coupled workspace NOT verified"
            )
            self.workspace_limit_status.setText(
                "LIVE ACTUATOR LIMITS READ • Cartesian values are NOMINAL "
                "REFERENCE ONLY • translation Lines/jogs use native HXP preflight"
            )
            self._diag(
                "Read physical actuator limits from live HXP; "
                "HEXAPOD.X…W are virtual Cartesian channels and do not support "
                "PositionerUserTravelLimitsGet on this controller"
            )
            for line in actuator_lines:
                self._diag(line)

            QtWidgets.QMessageBox.information(
                self,
                "HXP limits read",
                "Live limits were read for HEXAPOD.1…6.\n\n"
                "The Cartesian boxes now show the nominal HXP100-family travel "
                "reference, but they remain deliberately UNVERIFIED because the "
                "real six-axis workspace is coupled.\n\n"
                "For a first real motion test, use an X/Y/Z jog or MOVE LINE: "
                "those are preflighted by the HXP controller itself.",
            )
            self._update_recipe_preflight_view()
        except Exception as exc:
            self.limits_verified.setChecked(False)
            QtWidgets.QMessageBox.critical(
                self,
                "Could not read HXP actuator limits",
                str(exc),
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f151c;
                color: #e8edf2;
                font-size: 13px;
            }
            QLabel#title {
                font-size: 22px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#headerLabel, QLabel#section {
                color: #93a2b0;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#muted { color: #8794a0; }
            QLabel#dropHint {
                color:#9bb1c1;
                background:#121b23;
                border:1px dashed #496276;
                border-radius:6px;
                padding:8px;
                font-weight:700;
            }
            QFrame#runBar {
                background:#141d26;
                border:1px solid #2c3a46;
                border-radius:8px;
            }
            QPushButton#chipButton {
                padding:4px 6px;
                background:#1b2630;
                border:1px solid #33414d;
                font-weight:700;
                color:#b9c6d1;
            }
            QPushButton#chipButton:hover { border-color:#6ea2c4; }
            QListWidget#sampleList {
                font-family: Consolas, 'Courier New';
                font-size: 11px;
            }
            QListWidget#sampleList::item { padding:3px 6px; }
            QProgressBar {
                background:#141d26;
                border:1px solid #2f3c48;
                border-radius:5px;
                text-align:center;
                color:#c6d3de;
                min-height:20px;
            }
            QProgressBar::chunk {
                background:#2d6b92;
                border-radius:4px;
            }
            QCheckBox { spacing:6px; }
            QToolTip {
                background:#16202a;
                color:#dce5ec;
                border:1px solid #3a4a57;
                padding:5px;
            }
            QLabel#mono {
                font-family: Consolas, 'Courier New';
                font-size: 12px;
            }
            QLabel#statusPill {
                background:#1c2630;
                border:1px solid #34414d;
                border-radius:6px;
                padding:7px;
            }
            QLabel#chipNeutral, QLabel#chipSafe, QLabel#chipLive,
            QLabel#chipWarn {
                border-radius:7px;
                padding:7px 10px;
                font-weight:700;
            }
            QLabel#chipNeutral {
                background:#1c2630;
                border:1px solid #3a4855;
                color:#d4dce3;
            }
            QLabel#chipSafe {
                background:#183427;
                border:1px solid #326c49;
                color:#b8efca;
            }
            QLabel#chipLive {
                background:#5b361a;
                border:1px solid #b16b2f;
                color:#ffe0a9;
            }
            QLabel#chipWarn {
                background:#492426;
                border:1px solid #8e4549;
                color:#ffc4c4;
            }
            QLabel#laserOff {
                background:#183427;
                border:1px solid #326c49;
                color:#b8efca;
                border-radius:7px;
                padding:10px;
                font-weight:800;
            }
            QLabel#laserOn {
                background:#5b361a;
                border:1px solid #b16b2f;
                color:#ffe0a9;
                border-radius:7px;
                padding:10px;
                font-weight:800;
            }
            QPushButton, QComboBox, QLineEdit, QSpinBox,
            QDoubleSpinBox, QPlainTextEdit {
                background:#18212a;
                border:1px solid #35424f;
                border-radius:5px;
                padding:6px;
                min-height:23px;
            }
            QPushButton:hover { border-color:#687a8a; }
            QPushButton:disabled {
                color:#5f6972;
                background:#151b21;
                border-color:#252d35;
            }
            QPushButton#primary {
                background:#214d69;
                border-color:#3f80a6;
                font-weight:800;
            }
            QPushButton#danger {
                background:#592627;
                border-color:#9a4446;
                color:#ffd0d0;
                font-weight:800;
            }
            QPushButton#safeAction {
                background:#1c402d;
                border-color:#397a55;
                color:#c8f4d7;
                font-weight:800;
            }
            QPushButton#beamOn, QPushButton#beamOnSmall {
                background:#603818;
                border:1px solid #b97131;
                color:#ffe1a9;
                font-weight:900;
            }
            QPushButton#beamOff, QPushButton#beamOffSmall {
                background:#1b432e;
                border:1px solid #3c8059;
                color:#c5f1d4;
                font-weight:900;
            }
            QGroupBox {
                border:1px solid #2e3944;
                border-radius:8px;
                margin-top:11px;
                padding-top:11px;
                font-weight:800;
                color:#a9b6c2;
            }
            QTabWidget::pane {
                border:1px solid #28333d;
                border-radius:6px;
            }
            QTabBar::tab {
                background:#151d25;
                border:1px solid #29343f;
                padding:9px 20px;
                margin-right:2px;
                font-weight:700;
                color:#8f9da9;
            }
            QTabBar::tab:selected {
                background:#22303d;
                color:#eef3f7;
                border-bottom:2px solid #4d8bb3;
            }
            QListWidget {
                background:#0c1218;
                border:1px solid #303b46;
                alternate-background-color:#121a22;
            }
            QListWidget::item {
                padding:8px;
                border-bottom:1px solid #202a33;
            }
            QListWidget::item:selected {
                background:#23445a;
                color:#f2f7fb;            }
            QCheckBox#dangerCheck {
                color:#ffb5ae;
                font-weight:800;
            }
            QSplitter::handle {
                background:#26313b;
                width:2px;
                height:2px;
            }
            QSlider::groove:horizontal {
                height:6px;
                background:#26313b;
                border-radius:3px;
            }
            QSlider::handle:horizontal {
                width:15px;
                margin:-5px 0;
                background:#73a9ce;
                border-radius:7px;
            }
            """
        )

    # ------------------------------------------------- sample calibration
    def _capture_sample_point(self) -> None:
        """Log the live pose as a sample edge/corner."""

        snapshot = self._last_stage_snapshot
        if not snapshot.connected:
            QtWidgets.QMessageBox.warning(
                self,
                "Stage not connected",
                "Connect the stage before logging a sample edge point.",
            )
            return
        if snapshot.state == MotionState.MOVING:
            QtWidgets.QMessageBox.information(
                self,
                "Stage is moving",
                "Wait until the move finishes before logging this edge point.",
            )
            return

        label = next_point_label(self.sample_calibration.point_count)
        point = SamplePoint(
            label=label,
            pose=snapshot.actual,
            beam_xy_mm=self.viewer.beam_hit_sample_xy,
        )
        self.sample_calibration = self.sample_calibration.with_point(point)
        self._refresh_sample_views()
        self.statusBar().showMessage(
            f"Logged {label} at X={point.pose.x:.4f}, Y={point.pose.y:.4f}, "
            f"Z={point.pose.z:.4f} mm",
            6000,
        )

    def _remove_sample_point(self) -> None:
        row = self.sample_points_list.currentRow()
        if row < 0:
            return
        self.sample_calibration = self.sample_calibration.without_point(row)
        self._enforce_sample_bounds_are_reachable()
        self._refresh_sample_views()

    def _clear_sample_points(self) -> None:
        if not self.sample_calibration.points:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear sample calibration?",
            "Discard every logged sample edge point and release the bounds?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.sample_calibration = self.sample_calibration.cleared()
        for widget in (self.sample_bound_xy, self.sample_bound_z):
            widget.blockSignals(True)
            widget.setChecked(False)
            widget.blockSignals(False)
        self._refresh_sample_views()

    def _apply_sample_settings(self, *_args) -> None:
        """Push the margin/bind widgets into the calibration object."""

        if not hasattr(self, "sample_margin"):
            return
        want_xy = self.sample_bound_xy.isChecked()
        want_z = self.sample_bound_z.isChecked()
        candidate = replace(
            self.sample_calibration,
            name=self.sample_name.text().strip() or "Sample",
            margin_mm=self.sample_margin.value(),
            z_above_mm=self.sample_z_above.value(),
            z_below_mm=self.sample_z_below.value(),
            enforce_xy=want_xy,
            enforce_z=want_z,
        )

        if want_xy and candidate.stage_region() is None:
            self.sample_bound_xy.setChecked(False)
            QtWidgets.QMessageBox.warning(
                self,
                "Sample not calibrated",
                "Log at least two distinct sample corners before binding "
                "motion to the sample.",
            )
            return

        # Refuse to arm bounds that the current pose already violates, which
        # would otherwise trap the stage outside its own permitted region.
        issues = candidate.violations(self._last_stage_snapshot.actual)
        if issues and (want_xy or want_z):
            for widget in (self.sample_bound_xy, self.sample_bound_z):
                widget.blockSignals(True)
                widget.setChecked(False)
                widget.blockSignals(False)
            QtWidgets.QMessageBox.warning(
                self,
                "Current pose is outside the sample bounds",
                issues[0]
                + "\n\nMove back inside the calibrated area (or reduce the "
                "margin) before binding motion.",
            )
            candidate = replace(candidate, enforce_xy=False, enforce_z=False)

        self.sample_calibration = candidate
        self._refresh_sample_views()

    def _enforce_sample_bounds_are_reachable(self) -> None:
        """Release the bind flags when the live pose can no longer satisfy them.

        Bound motion is a guard on commanded moves. If the stage ends up outside
        the calibrated area by another route - homing, an abort, the Newport
        front panel, or an edit to the captured points - holding the bound would
        block the move back in as well. The margin is ignored here so ordinary
        jitter at the keep-out line never releases the bound spuriously.
        """

        calibration = self.sample_calibration
        if not (calibration.enforce_xy or calibration.enforce_z):
            return
        measured = replace(calibration, margin_mm=0.0)
        blocked = (
            calibration.enforce_xy and calibration.stage_region() is None
        ) or bool(measured.violations(self._last_stage_snapshot.actual))
        if not blocked:
            return
        self.sample_calibration = replace(
            calibration,
            enforce_xy=False,
            enforce_z=False,
        )
        for widget in (self.sample_bound_xy, self.sample_bound_z):
            widget.blockSignals(True)
            widget.setChecked(False)
            widget.blockSignals(False)
        self.statusBar().showMessage(
            "Sample bounds released: the calibration no longer covers the "
            "current pose",
            8000,
        )
        # The map and the twin hold their own copy of the calibration.
        self._refresh_sample_views()

    def _goto_sample_centre(self) -> None:
        centre = self.sample_calibration.centre()
        if centre is None:
            QtWidgets.QMessageBox.information(
                self,
                "No sample calibration",
                "Log at least one sample edge point first.",
            )
            return
        try:
            self._require_requested_motion(centre)
            self._stage_provider().move_absolute(centre)
            for axis, value in zip("XYZUVW", centre.as_tuple()):
                self.pose_boxes[axis].setValue(value)
            self.statusBar().showMessage(
                "Moving to the centre of the calibrated sample area",
                5000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Move to sample centre failed",
                str(exc),
            )

    def _refresh_sample_views(self) -> None:
        """Re-render every surface that depends on the sample calibration."""

        calibration = self.sample_calibration
        if hasattr(self, "sample_points_list"):
            self.sample_points_list.clear()
            for index, point in enumerate(calibration.points, start=1):
                item = QtWidgets.QListWidgetItem(
                    f"{index}. {point.label}   "
                    f"X {point.pose.x:+.4f}   Y {point.pose.y:+.4f}   "
                    f"Z {point.pose.z:+.4f}"
                )
                item.setToolTip(
                    "Captured pose "
                    + "  ".join(
                        f"{name}={value:+.4f}"
                        for name, value in zip("XYZUVW", point.pose.as_tuple())
                    )
                )
                self.sample_points_list.addItem(item)
            self.sample_status.setText(calibration.summary())

        if hasattr(self, "movement_map"):
            self.movement_map.set_sample_calibration(calibration)
        if hasattr(self, "viewer"):
            self.viewer.set_sample_outline(
                calibration.sample_region()
                if self.show_sample_check.isChecked()
                else None
            )
        self._update_recipe_preflight_view()

    def _save_sample_calibration(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save sample calibration",
            f"{self.sample_calibration.name.replace(' ', '_')}_sample.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self.sample_calibration.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not save sample calibration",
                str(exc),
            )
            return
        self.statusBar().showMessage(f"Sample calibration saved to {path}", 5000)

    def _load_sample_calibration(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load sample calibration",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            calibration = SampleCalibration.from_dict(
                json.loads(Path(path).read_text(encoding="utf-8"))
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not load sample calibration",
                str(exc),
            )
            return
        # A loaded file describes geometry; it never re-arms the bounds silently.
        self._set_sample_calibration(
            replace(calibration, enforce_xy=False, enforce_z=False)
        )
        self.statusBar().showMessage(
            f"Loaded sample calibration from {path}",
            5000,
        )

    def _set_sample_calibration(self, calibration: SampleCalibration) -> None:
        self.sample_calibration = calibration
        self.sample_name.blockSignals(True)
        self.sample_name.setText(calibration.name)
        self.sample_name.blockSignals(False)
        for spin, value in (
            (self.sample_margin, calibration.margin_mm),
            (self.sample_z_above, calibration.z_above_mm),
            (self.sample_z_below, calibration.z_below_mm),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        for check, value in (
            (self.sample_bound_xy, calibration.enforce_xy),
            (self.sample_bound_z, calibration.enforce_z),
        ):
            check.blockSignals(True)
            check.setChecked(value)
            check.blockSignals(False)
        self._refresh_sample_views()

    def _fit_map_view(self) -> None:
        view = self.movement_map.suggested_view()
        if view is None:
            return
        span, cx, cy = view
        self.movement_map.set_centre_mm(cx, cy)
        self.map_span.setValue(
            max(self.map_span.minimum(), min(self.map_span.maximum(), span))
        )

    def _apply_trace_visibility(self, *_args) -> None:
        travel = self.show_travel_check.isChecked()
        written = self.show_written_check.isChecked()
        self.movement_map.set_show_travel(travel)
        self.movement_map.set_show_written(written)
        self.movement_map.set_show_sample(self.show_sample_check.isChecked())
        self.viewer.set_trace_visibility(travel=travel, written=written)
        self.viewer.set_sample_outline(
            self.sample_calibration.sample_region()
            if self.show_sample_check.isChecked()
            else None
        )

    # ---------------------------------------------------------- lab modes
    def _selected_candidate(self) -> PockelsCandidate:
        key = self.legacy_profile_combo.currentData()
        if not key:
            key = self._selected_legacy_candidate_key
        return self.legacy_profile.candidate(str(key))

    def _legacy_profile_changed(self, _index: int) -> None:
        key = self.legacy_profile_combo.currentData()
        if key:
            self._selected_legacy_candidate_key = str(key)
        if hasattr(self, "commission_profile"):
            idx = self.commission_profile.findData(
                self._selected_legacy_candidate_key
            )
            if idx >= 0 and self.commission_profile.currentIndex() != idx:
                self.commission_profile.blockSignals(True)
                self.commission_profile.setCurrentIndex(idx)
                self.commission_profile.blockSignals(False)
                self._commission_candidate_changed(idx)
        self._apply_mock_profile()

    def _apply_mock_profile(self) -> None:
        if not hasattr(self, "legacy_profile_combo"):
            return
        candidate = self._selected_candidate()

        # When polarity is unresolved, the virtual provider still has a logical
        # OPEN/CLOSED state but does not pretend a raw GPIO value is known.
        self.virtual_laser.configure_profile(
            profile_label=candidate.label,
            gpio_name=candidate.gpio_name,
            mask=candidate.mask,
            open_value=candidate.open_value,
            closed_value=candidate.closed_value,
        )
        if hasattr(self, "mode_chip") and self.lab_mode.currentIndex() == 0:
            self.mode_chip.setText(
                "MOCK • " + candidate.key.replace("_", " ").upper()
            )
        if hasattr(self, "commission_evidence_text"):
            self._commission_candidate_changed(
                self.commission_profile.currentIndex()
            )

    def _lab_mode_changed(self, index: int) -> None:
        if not hasattr(self, "laser_mode"):
            return
        self._close_all_pockels()
        self.real_script_arm.setChecked(False)
        self.manual_beam_arm.setChecked(False)

        if int(index) == 0:
            self.stage_mode.setCurrentIndex(0)
            self.laser_mode.setCurrentIndex(0)
            self.attenuator_mode.setCurrentIndex(0)
            if not self.virtual_stage.snapshot().connected:
                self.virtual_stage.connect()
            if not self.virtual_laser.snapshot().connected:
                self.virtual_laser.connect()
            if not self.virtual_attenuator.snapshot().connected:
                self.virtual_attenuator.connect()
            self._apply_mock_profile()
            self.mode_chip.setText("MOCK • SAFE")
            self._set_object_style(self.mode_chip, "chipSafe")
            if hasattr(self, "mock_scenario"):
                self._mock_scenario_changed(
                    self.mock_scenario.currentIndex()
                )
            self.statusBar().showMessage(
                "MOCK LAB — no real hardware commands can be issued",
                5000,
            )
        else:
            self.stage_mode.setCurrentIndex(1)
            self.laser_mode.setCurrentIndex(1)
            self.attenuator_mode.setCurrentIndex(1)
            self.mode_chip.setText("REAL LAB • DISARMED")
            self._set_object_style(self.mode_chip, "chipWarn")
            self.statusBar().showMessage(
                "REAL LAB selected — HXP/attenuator can connect; Pockels remains commissioning-locked",
                7000,
            )
        self._update_recipe_preflight_view()

    # ------------------------------------------------------ quick line moves
    def _start_quick_line(self, *, write: bool) -> None:
        if self._recipe_running or self._manual_write_line_active:
            QtWidgets.QMessageBox.information(
                self,
                "Controller busy",
                "Stop the active script/writing move before starting another line.",
            )
            return

        dx = self.quick_dx.value()
        dy = self.quick_dy.value()
        dz = self.quick_dz.value()
        velocity = self.quick_velocity.value()

        try:
            stage = self._stage_provider()
            if stage.is_busy():
                raise RuntimeError("stage is already moving")
            target = self._planned_stage_pose().plus(
                Pose6D(x=dx, y=dy, z=dz)
            )
            self._require_requested_motion(
                target,
                velocity_mm_s=velocity,
            )

            if write:
                if (
                    self.laser_mode.currentIndex() == 1
                    and not self.manual_beam_arm.isChecked()
                ):
                    raise RuntimeError(
                        "Arm manual real-beam control before a real writing move"
                    )
                self._laser_provider().set_gate(True)

            stage.move_line_incremental_with_target_velocity(
                dx,
                dy,
                dz,
                velocity,
            )
            if write:
                self._manual_write_line_active = True
                self._manual_write_line_started = True
                self._manual_write_line_description = (
                    f"dX={dx:.3f}, dY={dy:.3f}, dZ={dz:.3f} mm "
                    f"@ {velocity:.3f} mm/s"
                )
                self.quick_line_status.setText(
                    "WRITING • Pockels OPEN • "
                    + self._manual_write_line_description
                )
            else:
                self.quick_line_status.setText(
                    f"LINE MOVE • dX={dx:.3f}, dY={dy:.3f}, "
                    f"dZ={dz:.3f} mm @ {velocity:.3f} mm/s"
                )
        except Exception as exc:
            if write:
                self._close_all_pockels()
            QtWidgets.QMessageBox.critical(
                self,
                "Line move failed",
                str(exc),
            )

    def _quick_return_row(self) -> None:
        if self._recipe_running or self._manual_write_line_active:
            QtWidgets.QMessageBox.information(
                self,
                "Controller busy",
                "Finish/stop the active operation first.",
            )
            return
        self._close_all_pockels()
        try:
            stage = self._stage_provider()
            if stage.is_busy():
                raise RuntimeError("stage is already moving")
            dx = -self.quick_dx.value()
            dy = self.quick_row_pitch.value()
            velocity = self.quick_velocity.value()
            target = self._planned_stage_pose().plus(
                Pose6D(x=dx, y=dy)
            )
            self._require_requested_motion(
                target,
                velocity_mm_s=velocity,
            )
            stage.move_line_incremental_with_target_velocity(
                dx,
                dy,
                0.0,
                velocity,
            )
            self.quick_line_status.setText(
                f"RETURN • dX={dx:.3f} mm • row dY={dy:.4f} mm • beam CLOSED"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Return move failed",
                str(exc),
            )

    def _manual_write_line_tick(self) -> None:
        if not self._manual_write_line_active:
            return
        try:
            busy = self._stage_provider().is_busy()
        except Exception as exc:
            self._manual_write_line_active = False
            self._manual_write_line_started = False
            self._close_all_pockels()
            try:
                self._stage_provider().abort()
            except Exception as abort_exc:
                self.log_internal_error(
                    "manual write fail-safe abort",
                    abort_exc,
                )
            self.quick_line_status.setText(
                f"FAILED • beam close + motion abort requested • {exc}"
            )
            return

        if self._manual_write_line_started and not busy:
            self._manual_write_line_active = False
            self._manual_write_line_started = False
            self._close_all_pockels()
            self.quick_line_status.setText(
                "COMPLETE • Pockels CLOSED • "
                + self._manual_write_line_description
            )

    # ---------------------------------------------------------- commissioning
    def _commission_candidate_changed(self, _index: int) -> None:
        if not hasattr(self, "commission_profile"):
            return
        key = self.commission_profile.currentData()
        if not key:
            return
        candidate = self.legacy_profile.candidate(str(key))
        polarity = (
            f"OPEN={candidate.open_value}, CLOSED={candidate.closed_value}"
            if candidate.polarity_known
            else "states 0/1 observed; OPEN/CLOSED polarity unresolved"
        )
        self.commission_evidence_text.setText(
            f"{candidate.label}\n"
            f"GPIO: {candidate.gpio_name} • mask {candidate.mask} • {polarity}\n"
            f"Confidence: {candidate.confidence}\n"
            "Loading this candidate never marks the present wiring verified."
        )

    def _apply_commission_candidate_to_setup(self) -> None:
        key = self.commission_profile.currentData()
        if not key:
            return
        candidate = self.legacy_profile.candidate(str(key))
        idx = self.legacy_profile_combo.findData(candidate.key)
        if idx >= 0:
            self.legacy_profile_combo.setCurrentIndex(idx)

        self.gpio_name.setText(candidate.gpio_name)
        self.gpio_mask.setValue(candidate.mask)
        if candidate.polarity_known:
            self.gpio_open.setValue(int(candidate.open_value))
            self.gpio_closed.setValue(int(candidate.closed_value))
        else:
            # The files prove the two states are 0 and 1 but not their physical
            # polarity on the current wiring. Populate a visible draft pair only;
            # verification remains false and real provider arming remains blocked.
            self.gpio_open.setValue(1)
            self.gpio_closed.setValue(0)

        self.wiring_verified.setChecked(False)
        for cb in self.commission_checks:
            cb.setChecked(False)
        self.commission_log.appendPlainText(
            "Loaded candidate into Setup WITHOUT verification: "
            + candidate.label
        )
        if not candidate.polarity_known:
            self.commission_log.appendPlainText(
                "  NOTE: OPEN=1/CLOSED=0 is only a draft orientation for "
                "the LabVIEW-v3 0/1 states. Confirm physical polarity first."
            )

    def _mock_scenario_changed(self, index: int) -> None:
        if not hasattr(self, "lab_mode") or self.lab_mode.currentIndex() != 0:
            return

        # Re-establish nominal state first, then inject one failure.
        self.virtual_stage.connect()
        self.virtual_laser.connect()
        self.virtual_attenuator.connect()
        self._apply_mock_profile()

        if index == 1:
            self.virtual_stage.disconnect()
        elif index == 2:
            self.virtual_laser.disconnect()
        elif index == 3:
            self.virtual_attenuator.disconnect()

        if hasattr(self, "commission_log"):
            self.commission_log.appendPlainText(
                "MOCK scenario: " + self.mock_scenario.currentText()
            )

    def _run_read_only_self_test(self) -> None:
        if self._selftest_future is not None and not self._selftest_future.done():
            return

        self.commission_log.appendPlainText(
            "\n=== READ-ONLY SELF TEST ==="
        )

        if self.lab_mode.currentIndex() == 0:
            candidate = self._selected_candidate()
            snap = self.virtual_stage.snapshot()
            laser = self.virtual_laser.snapshot()
            attenuator = self.virtual_attenuator.snapshot()
            self.commission_log.appendPlainText(
                "MODE: MOCK LAB\n"
                f"Stage connected: {snap.connected}\n"
                f"Pose: {snap.actual.as_tuple()}\n"
                f"Mock Pockels profile: {candidate.label}\n"
                f"Mock logical Pockels open: {laser.pockels_open}\n"
                f"Mock GPIO metadata: {dict(laser.metadata)}\n"
                f"Mock attenuator connected: {attenuator.connected}\n"
                f"Mock attenuation: {attenuator.transmission_percent:.1f} %\n"
                "RESULT: mock read-only checks complete"
            )
            return

        if self.real_stage is None or not self.real_stage.client.connected:
            self.commission_log.appendPlainText(
                "RESULT: FAIL — real HXP is not connected"
            )
            return

        client = self.real_stage.client
        group = self.hxp_group.text().strip() or "HEXAPOD"

        def worker() -> list[str]:
            lines = [
                "MODE: REAL LAB",
                f"HXP: {self.hxp_host.text().strip()}:{self.hxp_port.value()}",
                f"Firmware: {client.firmware_version()}",
            ]
            pose = client.current_pose(group)
            lines.append(
                "Pose: "
                + ", ".join(
                    f"{axis}={value:.6f}"
                    for axis, value in zip("XYZUVW", pose.as_tuple())
                )
            )
            for gpio in ("GPIO1.DO", "GPIO3.DO", "GPIO4.DO"):
                try:
                    lines.append(
                        f"{gpio}: {client.digital_get(gpio)}"
                    )
                except Exception as exc:
                    lines.append(f"{gpio}: READ FAILED ({exc})")
            try:
                lines.append(
                    "GPIO2.DAC1: "
                    f"{client.analog_get('GPIO2.DAC1'):.6g}"
                )
            except Exception as exc:
                lines.append(
                    f"GPIO2.DAC1: READ FAILED ({exc})"
                )
            lines.append(
                "RESULT: read-only test complete — no motion/GPIO writes issued"
            )
            return lines

        self._selftest_future = self._poll_pool.submit(worker)
        self.commission_log.appendPlainText(
            "Reading firmware / pose / legacy GPIO channels…"
        )

    def _commission_selftest_tick(self) -> None:
        if self._selftest_future is None or not self._selftest_future.done():
            return
        try:
            lines = self._selftest_future.result()
        except Exception as exc:
            lines = [f"RESULT: FAIL — {exc}"]
        self._selftest_future = None
        self.commission_log.appendPlainText("\n".join(lines))

    def _read_raw_analog(self) -> None:
        gpio = self.raw_analog_gpio.text().strip()
        if not gpio:
            return
        if self.lab_mode.currentIndex() == 0:
            self.commission_log.appendPlainText(
                f"MOCK raw analogue {gpio}: "
                f"{self.raw_analog_value.value():.3f} (simulated)"
            )
            return
        if self.real_stage is None or not self.real_stage.client.connected:
            QtWidgets.QMessageBox.warning(
                self,
                "HXP not connected",
                "Connect the real HXP before reading an analogue GPIO.",
            )
            return
        try:
            value = self.real_stage.client.analog_get(gpio)
            self.commission_log.appendPlainText(
                f"READ {gpio} = {value:.6g}"
            )
        except Exception as exc:
            self.commission_log.appendPlainText(
                f"READ {gpio} FAILED: {exc}"
            )

    def _write_raw_analog(self) -> None:
        if self.lab_mode.currentIndex() != 1:
            QtWidgets.QMessageBox.information(
                self,
                "REAL LAB required",
                "Raw analogue writes are only available in REAL LAB commissioning.",
            )
            return
        if not self.raw_analog_arm.isChecked():
            QtWidgets.QMessageBox.warning(
                self,
                "Raw analogue path not confirmed",
                "Confirm that the selected GPIO is the present power/attenuation "
                "path before issuing a raw command.",
            )
            return
        if self.real_stage is None or not self.real_stage.client.connected:
            QtWidgets.QMessageBox.warning(
                self,
                "HXP not connected",
                "Connect the real HXP first.",
            )
            return
        if self._laser_snapshot().pockels_open:
            QtWidgets.QMessageBox.warning(
                self,
                "Close process beam first",
                "Raw analogue commissioning writes are blocked while the "
                "Pockels cell is OPEN.",
            )
            return
        if self.real_stage.is_busy():
            QtWidgets.QMessageBox.warning(
                self,
                "Stage moving",
                "Wait for stage motion to finish before changing the raw analogue output.",
            )
            return

        gpio = self.raw_analog_gpio.text().strip()
        value = self.raw_analog_value.value()
        answer = QtWidgets.QMessageBox.warning(
            self,
            "Write uncalibrated analogue value?",
            f"Write raw value {value:.3f} to {gpio}?\n\n"
            "This is a commissioning command taken from the legacy software "
            "pattern; it is not calibrated optical transmission or pulse energy.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            self.real_stage.client.analog_set(gpio, value)
            readback = self.real_stage.client.analog_get(gpio)
            self.commission_log.appendPlainText(
                f"WRITE {gpio} <- {value:.6g}; readback={readback:.6g}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Analogue write failed",
                str(exc),
            )

    def _mark_current_mapping_verified(self) -> None:
        if self.lab_mode.currentIndex() != 1:
            QtWidgets.QMessageBox.information(
                self,
                "Select REAL LAB",
                "Physical mapping verification is only meaningful in REAL LAB mode.",
            )
            return
        if not all(cb.isChecked() for cb in self.commission_checks):
            QtWidgets.QMessageBox.warning(
                self,
                "Commissioning incomplete",
                "Complete every physical commissioning checkbox before marking "
                "the current Pockels mapping verified.",
            )
            return
        if not self.gpio_name.text().strip() or self.gpio_mask.value() <= 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid mapping",
                "Enter a GPIO name and non-zero mask first.",
            )
            return
        if self.gpio_open.value() == self.gpio_closed.value():
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid mapping",
                "OPEN and CLOSED values must be different.",
            )
            return

        self.wiring_verified.setChecked(True)
        self.commission_log.appendPlainText(
            "CURRENT MAPPING MARKED VERIFIED by operator: "
            f"{self.gpio_name.text().strip()} mask={self.gpio_mask.value()} "
            f"OPEN={self.gpio_open.value()} CLOSED={self.gpio_closed.value()}"
        )
        self.statusBar().showMessage(
            "Pockels mapping marked verified; provider still must be explicitly armed",
            7000,
        )

    def _clear_hardware_verification(self) -> None:
        self._close_all_pockels()
        if self.real_laser is not None:
            try:
                self.real_laser.disconnect()
            except Exception:
                pass
            self.real_laser = None
        self.wiring_verified.setChecked(False)
        for cb in self.commission_checks:
            cb.setChecked(False)
        self.commission_log.appendPlainText(
            "Hardware verification cleared; real Pockels provider disarmed"
        )

    # ------------------------------------------------------------ sweep tool
    def _generate_sweep_recipe(self) -> None:
        if self._recipe_running:
            QtWidgets.QMessageBox.information(
                self,
                "Script is running",
                "Stop the active script before replacing its modules.",
            )
            return
        try:
            spec = RasterSweepSpec(
                name="Raster writing sweep",
                write_dx_mm=self.sweep_write_dx.value(),
                row_pitch_mm=self.sweep_row_pitch.value(),
                series_spacing_mm=self.sweep_series_spacing.value(),
                velocity_start_mm_s=self.sweep_v_start.value(),
                velocity_stop_mm_s=self.sweep_v_stop.value(),
                velocity_step_mm_s=self.sweep_v_step.value(),
                attenuation_start_percent=self.sweep_att_start.value(),
                attenuation_stop_percent=self.sweep_att_stop.value(),
                attenuation_step_percent=self.sweep_att_step.value(),
                include_attenuator_steps=self.sweep_use_attenuation.isChecked(),
                return_velocity_mm_s=self.sweep_return_velocity.value(),
                use_write_blocks=self.sweep_write_blocks.isChecked(),
            )
            recipe, summary = build_raster_sweep(spec)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Sweep definition invalid",
                str(exc),
            )
            return

        if self.recipe_list.count() > 0:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace current recipe?",
                "Generating the sweep will replace the current recipe.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        self.recipe = recipe
        self.recipe_name.setText(recipe.name)
        self._refresh_recipe_list()
        self.sweep_summary.setText(
            f"Generated {summary.total_write_lines} writing lines across "
            f"{summary.series_count} series • {summary.total_steps} recipe steps • "
            f"~{summary.approximate_motion_time_s:.1f} s motion time "
            "(excludes controller/setup overhead)."
        )
        self.statusBar().showMessage(
            f"Generated sweep with {summary.total_write_lines} writing lines",
            5000,
        )

    # ------------------------------------------------------------- providers
    def _stage_provider(self) -> HexapodProvider:
        if self.stage_mode.currentIndex() == 0:
            return self.virtual_stage
        if self.real_stage is None or not self.real_stage.client.connected:
            raise ConnectionError(
                "Real HXP is selected but is not connected"
            )
        return self.real_stage

    def _laser_provider(self) -> LaserGateProvider:
        if self.laser_mode.currentIndex() == 0:
            return self.virtual_laser
        if self.real_laser is None:
            raise ConnectionError(
                "Real PHAROS LX13/Pockels provider is selected but not armed"
            )
        snap = self.real_laser.snapshot()
        if not snap.connected:
            raise ConnectionError(
                "Real PHAROS LX13/Pockels provider is not connected"
            )
        return self.real_laser

    def _attenuator_provider(self) -> AttenuatorProvider:
        if self.attenuator_mode.currentIndex() == 0:
            return self.virtual_attenuator
        return self.real_attenuator

    def _laser_snapshot(self) -> LaserSnapshot:
        if self.laser_mode.currentIndex() == 0:
            return self.virtual_laser.snapshot()
        if self.real_laser is None:
            return LaserSnapshot(
                timestamp_s=time.time(),
                gate_enabled=False,
                connected=False,
                provider="hxp-lx13-real",
                connector_name="PHAROS LX13",
                readback_known=False,
            )
        return self.real_laser.snapshot()

    def _attenuator_snapshot(self) -> AttenuatorSnapshot:
        if self.attenuator_mode.currentIndex() == 0:
            return self.virtual_attenuator.snapshot()
        return self.real_attenuator.snapshot()

    # ---------------------------------------------------------- configuration
    def _config_dict(self) -> dict:
        return {
            "operating_mode": (
                "mock" if self.lab_mode.currentIndex() == 0 else "real"
            ),
            "legacy_profile_key": self._selected_legacy_candidate_key,
            "mock_time_scale": self.mock_time_scale.value(),
            "workspace_limits": self.workspace_limits.to_dict(),
            "sample_calibration": self.sample_calibration.to_dict(),
            "hxp": {
                "host": self.hxp_host.text().strip(),
                "port": self.hxp_port.value(),
                "timeout_s": self.hxp_timeout.value(),
                "group": self.hxp_group.text().strip() or "HEXAPOD",
                "coordinate_system": self.hxp_coords.currentText(),
            },
            "pockels": {
                "connector": "PHAROS LX13",
                "provider": (
                    "virtual"
                    if self.laser_mode.currentIndex() == 0
                    else "hxp_gpio"
                ),
                "gpio_name": self.gpio_name.text().strip(),
                "mask": self.gpio_mask.value(),
                "open_value": self.gpio_open.value(),
                "closed_value": self.gpio_closed.value(),
                "wiring_verified": self.wiring_verified.isChecked(),
            },
            "attenuator": {
                "provider": (
                    "virtual"
                    if self.attenuator_mode.currentIndex() == 0
                    else "hxp_analog"
                ),
                "model": self.attenuator_model.text().strip(),
                "transmission_percent": self.attenuator_spin.value(),
            },
        }

    def _load_config_dict(self, cfg: dict) -> None:
        self.workspace_limits = WorkspaceLimits.from_dict(            cfg.get("workspace_limits"),
            self.kinematics,
        )
        self._workspace_pending_source = self.workspace_limits.source
        self._apply_workspace_limits_to_widgets()

        # Sample geometry is restored, but the bounds are never re-armed by a
        # file: the operator confirms the sample is really there.
        self._set_sample_calibration(
            replace(
                SampleCalibration.from_dict(cfg.get("sample_calibration")),
                enforce_xy=False,
                enforce_z=False,
            )
        )

        profile_key = str(
            cfg.get("legacy_profile_key", self._selected_legacy_candidate_key)
        )
        idx = self.legacy_profile_combo.findData(profile_key)
        if idx >= 0:
            self.legacy_profile_combo.setCurrentIndex(idx)
        self.mock_time_scale.setValue(
            float(cfg.get("mock_time_scale", 1.0))
        )

        hxp = cfg.get("hxp", {})
        self.hxp_host.setText(str(hxp.get("host", self.hxp_host.text())))
        self.hxp_port.setValue(int(hxp.get("port", self.hxp_port.value())))
        self.hxp_timeout.setValue(
            float(
                hxp.get(
                    "timeout_s",
                    float(hxp.get("timeout_ms", 10000)) / 1000.0,
                )
            )
        )
        self.hxp_group.setText(
            str(hxp.get("group", self.hxp_group.text()))
        )
        coord = str(
            hxp.get("coordinate_system", self.hxp_coords.currentText())
        )
        idx = self.hxp_coords.findText(coord)
        if idx >= 0:
            self.hxp_coords.setCurrentIndex(idx)

        pockels = cfg.get("pockels", cfg.get("laser", {}))
        provider = str(pockels.get("provider", "virtual"))
        self.laser_mode.setCurrentIndex(
            1 if provider in {"hxp_gpio", "real"} else 0
        )
        self.gpio_name.setText(str(pockels.get("gpio_name", "")))
        self.gpio_mask.setValue(int(pockels.get("mask", 0)))
        self.gpio_open.setValue(
            int(
                pockels.get(
                    "open_value",
                    pockels.get("enabled_value", 0),
                )
            )
        )
        self.gpio_closed.setValue(
            int(
                pockels.get(
                    "closed_value",
                    pockels.get("disabled_value", 0),
                )
            )
        )
        self.wiring_verified.setChecked(
            bool(pockels.get("wiring_verified", False))
        )

        att = cfg.get("attenuator", {})
        self.attenuator_mode.setCurrentIndex(
            1 if att.get("provider") in {"hxp_analog", "real"} else 0
        )
        self.attenuator_model.setText(str(att.get("model", "")))
        transmission = float(att.get("transmission_percent", 0.0))
        self.attenuator_spin.setValue(
            max(0.0, min(100.0, transmission))
        )
        if self.attenuator_mode.currentIndex() == 0:
            self.virtual_attenuator.set_transmission_percent(
                self.attenuator_spin.value()
            )

        operating_mode = str(cfg.get("operating_mode", "mock")).lower()
        target_mode = 1 if operating_mode == "real" else 0
        self.lab_mode.setCurrentIndex(target_mode)
        # setCurrentIndex does not emit when already at the requested index.
        self._lab_mode_changed(target_mode)

    def _load_default_config(self) -> None:
        if not DEFAULT_CONFIG.is_file():
            return
        try:
            self._load_config_dict(
                json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Config warning: {exc}")

    def _load_hardware_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load hardware configuration",
            str(APP_ROOT),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            cfg = json.loads(Path(path).read_text(encoding="utf-8"))
            self._load_config_dict(cfg)
            self.statusBar().showMessage(
                f"Loaded configuration: {Path(path).name}",
                6000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Configuration load failed",
                str(exc),
            )

    def _save_hardware_config_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save hardware configuration",
            str(APP_ROOT / "hardware_config.local.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(self._config_dict(), indent=2),
                encoding="utf-8",
            )
            self.statusBar().showMessage(
                f"Saved configuration: {Path(path).name}",
                6000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Configuration save failed",
                str(exc),
            )

    # --------------------------------------------------------------- HXP
    def _stage_mode_changed(self, _index: int) -> None:
        self.real_script_arm.setChecked(False)
        self.manual_beam_arm.setChecked(False)
        if self.stage_mode.currentIndex() == 0:
            if not self.virtual_stage.snapshot().connected:
                self.virtual_stage.connect()
            self._last_stage_snapshot = self.virtual_stage.snapshot()
        elif (
            self.real_stage is not None
            and self.real_stage.client.connected
        ):
            try:
                self._last_stage_snapshot = self.real_stage.snapshot()
                self._ensure_pose_widget_ranges_include(
                    self._last_stage_snapshot.actual
                )
                self._target_from_actual()
            except Exception as exc:
                self._last_stage_snapshot = HexapodSnapshot(
                    timestamp_s=time.time(),
                    actual=self._last_stage_snapshot.actual,
                    state=MotionState.FAULT,
                    connected=True,
                    provider="hxp-real",
                    status_text=f"INITIAL POLL ERROR: {exc}",
                )
        else:
            self._last_stage_snapshot = HexapodSnapshot(
                timestamp_s=time.time(),
                actual=self._last_stage_snapshot.actual,
                state=MotionState.DISCONNECTED,
                connected=False,
                provider="hxp-real",
                status_text="REAL HXP NOT CONNECTED",
            )
        self._update_recipe_preflight_view()

    def _connect_stage(self) -> None:
        if self.stage_mode.currentIndex() == 0:
            self.virtual_stage.connect()
            self._last_stage_snapshot = self.virtual_stage.snapshot()
            self.statusBar().showMessage(
                "Virtual hexapod connected",
                4000,
            )
            return
        self._connect_real_hxp()

    def _connect_real_hxp(self) -> None:
        try:
            cfg = HXPProviderConfig(
                host=self.hxp_host.text().strip(),
                port=self.hxp_port.value(),
                group=self.hxp_group.text().strip() or "HEXAPOD",
                coordinate_system=self.hxp_coords.currentText(),
                timeout_s=self.hxp_timeout.value(),
            )
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )

            # A real Pockels provider owns HXP I/O sockets. Tear it down before
            # replacing/reconnecting the HXP client so no stale client survives.
            if self.real_laser is not None:
                try:
                    self.real_laser.disconnect()
                except Exception as exc:
                    self._diag(f"Pockels close/disconnect warning: {exc}")
                self.real_laser = None

            if self.real_stage is not None:
                try:
                    self.real_stage.disconnect()
                except Exception:
                    pass

            self.real_stage = HXPProvider(cfg)
            self.real_stage.connect()
            self._real_poll_failures = 0

            # Verify the two mutable HXP Cartesian frames against the controller
            # backup/live commissioning snapshot. A frame changed elsewhere can
            # make otherwise-correct sample coordinates point somewhere else.
            try:
                live_work = self.real_stage.client.coordinate_system_get(
                    "Work", cfg.group
                )
                live_tool = self.real_stage.client.coordinate_system_get(
                    "Tool", cfg.group
                )
                expected_work = Pose6D.from_iterable(
                    self.controller_profile.work_in_world
                )
                expected_tool = Pose6D.from_iterable(
                    self.controller_profile.tool_in_carriage
                )
                max_frame_error = max(
                    live_work.max_abs_delta(expected_work),
                    live_tool.max_abs_delta(expected_tool),
                )
                self._real_frames_match_profile = max_frame_error <= 1e-4
                self._diag(
                    "Live frames: Work="
                    + str(live_work.as_tuple())
                    + " Tool="
                    + str(live_tool.as_tuple())
                )
                if not self._real_frames_match_profile:
                    self._diag(
                        "FRAME MISMATCH: live Work/Tool differ from the "
                        "commissioned controller profile"
                    )
            except Exception as exc:
                self._real_frames_match_profile = False
                self._diag(f"Could not verify Work/Tool frames: {exc}")

            self.real_attenuator = HXPAnalogAttenuatorProvider(
                self.real_stage.client,
                HXPAnalogAttenuatorConfig(
                    gpio_name=self.controller_profile.attenuator_gpio,
                    raw_min=self.controller_profile.attenuator_raw_min,
                    raw_max=self.controller_profile.attenuator_raw_max,
                    transmission_min_percent=(
                        self.controller_profile.attenuator_transmission_min_percent
                    ),
                    transmission_max_percent=(
                        self.controller_profile.attenuator_transmission_max_percent
                    ),
                    inverted=self.controller_profile.attenuator_inverted,
                ),
            )
            try:
                self.real_attenuator.connect()
                att_snap = self.real_attenuator.snapshot()
                self.attenuator_spin.setValue(
                    att_snap.transmission_percent
                )
                self._diag(
                    f"Attenuator readback: "
                    f"{att_snap.transmission_percent:.1f}% "
                    f"from {self.controller_profile.attenuator_gpio}"
                )
            except Exception as exc:
                self._diag(f"Attenuator readback unavailable: {exc}")

            if self.stage_mode.currentIndex() == 1:
                self._last_stage_snapshot = self.real_stage.snapshot()
                self._ensure_pose_widget_ranges_include(
                    self._last_stage_snapshot.actual
                )
                self._target_from_actual()

            self._diag(
                f"Connected HXP {cfg.host}:{cfg.port}; "
                f"group={cfg.group}, frame={cfg.coordinate_system}"
            )
            if self._real_frames_match_profile:
                self.statusBar().showMessage(
                    f"Connected to HXP {cfg.host}:{cfg.port}; "
                    "Work/Tool frames match commissioned profile",
                    6000,
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    "HXP frame verification required",
                    "The HXP connected, but the live Work/Tool coordinate "
                    "systems could not be confirmed against the commissioned "
                    "profile. Real motion remains blocked until this is resolved.",
                )
        except Exception as exc:
            self._real_frames_match_profile = False
            if self.real_stage is not None:
                try:
                    self.real_stage.disconnect()
                except Exception:
                    pass
            QtWidgets.QMessageBox.critical(
                self,
                "HXP connection failed",
                str(exc),
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._update_recipe_preflight_view()

    def _disconnect_stage(self) -> None:
        if self.stage_mode.currentIndex() == 0:
            self.virtual_stage.disconnect()
            self.statusBar().showMessage(
                "Virtual stage disconnected",
                4000,
            )
            return
        self._disconnect_real_hxp()

    def _disconnect_real_hxp(self) -> None:
        warnings: list[str] = []
        try:
            # Fail toward a safe state before tearing down communication.
            if self.real_laser is not None:
                try:
                    self.real_laser.disconnect()
                    self._beam_close_failed = False
                except Exception as exc:
                    self._beam_close_failed = True
                    warnings.append(f"Pockels close failed: {exc}")
                finally:
                    self.real_laser = None

            if self.real_stage is not None:
                try:
                    if self.real_stage.is_busy():
                        self.real_stage.abort()
                except Exception as exc:
                    warnings.append(f"motion abort failed: {exc}")

            try:
                self.real_attenuator.disconnect()
            except Exception as exc:
                warnings.append(f"attenuator disconnect failed: {exc}")
            self.real_attenuator = UnconfiguredAttenuatorProvider()

            if self.real_stage is not None:
                try:
                    self.real_stage.disconnect()
                except Exception as exc:
                    warnings.append(f"HXP socket close failed: {exc}")

            self._real_frames_match_profile = False
            self._real_poll_failures = 0
            self._poll_future = None

            if warnings:
                for warning in warnings:
                    self._diag("Disconnect warning: " + warning)
                self.statusBar().showMessage(
                    "HXP disconnected with warnings: " + warnings[0],
                    10000,
                )
            else:
                self.statusBar().showMessage(
                    "Real HXP disconnected; motion stopped and Pockels close requested",
                    5000,
                )
        finally:
            self._update_recipe_preflight_view()

    def _ensure_pose_widget_ranges_include(
        self,
        pose: Pose6D,
    ) -> None:
        """Keep live real poses representable without blessing fake limits.

        Spin-box ranges are a UI concern. Widening them to include the measured
        live pose does not mark the workspace verified and does not change the
        HXP's own motion limits.
        """
        values = pose.as_tuple()
        for boxes_name in ("pose_boxes", "script_pose_boxes"):
            boxes = getattr(self, boxes_name, {})
            for index, axis in enumerate("XYZUVW"):
                box = boxes.get(axis)
                if box is None:
                    continue
                value = float(values[index])
                margin = 1.0 if axis in "XYZ" else 0.5
                if value < box.minimum():
                    box.setMinimum(value - margin)
                if value > box.maximum():
                    box.setMaximum(value + margin)

    def _real_readback_is_fresh(self, max_age_s: float = 1.0) -> bool:
        if self.stage_mode.currentIndex() != 1:
            return True
        snap = self._last_stage_snapshot
        return bool(
            snap.connected
            and snap.state in (MotionState.IDLE, MotionState.MOVING)
            and (time.time() - snap.timestamp_s) <= max_age_s
        )

    def _target_pose(self) -> Pose6D:
        return Pose6D(
            *(self.pose_boxes[axis].value() for axis in "XYZUVW")
        )

    def _move_absolute(self) -> None:
        try:
            target = self._target_pose()
            stage = self._stage_provider()

            if self.stage_mode.currentIndex() == 1:
                current = self._planned_stage_pose()
                rotation_unchanged = all(
                    abs(a - b) <= 5e-4
                    for a, b in zip(
                        target.as_tuple()[3:],
                        current.as_tuple()[3:],
                    )
                )
                if rotation_unchanged:
                    dx = target.x - current.x
                    dy = target.y - current.y
                    dz = target.z - current.z
                    if max(abs(dx), abs(dy), abs(dz)) <= 1e-12:
                        return
                    velocity = 0.10
                    self._require_requested_motion(
                        target,
                        velocity_mm_s=velocity,
                    )
                    stage.move_line_incremental_with_target_velocity(
                        dx,
                        dy,
                        dz,
                        velocity,
                    )
                    self.statusBar().showMessage(
                        "Real XYZ target move sent as controller-preflighted "
                        "Line at 0.10 mm/s",
                        5000,
                    )
                    return

            self._require_requested_motion(target)
            stage.move_absolute(target)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Move failed",
                str(exc),
            )

    def _target_from_actual(self) -> None:
        pose = self._last_stage_snapshot.actual
        self._ensure_pose_widget_ranges_include(pose)
        for axis, value in zip("XYZUVW", pose.as_tuple()):
            self.pose_boxes[axis].setValue(value)

    def _jog(self, axis: str, sign: float) -> None:
        values = [0.0] * 6
        idx = "XYZUVW".index(axis)
        values[idx] = sign * (
            self.jog_mm.value() if idx < 3 else self.jog_deg.value()
        )
        try:
            delta = Pose6D.from_iterable(values)
            target = self._planned_stage_pose().plus(delta)
            stage = self._stage_provider()

            if self.stage_mode.currentIndex() == 1 and axis in "XYZ":
                # Commissioning-safe translation jog: ask the HXP whether the
                # complete Line is executable before issuing the matching Line
                # command. No fake Cartesian box or decorative STEP kinematics
                # are used as safety authority.
                velocity = 0.10
                self._require_requested_motion(
                    target,
                    velocity_mm_s=velocity,
                )
                stage.move_line_incremental_with_target_velocity(
                    delta.x,
                    delta.y,
                    delta.z,
                    velocity,
                )
                return

            # Virtual jogs and real rotational jogs keep the ordinary path.
            # Real U/V/W therefore remain blocked until a coupled rotational
            # workspace is deliberately commissioned.
            self._require_requested_motion(target)
            stage.move_incremental(delta)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Jog failed",
                str(exc),
            )

    def _home(self) -> None:
        real = self.stage_mode.currentIndex() == 1
        if real:
            if self.real_stage is None or not self.real_stage.client.connected:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Home blocked",
                    "Connect the real HXP first.",
                )
                return
            try:
                status = self.real_stage.client.group_status(
                    self.real_stage.config.group
                )
                try:
                    status_text = self.real_stage.client.group_status_text(
                        status
                    )
                except Exception:
                    status_text = f"status {status}"
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Home blocked",
                    f"Could not read HXP group state: {exc}",
                )
                return

            if status in (11, 12):
                QtWidgets.QMessageBox.information(
                    self,
                    "HXP already referenced",
                    "The HXP is already in a referenced Ready state "
                    f"({status}: {status_text}). No home search was sent.\n\n"
                    "A deliberate re-home from this state requires the Newport "
                    "kill → initialize → home sequence; that sequence is not "
                    "exposed as a one-click action in this GUI.",
                )
                return

            if status != 42:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Home blocked",
                    "The controller is not in the expected Not Referenced "
                    f"state for homing: {status} ({status_text}).",
                )
                return

            answer = QtWidgets.QMessageBox.warning(
                self,
                "Home real HXP?",
                "The group is Not Referenced. A real home search can move all "
                "six struts through a substantial path. Confirm the physical "
                "setup is clear and the process beam is safely blocked.",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        try:
            if not real:
                self._require_requested_motion(Pose6D())
            self._stage_provider().home()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Home failed",
                str(exc),
            )

    def _abort_all(self) -> None:
        self._manual_write_line_active = False
        self._manual_write_line_started = False
        self._recipe_write_block_open = False
        self._close_all_pockels()
        try:
            self._stage_provider().abort()
        except Exception as exc:
            self.statusBar().showMessage(
                f"Motion-abort warning: {exc}",
                7000,
            )
        self._recipe_running = False
        if hasattr(self, "recipe_list"):
            self.recipe_list.setEnabled(True)
        if hasattr(self, "module_library"):
            self.module_library.setEnabled(True)
        self.recipe_progress.setText(
            "STOPPED — "
            + (
                "BEAM CLOSE FAILED / STATE UNKNOWN"
                if self._beam_close_failed
                else "close-beam command issued"
            )
        )

    # ------------------------------------------------------ Pockels / laser
    def _laser_mode_changed(self, _index: int) -> None:
        # Mode changes must never leave a previously-selected real Pockels
        # command OPEN and then hide that provider from the operator.
        self._close_all_pockels()
        self.manual_beam_arm.setChecked(False)
        self.real_script_arm.setChecked(False)
        if self.laser_mode.currentIndex() == 0:
            if not self.virtual_laser.snapshot().connected:
                self.virtual_laser.connect()
        self._update_recipe_preflight_view()

    def _connect_laser(self) -> None:
        try:
            if self.laser_mode.currentIndex() == 0:
                self.virtual_laser.connect()
                self.virtual_laser.set_gate(False)
                self.statusBar().showMessage(
                    "Virtual Pockels provider connected",
                    4000,
                )
                return

            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                raise RuntimeError(
                    "Connect the real HXP first. The current LX13 implementation "
                    "uses an HXP digital-output socket."
                )

            cfg = HXPDigitalLaserConfig(
                gpio_name=self.gpio_name.text().strip(),
                mask=self.gpio_mask.value(),
                enabled_value=self.gpio_open.value(),
                disabled_value=self.gpio_closed.value(),
                connector_name="PHAROS LX13 / Pockels cell",
                wiring_verified=self.wiring_verified.isChecked(),
            )
            self.real_laser = HXPDigitalLaserGate(
                self.real_stage.client,
                cfg,
            )
            self.real_laser.connect()
            self._beam_close_failed = False
            self._diag(
                "Real LX13/Pockels provider armed; CLOSED state requested"
            )
            self.statusBar().showMessage(
                "Real LX13/Pockels provider armed; initial state CLOSED",
                6000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Pockels provider not armed",
                str(exc),
            )
        self._update_recipe_preflight_view()

    def _set_pockels(
        self,
        open_: bool,
        *,
        manual: bool = False,
        script: bool = False,
    ) -> None:
        try:
            if (
                open_
                and self.laser_mode.currentIndex() == 1
                and manual
                and not self.manual_beam_arm.isChecked()
            ):
                raise RuntimeError(
                    "Real manual beam control is not armed. "
                    "Check ARM MANUAL REAL-BEAM CONTROL first."
                )
            if (
                open_
                and self.laser_mode.currentIndex() == 1
                and script
                and not self.real_script_arm.isChecked()
            ):
                raise RuntimeError(
                    "Real script execution is not armed"
                )
            self._laser_provider().set_gate(bool(open_))
            self.statusBar().showMessage(
                "Pockels OPEN — process beam enabled"
                if open_
                else "Pockels CLOSED — process beam disabled",
                3500,
            )
        except Exception as exc:
            if script:
                raise
            QtWidgets.QMessageBox.critical(
                self,
                "Pockels command failed",
                str(exc),
            )

    def _close_all_pockels(self) -> None:
        # Request closure on every provider we may have touched. A failure on
        # the real provider is never silently presented as a confirmed CLOSED
        # state; the physical interlock/shutter remains authoritative.
        errors: list[str] = []
        for provider in (self.virtual_laser, self.real_laser):
            if provider is None:
                continue
            try:
                provider.safe_off()
            except Exception as exc:
                self.log_internal_error(
                    f"Pockels close via {provider.name}",
                    exc,
                )
                if provider is self.real_laser:
                    errors.append(str(exc))

        self._beam_close_failed = bool(errors)
        if errors:
            self.statusBar().showMessage(
                "REAL POCKELS CLOSE FAILED — beam state unknown; use the "
                "physical interlock/shutter: " + errors[0],
                12000,
            )
            if hasattr(self, "beam_chip"):
                self.beam_chip.setText("BEAM STATE UNKNOWN")
                self._set_object_style(self.beam_chip, "chipWarn")
        else:
            self.statusBar().showMessage(
                "Close-beam request issued",
                3500,
            )


    # ----------------------------------------------------------- attenuator
    def _attenuator_mode_changed(self, _index: int) -> None:
        self.real_script_arm.setChecked(False)
        if self.attenuator_mode.currentIndex() == 0:
            if not self.virtual_attenuator.snapshot().connected:
                self.virtual_attenuator.connect()
        self._update_recipe_preflight_view()

    def _connect_attenuator(self) -> None:
        try:
            self._attenuator_provider().connect()
            self.statusBar().showMessage(
                "Attenuator provider connected",
                4000,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.information(
                self,
                "Attenuator driver required",
                str(exc),
            )
        self._update_recipe_preflight_view()

    def _attenuator_preset(self, value: float) -> None:
        self.attenuator_spin.setValue(float(value))

    def _set_attenuator_manual(self) -> None:
        try:
            self._set_attenuator(
                self.attenuator_spin.value(),
                script=False,
            )
        except Exception as exc:
            self.statusBar().showMessage(
                f"Attenuator command failed: {exc}",
                8000,
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Attenuator command failed",
                str(exc),
            )

    def _set_attenuator(
        self,
        value: float,
        *,
        script: bool = False,
    ) -> None:
        real = self.attenuator_mode.currentIndex() == 1
        if real and script and not self.real_script_arm.isChecked():
            raise RuntimeError("Real script execution is not armed")
        if real and self._laser_snapshot().pockels_open:
            raise RuntimeError(
                "Real attenuator changes are blocked while the Pockels cell is OPEN"
            )
        provider = self._attenuator_provider()
        provider.set_transmission_percent(float(value))
        snap = provider.snapshot()
        raw = snap.metadata.get("raw_readback")
        suffix = f" • DAC={float(raw):.3f}" if raw is not None else ""
        self.statusBar().showMessage(
            f"Attenuator setpoint {float(value):.1f} %{suffix}",
            3000,
        )

    # --------------------------------------------------------------- CAD
    def _autoload_step_if_present(self) -> None:
        candidates = (
            APP_ROOT / "Stewart Platform.STEP",
            APP_ROOT / "Stewart Platform.step",
            ASSETS_DIR / "Stewart Platform.STEP",
            APP_ROOT.parent / "Stewart Platform.STEP",
        )
        step_path = next((p for p in candidates if p.is_file()), None)
        if step_path is None:
            return
        try:
            self.viewer.load_step(str(step_path))
            label = f"Exact STEP CAD • {step_path.name} • AUTO"
            self.cad_status.setText(label)
            self.setup_cad_status.setText(label)
        except Exception as exc:
            message = (
                f"STEP found ({step_path.name}) but exact CAD was not loaded: {exc}"
            )
            self.cad_status.setText("CAD-derived rig • STEP load unavailable")
            self.setup_cad_status.setText(message)
            self.statusBar().showMessage(message, 9000)

    def _load_step_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Stewart-platform STEP",
            str(APP_ROOT),
            "STEP files (*.step *.stp *.STEP *.STP)",
        )
        if not path:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )
            self.viewer.load_step(path)
            label = f"Exact STEP CAD • {Path(path).name}"
            self.cad_status.setText(label)
            self.setup_cad_status.setText(label)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "CAD load failed",
                str(exc),
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _cad_message(self, message: str) -> None:
        self.cad_status.setText(message)
        self.setup_cad_status.setText(message)
        self.statusBar().showMessage(message, 8000)

    def _clear_visual_traces(self) -> None:
        self.viewer.clear_traces()
        self.movement_map.clear()

    # -------------------------------------------------------------- recipes
    def _sync_recipe_from_list(self) -> None:
        self.recipe.name = self.recipe_name.text().strip() or "Untitled recipe"
        steps: list[RecipeStep] = []
        for i in range(self.recipe_list.count()):
            item = self.recipe_list.item(i)
            data = item.data(QtCore.Qt.ItemDataRole.UserRole)
            steps.append(RecipeStep.from_dict(data))
        self.recipe.steps = steps

    def _refresh_recipe_list(self) -> None:
        self.recipe_list.clear()
        for step in self.recipe.steps:
            item = QtWidgets.QListWidgetItem(
                f"{step.label}\n{step.describe()}"
            )
            item.setData(
                QtCore.Qt.ItemDataRole.UserRole,
                step.to_dict(),
            )
            item.setToolTip(
                "Drag to reorder. Double-click or press Enter to edit. "
                "Ctrl+D duplicates; Delete removes."
            )
            self.recipe_list.addItem(item)
        self._update_recipe_preflight_view()

    def _append_recipe_step(self, step: RecipeStep) -> None:
        if self._recipe_running:
            QtWidgets.QMessageBox.information(
                self,
                "Script is running",
                "Stop the active script before adding modules.",
            )
            return
        self._sync_recipe_from_list()
        self.recipe.steps.append(step)
        self._refresh_recipe_list()
        self.recipe_list.setCurrentRow(self.recipe_list.count() - 1)

    def _default_step_for_module(self, key: str) -> RecipeStep:
        pose = Pose6D(
            *(self.script_pose_boxes[axis].value() for axis in "XYZUVW")
        )
        if key == "move_absolute":
            return RecipeStep.move_absolute(pose)
        if key == "move_incremental":
            return RecipeStep.move_incremental(pose)
        if key == "move_line_velocity":
            return RecipeStep.move_line_velocity(
                pose.x,
                pose.y,
                pose.z,
                self.script_line_velocity.value(),
            )
        if key == "write_line":
            delta = (pose.x, pose.y, pose.z)
            if not any(abs(v) > 0 for v in delta):
                # A zero-length write block is never what the operator means;
                # fall back to the familiar legacy writing line.
                delta = (self.quick_dx.value(), 0.0, 0.0)
            return RecipeStep.write_line(
                delta[0],
                delta[1],
                delta[2],
                self.script_line_velocity.value(),
            )
        if key == "pockels_open":
            return RecipeStep.pockels_cell(True)
        if key == "pockels_closed":
            return RecipeStep.pockels_cell(False)
        if key == "attenuator_set":
            return RecipeStep.attenuator_set(
                self.script_attenuator.value()
            )
        if key == "wait":
            return RecipeStep.wait(self.script_wait.value())
        raise ValueError(f"unknown recipe module: {key}")

    def _recipe_module_dropped(self, key: str, row: int) -> None:
        if self._recipe_running:
            QtWidgets.QMessageBox.information(
                self,
                "Script is running",
                "Stop the active script before editing its modules.",
            )
            return
        self._sync_recipe_from_list()
        try:
            step = self._default_step_for_module(key)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not add module",
                str(exc),
            )
            return
        row = max(0, min(int(row), len(self.recipe.steps)))
        self.recipe.steps.insert(row, step)
        self._refresh_recipe_list()
        self.recipe_list.setCurrentRow(row)

    def _recipe_edit_selected(self) -> None:
        if self._recipe_running:
            QtWidgets.QMessageBox.information(
                self,
                "Script is running",
                "Stop the active script before editing its modules.",
            )
            return
        self._sync_recipe_from_list()
        row = self.recipe_list.currentRow()
        if row < 0:
            return
        dialog = RecipeStepEditorDialog(
            self.recipe.steps[row],
            self.workspace_limits,
            self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.recipe.steps[row] = dialog.edited_step()
        self._refresh_recipe_list()
        self.recipe_list.setCurrentRow(row)

    def _recipe_add_move(self) -> None:
        pose = Pose6D(
            *(
                self.script_pose_boxes[axis].value()
                for axis in "XYZUVW"
            )
        )
        kind = self.script_move_kind.currentIndex()
        try:
            if kind == 0:
                step = RecipeStep.move_absolute(pose)
            elif kind == 1:
                step = RecipeStep.move_incremental(pose)
            elif kind == 2:
                step = RecipeStep.move_line_velocity(
                    pose.x,
                    pose.y,
                    pose.z,
                    self.script_line_velocity.value(),
                )
            else:
                step = self._default_step_for_module("write_line")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Could not add the module",
                str(exc),
            )
            return
        self._append_recipe_step(step)

    def _script_pose_from_live(self) -> None:
        pose = self._last_stage_snapshot.actual
        self._ensure_pose_widget_ranges_include(pose)
        for axis, value in zip("XYZUVW", pose.as_tuple()):
            self.script_pose_boxes[axis].setValue(value)

    def _recipe_add_attenuator(self) -> None:
        self._append_recipe_step(
            RecipeStep.attenuator_set(
                self.script_attenuator.value()
            )
        )

    def _recipe_add_wait(self) -> None:
        self._append_recipe_step(
            RecipeStep.wait(self.script_wait.value())
        )

    def _recipe_remove(self) -> None:
        if self._recipe_running:
            return
        self._sync_recipe_from_list()
        row = self.recipe_list.currentRow()
        if row < 0:
            return
        self.recipe.steps.pop(row)
        self._refresh_recipe_list()
        if self.recipe.steps:
            self.recipe_list.setCurrentRow(
                min(row, len(self.recipe.steps) - 1)
            )

    def _recipe_move_selected(self, delta: int) -> None:
        if self._recipe_running:
            return
        self._sync_recipe_from_list()
        row = self.recipe_list.currentRow()
        target = row + int(delta)
        if row < 0 or not 0 <= target < len(self.recipe.steps):
            return
        steps = self.recipe.steps
        steps[row], steps[target] = steps[target], steps[row]
        self._refresh_recipe_list()
        self.recipe_list.setCurrentRow(target)
        self.recipe_list.scrollToItem(
            self.recipe_list.item(target),
            QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible,
        )

    def _update_sequence_controls(self) -> None:
        if not hasattr(self, "sequence_buttons"):
            return
        count = self.recipe_list.count()
        row = self.recipe_list.currentRow()
        selected = 0 <= row < count
        up, down, edit, remove, duplicate = self.sequence_buttons
        editable = selected and not self._recipe_running
        up.setEnabled(editable and row > 0)
        down.setEnabled(editable and row < count - 1)
        for button in (edit, remove, duplicate):
            button.setEnabled(editable)
        self.sequence_hint.setVisible(count == 0 or not self._recipe_running)

    def _recipe_duplicate(self) -> None:
        if self._recipe_running:
            return
        self._sync_recipe_from_list()
        row = self.recipe_list.currentRow()
        if row < 0:
            return
        clone = RecipeStep.from_dict(
            self.recipe.steps[row].to_dict()
        )
        self.recipe.steps.insert(row + 1, clone)
        self._refresh_recipe_list()
        self.recipe_list.setCurrentRow(row + 1)

    def _recipe_clear(self) -> None:
        if self._recipe_running:
            return
        self._sync_recipe_from_list()
        if not self.recipe.steps and self.recipe_list.count() == 0:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear recipe?",
            "Remove every step from the current recipe?",
        )
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self.recipe.steps.clear()
            self._refresh_recipe_list()

    def _recipe_hardware_issues(self) -> list[tuple[str, str]]:
        self._sync_recipe_from_list()
        kinds = {step.kind for step in self.recipe.steps}
        issues: list[tuple[str, str]] = []

        uses_stage = bool(kinds & MOTION_KINDS)
        uses_pockels = bool(kinds & POCKELS_KINDS)
        uses_attenuator = StepKind.ATTENUATOR_SET in kinds

        if uses_stage and self.stage_mode.currentIndex() == 1:
            uses_general_motion = bool(
                kinds & {StepKind.MOVE_ABSOLUTE, StepKind.MOVE_INCREMENTAL}
            )
            if uses_general_motion and not self.workspace_limits.controller_verified:
                issues.append(
                    (
                        "error",
                        "Real absolute/relative motion requires a verified Cartesian envelope; "
                        "Line/Write-Line moves are preflighted natively by the HXP at runtime",
                    )
                )
            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                issues.append(
                    ("error", "Real HXP is selected but not connected")
                )

        if uses_pockels and self.laser_mode.currentIndex() == 1:
            if (
                self.real_laser is None
                or not self.real_laser.snapshot().connected
            ):
                issues.append(
                    (
                        "error",
                        "Real LX13/Pockels provider is selected but not armed",
                    )
                )

        if uses_attenuator and self.attenuator_mode.currentIndex() == 1:
            if not self.real_attenuator.snapshot().connected:
                issues.append(
                    (
                        "error",
                        "Real HXP attenuator is selected but not connected",
                    )
                )

        real_paths = []
        virtual_paths = []
        if uses_stage:
            (
                real_paths
                if self.stage_mode.currentIndex() == 1
                else virtual_paths
            ).append("stage")
        if uses_pockels:
            (
                real_paths
                if self.laser_mode.currentIndex() == 1
                else virtual_paths
            ).append("Pockels")
        if uses_attenuator:
            (
                real_paths
                if self.attenuator_mode.currentIndex() == 1
                else virtual_paths
            ).append("attenuator")

        if real_paths and virtual_paths:
            issues.append(
                (
                    "warning",
                    "Mixed execution providers: real "
                    + ", ".join(real_paths)
                    + " with virtual "
                    + ", ".join(virtual_paths),
                )
            )

        return issues

    def _recipe_workspace_issues(self):
        self._sync_recipe_from_list()
        if self.stage_mode.currentIndex() == 0:
            return validate_recipe_workspace(
                self.recipe,
                self._planned_stage_pose(),
                self.workspace_limits,
                self.kinematics,
                self.sample_calibration,
            )

        # REAL LAB deliberately does not use STEP-derived strut lengths. Static
        # preflight checks sample bounds and any operator-verified Cartesian
        # envelope. Each Line/Write-Line is then checked by the HXP's native
        # HexapodMoveIncrementalControlLimitGet immediately before execution.
        issues: list[RecipeWorkspaceIssue] = []
        current = self._planned_stage_pose()
        for index, step in enumerate(self.recipe.steps):
            try:
                target = None
                if step.kind == StepKind.MOVE_ABSOLUTE:
                    target = Pose6D.from_iterable(step.payload["pose"])
                    if self.workspace_limits.controller_verified:
                        self.workspace_limits.require_cartesian_pose(target)
                elif step.kind == StepKind.MOVE_INCREMENTAL:
                    target = current.plus(Pose6D.from_iterable(step.payload["delta"]))
                    if self.workspace_limits.controller_verified:
                        self.workspace_limits.require_cartesian_pose(target)
                elif step.kind in (StepKind.MOVE_LINE_VELOCITY, StepKind.WRITE_LINE):
                    dx, dy, dz = (float(v) for v in step.payload["delta_xyz_mm"])
                    target = current.plus(Pose6D(x=dx, y=dy, z=dz))
                if target is not None:
                    sample_issues = self.sample_calibration.path_violations(current, target)
                    if sample_issues:
                        issues.append(RecipeWorkspaceIssue(index, sample_issues[0]))
                    current = target
            except Exception as exc:
                issues.append(RecipeWorkspaceIssue(index, str(exc)))
        return issues

    def _script_uses_real_hardware(self) -> bool:
        self._sync_recipe_from_list()
        for step in self.recipe.steps:
            if (
                step.kind in MOTION_KINDS
                and self.stage_mode.currentIndex() == 1
            ):
                return True
            if (
                step.kind in POCKELS_KINDS
                and self.laser_mode.currentIndex() == 1
            ):
                return True
            if (
                step.kind == StepKind.ATTENUATOR_SET
                and self.attenuator_mode.currentIndex() == 1
            ):
                return True
        return False

    def _update_recipe_preflight_view(self) -> None:
        if not hasattr(self, "preflight_text"):
            return
        self._sync_recipe_from_list()
        issues = preflight_recipe(self.recipe)
        hardware = self._recipe_hardware_issues()
        workspace = self._recipe_workspace_issues()

        lines: list[str] = []
        if not issues and not hardware and not workspace:
            lines.append("PASS — recipe and selected providers are ready.")
        for issue in issues:
            prefix = issue.severity.upper()
            location = (
                f"step {issue.step_index + 1}: "
                if issue.step_index is not None
                else ""
            )
            lines.append(f"{prefix}: {location}{issue.message}")
        for severity, message in hardware:
            lines.append(f"{severity.upper()}: {message}")
        for issue in workspace:
            lines.append(
                f"ERROR: step {issue.step_index + 1}: {issue.message}"
            )

        self.preflight_text.setPlainText("\n".join(lines))

        error_count = sum(
            1 for issue in issues if issue.severity == "error"
        ) + sum(
            1 for severity, _ in hardware if severity == "error"
        ) + len(workspace)
        warning_count = sum(
            1 for issue in issues if issue.severity == "warning"
        ) + sum(1 for severity, _ in hardware if severity == "warning")
        if hasattr(self, "home_preflight_chip"):
            steps = len(self.recipe.steps)
            self.home_script_name.setText(
                f"{self.recipe.name}  \u2014  {steps} step(s)"
                if steps
                else "No script loaded"
            )
            if not steps:
                self.home_preflight_chip.setText("EMPTY")
                self._set_object_style(self.home_preflight_chip, "chipNeutral")
            elif error_count:
                self.home_preflight_chip.setText(
                    f"{error_count} PREFLIGHT ERROR(S)"
                )
                self._set_object_style(self.home_preflight_chip, "chipWarn")
            elif warning_count:
                self.home_preflight_chip.setText(
                    f"READY \u2022 {warning_count} WARNING(S)"
                )
                self._set_object_style(self.home_preflight_chip, "chipLive")
            else:
                self.home_preflight_chip.setText("PREFLIGHT PASS")
                self._set_object_style(self.home_preflight_chip, "chipSafe")
            self.home_preflight_chip.setToolTip(
                "\n".join(lines) if lines else "No preflight findings"
            )
        self._update_sequence_controls()
        self.execution_summary.setText(
            "Stage: "
            + ("REAL HXP" if self.stage_mode.currentIndex() else "VIRTUAL")
            + "\nPockels: "
            + (
                "REAL LX13"
                if self.laser_mode.currentIndex()
                else "VIRTUAL"
            )
            + "\nAttenuator: "
            + (
                "REAL / UNBOUND"
                if self.attenuator_mode.currentIndex()
                else "VIRTUAL"
            )
        )

    def _run_recipe(self) -> None:
        self._sync_recipe_from_list()
        if not self.recipe.steps:
            QtWidgets.QMessageBox.information(
                self,
                "Empty script",
                "Add at least one step before running the script.",
            )
            return
        issues = preflight_recipe(self.recipe)
        hw = self._recipe_hardware_issues()
        workspace = self._recipe_workspace_issues()
        errors = [
            issue
            for issue in issues
            if issue.severity == "error"
        ]
        errors.extend(
            message
            for severity, message in hw
            if severity == "error"
        )
        errors.extend(issue.message for issue in workspace)
        if errors:
            self._update_recipe_preflight_view()
            QtWidgets.QMessageBox.critical(
                self,
                "Script preflight failed",
                "Fix the preflight errors before running the script.",
            )
            return

        if (
            self._script_uses_real_hardware()
            and not self.real_script_arm.isChecked()
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Real script not armed",
                "This script will command real hardware. Check "
                "ARM REAL SCRIPT EXECUTION after reviewing preflight.",
            )
            return

        # Known safe beam state before the first recipe step.
        self._close_all_pockels()
        self._recipe_running = True
        self.recipe_list.setEnabled(False)
        self.module_library.setEnabled(False)
        self._recipe_index = 0
        self._recipe_step_issued = False
        self._recipe_write_block_open = False
        self._recipe_started_monotonic = time.monotonic()
        self._wait_until = 0.0
        self._clear_visual_traces()
        self.recipe_progress.setText(
            f"Running {self.recipe.name}…"
        )

    def _stop_recipe(self) -> None:
        if not self._recipe_running:
            return
        self._recipe_running = False
        self._recipe_write_block_open = False
        self.recipe_list.setEnabled(True)
        self.module_library.setEnabled(True)
        self._close_all_pockels()
        try:
            stage = self._stage_provider()
            if stage.is_busy():
                stage.abort()
        except Exception:
            pass
        self.recipe_progress.setText(
            "Stopped — close-beam command issued"
        )

    def _advance_recipe(self) -> None:
        self._recipe_index += 1
        self._recipe_step_issued = False
        self._wait_until = 0.0
        if self._recipe_index >= len(self.recipe.steps):
            self._recipe_running = False
            self._recipe_write_block_open = False
            self.recipe_list.setEnabled(True)
            self.module_library.setEnabled(True)
            self._close_all_pockels()
            elapsed = time.monotonic() - self._recipe_started_monotonic
            self.recipe_progress.setText(
                f"Complete in {elapsed:.1f} s — Pockels CLOSED"
            )

    def _recipe_tick(self) -> None:
        if not self._recipe_running:
            return
        if self._recipe_index >= len(self.recipe.steps):
            self._advance_recipe()
            return

        step = self.recipe.steps[self._recipe_index]
        self.recipe_progress.setText(
            f"Step {self._recipe_index + 1}/{len(self.recipe.steps)}  •  "
            f"{step.describe()}"
        )

        try:
            if step.kind == StepKind.MOVE_ABSOLUTE:
                stage = self._stage_provider()
                if not self._recipe_step_issued:
                    target = Pose6D.from_iterable(step.payload["pose"])
                    self._require_requested_motion(target)
                    stage.move_absolute(target)
                    self._recipe_step_issued = True
                elif not stage.is_busy():
                    self._advance_recipe()

            elif step.kind == StepKind.MOVE_INCREMENTAL:
                stage = self._stage_provider()
                if not self._recipe_step_issued:
                    delta = Pose6D.from_iterable(step.payload["delta"])
                    self._require_requested_motion(
                        self._planned_stage_pose().plus(delta)
                    )
                    stage.move_incremental(delta)
                    self._recipe_step_issued = True
                elif not stage.is_busy():
                    self._advance_recipe()

            elif step.kind == StepKind.MOVE_LINE_VELOCITY:
                stage = self._stage_provider()
                if not self._recipe_step_issued:
                    dx, dy, dz = (
                        float(v)
                        for v in step.payload["delta_xyz_mm"]
                    )
                    velocity = float(step.payload["velocity_mm_s"])
                    self._require_requested_motion(
                        self._planned_stage_pose().plus(
                            Pose6D(x=dx, y=dy, z=dz)
                        ),
                        velocity_mm_s=velocity,
                    )
                    stage.move_line_incremental_with_target_velocity(
                        dx,
                        dy,
                        dz,
                        velocity,
                    )
                    self._recipe_step_issued = True
                elif not stage.is_busy():
                    self._advance_recipe()

            elif step.kind == StepKind.WRITE_LINE:
                stage = self._stage_provider()
                if not self._recipe_step_issued:
                    dx, dy, dz = (
                        float(v)
                        for v in step.payload["delta_xyz_mm"]
                    )
                    velocity = float(step.payload["velocity_mm_s"])
                    self._require_requested_motion(
                        self._planned_stage_pose().plus(
                            Pose6D(x=dx, y=dy, z=dz)
                        ),
                        velocity_mm_s=velocity,
                    )
                    # Open first, then issue the move; the tick that sees the
                    # move finish closes the cell again.
                    self._set_pockels(True, script=True)
                    self._recipe_write_block_open = True
                    try:
                        stage.move_line_incremental_with_target_velocity(
                            dx,
                            dy,
                            dz,
                            velocity,
                        )
                    except Exception:
                        self._close_all_pockels()
                        self._recipe_write_block_open = False
                        raise
                    self._recipe_step_issued = True
                elif not stage.is_busy():
                    self._set_pockels(False, script=True)
                    self._recipe_write_block_open = False
                    self._advance_recipe()

            elif step.kind == StepKind.POCKELS_CELL:
                if not self._recipe_step_issued:
                    self._set_pockels(
                        bool(step.payload.get("open")),
                        script=True,
                    )
                    self._recipe_step_issued = True
                    self._advance_recipe()

            elif step.kind == StepKind.ATTENUATOR_SET:
                if not self._recipe_step_issued:
                    self._set_attenuator(
                        float(step.payload["transmission_percent"]),
                        script=True,
                    )
                    self._recipe_step_issued = True
                    self._advance_recipe()

            elif step.kind == StepKind.WAIT:
                if not self._recipe_step_issued:
                    self._wait_until = (
                        time.monotonic()
                        + float(step.payload.get("seconds", 0.0))
                    )
                    self._recipe_step_issued = True
                elif time.monotonic() >= self._wait_until:
                    self._advance_recipe()

        except Exception as exc:
            self._recipe_running = False
            self._recipe_write_block_open = False
            self.recipe_list.setEnabled(True)
            self.module_library.setEnabled(True)
            self._close_all_pockels()
            try:
                stage = self._stage_provider()
                if stage.is_busy():
                    stage.abort()
            except Exception as abort_exc:
                self.log_internal_error(
                    "recipe fail-safe abort",
                    abort_exc,
                )
            self.recipe_progress.setText(
                f"FAILED at step {self._recipe_index + 1}: {exc}"
            )
            QtWidgets.QMessageBox.critical(
                self,
                "Script failed",
                str(exc),
            )

    def _save_recipe(self) -> None:
        self._sync_recipe_from_list()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save recipe",
            f"{self.recipe.name.replace(' ', '_')}.json",
            "JSON (*.json)",
        )
        if path:
            self.recipe.save(path)

    def _load_recipe(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load recipe",
            "",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.recipe = Recipe.load(path)
            self.recipe_name.setText(self.recipe.name)
            self._refresh_recipe_list()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Recipe load failed",
                str(exc),
            )

    # ----------------------------------------------------------- diagnostics
    def _diag(self, message: str) -> None:
        if not hasattr(self, "diag_log"):
            return
        stamp = time.strftime("%H:%M:%S")
        self.diag_log.appendPlainText(f"[{stamp}] {message}")

    def _diagnostic_firmware(self) -> None:
        try:
            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                raise ConnectionError("Real HXP is not connected")
            version = self.real_stage.client.firmware_version()
            self._diag(f"HXP firmware: {version}")
        except Exception as exc:
            self._diag(f"Firmware read failed: {exc}")

    def _diagnostic_pose(self) -> None:
        try:
            if (
                self.real_stage is None
                or not self.real_stage.client.connected
            ):
                raise ConnectionError("Real HXP is not connected")
            snap = self.real_stage.snapshot()
            values = ", ".join(
                f"{axis}={value:.5f}"
                for axis, value in zip(
                    "XYZUVW",
                    snap.actual.as_tuple(),
                )
            )
            self._diag(
                f"Current pose: {values}; status={snap.status_text}"
            )
        except Exception as exc:
            self._diag(f"Pose poll failed: {exc}")

    # --------------------------------------------------------------- polling
    def _poll_stage(self, now: float, dt: float) -> None:
        if self.stage_mode.currentIndex() == 0:
            if (
                self._poll_future is not None
                and self._poll_future.done()
            ):
                self._poll_future = None
            if self.virtual_stage.snapshot().connected:
                self.virtual_stage.tick(
                    dt * self.mock_time_scale.value()
                )
                self._last_stage_snapshot = (
                    self.virtual_stage.snapshot()
                )
            else:
                self._last_stage_snapshot = HexapodSnapshot(
                    timestamp_s=time.time(),
                    actual=self._last_stage_snapshot.actual,
                    state=MotionState.DISCONNECTED,
                    connected=False,
                    provider="virtual",
                    status_text="DISCONNECTED",
                )
            return

        if (
            self.real_stage is None
            or not self.real_stage.client.connected
        ):
            self._last_stage_snapshot = HexapodSnapshot(
                timestamp_s=time.time(),
                actual=self._last_stage_snapshot.actual,
                state=MotionState.DISCONNECTED,
                connected=False,
                provider="hxp-real",
                status_text="NOT CONNECTED",
            )
            return

        if self._poll_future is not None and self._poll_future.done():
            try:
                snap = self._poll_future.result()
                self._last_stage_snapshot = snap
                self._real_poll_failures = 0
                self._ensure_pose_widget_ranges_include(snap.actual)
                if (
                    snap.state == MotionState.FAULT
                    and "last GUI motion failed:" in snap.status_text
                ):
                    self.statusBar().showMessage(
                        snap.status_text,
                        10000,
                    )
                    self._abort_all()
            except Exception as exc:
                self._real_poll_failures += 1
                connected = bool(
                    self.real_stage is not None
                    and self.real_stage.client.connected
                )
                self._last_stage_snapshot = HexapodSnapshot(
                    timestamp_s=time.time(),
                    actual=self._last_stage_snapshot.actual,
                    setpoint=self._last_stage_snapshot.setpoint,
                    target=self._last_stage_snapshot.target,
                    state=(
                        MotionState.FAULT
                        if connected
                        else MotionState.DISCONNECTED
                    ),
                    connected=connected,
                    provider="hxp-real",
                    status_text=f"POLL ERROR: {exc}",
                )
                self.statusBar().showMessage(
                    f"HXP polling error: {exc}",
                    7000,
                )
            self._poll_future = None

        if (
            self._poll_future is None
            and now - self._last_real_poll >= 0.15
            and self.real_stage is not None
            and self.real_stage.client.connected
        ):
            self._last_real_poll = now
            self._poll_future = self._poll_pool.submit(
                self.real_stage.snapshot
            )

    def _set_object_style(
        self,
        widget: QtWidgets.QWidget,
        object_name: str,
    ) -> None:
        if widget.objectName() == object_name:
            return
        widget.setObjectName(object_name)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _update_readouts(self) -> None:
        snap = self._last_stage_snapshot

        if self.lab_mode.currentIndex() == 0:
            self.mode_chip.setText("MOCK • SAFE")
            self._set_object_style(self.mode_chip, "chipSafe")
        else:
            real_laser_connected = (
                self.real_laser is not None
                and self.real_laser.snapshot().connected
            )
            if real_laser_connected:
                self.mode_chip.setText("REAL • POCKELS ARMED")
                self._set_object_style(self.mode_chip, "chipLive")
            elif self.wiring_verified.isChecked():
                self.mode_chip.setText("REAL • MAPPING VERIFIED")
                self._set_object_style(self.mode_chip, "chipWarn")
            else:
                self.mode_chip.setText("REAL • DISARMED")
                self._set_object_style(self.mode_chip, "chipWarn")

        target_pose = snap.target or snap.setpoint or snap.actual
        for axis, actual_value, target_value in zip(
            "XYZUVW",
            snap.actual.as_tuple(),
            target_pose.as_tuple(),
        ):
            unit = " mm" if axis in "XYZ" else " °"
            self.state_labels[axis].setText(
                f"{actual_value:+.4f}{unit}"
            )
            self.target_labels[axis].setText(
                f"{target_value:+.4f}{unit}"
            )
            self.error_labels[axis].setText(
                f"{target_value - actual_value:+.4f}"
            )

        self.motion_status.setText(
            f"{snap.state.value.upper()}  •  "
            f"{snap.status_text or snap.provider}"
        )
        self.manual_stage_status.setText(
            (
                "Connected"
                if snap.connected
                else "NOT CONNECTED"
            )
            + f"  •  {snap.provider}"
        )

        lengths = self.kinematics.leg_lengths_mm(snap.actual)
        home = self.kinematics.leg_lengths_mm(Pose6D())
        for label, length, reference in zip(
            self.leg_labels,
            lengths,
            home,
        ):
            label.setText(
                f"{length:8.3f} mm   Δ {length-reference:+.3f}"
            )

        laser = self._laser_snapshot()
        if (
            self._beam_close_failed
            and self.lab_mode.currentIndex() == 1
            and self.real_laser is not None
        ):
            self.manual_beam_status.setText(
                "BEAM STATE UNKNOWN • CLOSE COMMAND FAILED"
            )
            self._set_object_style(
                self.manual_beam_status,
                "laserOn",
            )
            self.beam_chip.setText("BEAM STATE UNKNOWN")
            self._set_object_style(
                self.beam_chip,
                "chipWarn",
            )
        elif laser.pockels_open:
            self.manual_beam_status.setText(
                "LASER ON • POCKELS OPEN"
                + (
                    ""
                    if laser.readback_known
                    else " • COMMAND STATE ONLY"
                )
            )
            self._set_object_style(
                self.manual_beam_status,
                "laserOn",
            )
            self.beam_chip.setText("BEAM OPEN")
            self._set_object_style(
                self.beam_chip,
                "chipLive",
            )
        else:
            suffix = (
                ""
                if laser.connected
                else " • PROVIDER NOT CONNECTED"
            )
            self.manual_beam_status.setText(
                "LASER OFF • POCKELS CLOSED" + suffix
            )
            self._set_object_style(
                self.manual_beam_status,
                "laserOff",
            )
            self.beam_chip.setText(
                "BEAM CLOSED"
                if laser.connected
                else "POCKELS NOT CONNECTED"
            )
            self._set_object_style(
                self.beam_chip,
                "chipSafe" if laser.connected else "chipWarn",
            )

        attenuator = self._attenuator_snapshot()
        if attenuator.connected:
            self.attenuator_status.setText(
                "Transmission setpoint: "
                f"{attenuator.transmission_percent:.1f} %"
                + (
                    ""
                    if attenuator.readback_known
                    else " • command state only"
                )
            )
            self.attenuator_chip.setText(
                f"ATT {attenuator.transmission_percent:.1f} %"
            )
            self._set_object_style(
                self.attenuator_chip,
                "chipNeutral",
            )
        else:
            self.attenuator_status.setText(
                "Real attenuator not bound to a hardware driver"
            )
            self.attenuator_chip.setText("ATT UNBOUND")
            self._set_object_style(
                self.attenuator_chip,
                "chipWarn",
            )

        real_mode = self.stage_mode.currentIndex() == 1
        readback_fresh = (
            not real_mode
            or (
                snap.connected
                and (time.time() - snap.timestamp_s) <= 1.0
            )
        )
        frames_ok = (
            not real_mode
            or self._real_frames_match_profile
        )

        if (
            real_mode
            and laser.pockels_open
            and not self._beam_close_failed
            and (
                not snap.connected
                or not readback_fresh
                or snap.state == MotionState.FAULT
                or not frames_ok
            )
        ):
            self._abort_all()
            laser = self._laser_snapshot()

        stage_ready = bool(
            snap.connected
            and snap.state == MotionState.IDLE
            and readback_fresh
            and frames_ok
        )

        if stage_ready:
            self._enforce_sample_bounds_are_reachable()
        self._update_sample_readouts(snap)
        self._update_script_readouts()

        if snap.state == MotionState.MOVING and readback_fresh:
            self.stage_chip.setText("STAGE MOVING")
            self._set_object_style(
                self.stage_chip,
                "chipLive",
            )
        elif not snap.connected:
            self.stage_chip.setText("STAGE NOT CONNECTED")
            self._set_object_style(
                self.stage_chip,
                "chipWarn",
            )
        elif not readback_fresh:
            self.stage_chip.setText("STAGE READBACK STALE")
            self._set_object_style(
                self.stage_chip,
                "chipWarn",
            )
        elif real_mode and not frames_ok:
            self.stage_chip.setText("STAGE FRAME MISMATCH")
            self._set_object_style(
                self.stage_chip,
                "chipWarn",
            )
        elif snap.state == MotionState.IDLE:
            self.stage_chip.setText("STAGE READY")
            self._set_object_style(
                self.stage_chip,
                "chipSafe",
            )
        else:
            self.stage_chip.setText("STAGE NOT READY")
            self._set_object_style(
                self.stage_chip,
                "chipWarn",
            )

        self.viewer.update_state(
            snap.actual,
            laser.pockels_open,
        )
        self.movement_map.update_state(
            snap.actual,
            laser.pockels_open,
            self.viewer.beam_hit_sample_xy,
        )

        operator_free = (
            not self._recipe_running
            and not self._manual_write_line_active
        )
        motion_ready = stage_ready and operator_free
        self.move_abs_btn.setEnabled(motion_ready)
        # HOME is state-aware itself: permit it on a fresh connected controller
        # so an explicitly Not Referenced group can be homed.
        self.home_btn.setEnabled(
            snap.connected
            and readback_fresh
            and operator_free
        )
        for button in self.jog_buttons:
            button.setEnabled(motion_ready)

        laser_ready = laser.connected and not self._beam_close_failed
        real_manual_ok = (
            self.laser_mode.currentIndex() == 0
            or self.manual_beam_arm.isChecked()
        )
        self.laser_on_btn.setEnabled(
            laser_ready and real_manual_ok and operator_free
        )
        # Closing the process beam remains available even while a script runs.
        self.laser_off_btn.setEnabled(laser.connected)

        quick_stage_ready = (
            stage_ready
            and not self._recipe_running
            and not self._manual_write_line_active
        )
        self.quick_move_btn.setEnabled(quick_stage_ready)
        self.quick_return_btn.setEnabled(quick_stage_ready)
        self.quick_write_btn.setEnabled(
            quick_stage_ready
            and laser_ready
            and real_manual_ok
        )

        att_ready = attenuator.connected
        self.set_attenuator_btn.setEnabled(
            att_ready and operator_free
        )
        self.capture_sample_btn.setEnabled(
            stage_ready and operator_free
        )

        # Do not allow provider switching/reconnection while motion or a script
        # is active, or while the most recent real readback is not trustworthy.
        provider_switch_safe = (
            not self._recipe_running
            and not self._manual_write_line_active
            and snap.state != MotionState.MOVING
        )
        self.lab_mode.setEnabled(provider_switch_safe)
        self.legacy_profile_combo.setEnabled(
            provider_switch_safe and self.lab_mode.currentIndex() == 0
        )
        self.mock_time_scale.setEnabled(
            self.lab_mode.currentIndex() == 0
        )
        self.stage_mode.setEnabled(False)
        self.laser_mode.setEnabled(False)
        self.attenuator_mode.setEnabled(False)
        self.connect_stage_btn.setEnabled(provider_switch_safe)
        self.disconnect_stage_btn.setEnabled(provider_switch_safe)
        self.real_script_arm.setEnabled(
            provider_switch_safe and stage_ready
        )

    def _update_sample_readouts(self, snap: HexapodSnapshot) -> None:
        """Keep the sample chip and clearance text in step with the stage."""

        calibration = self.sample_calibration
        bound = calibration.enforce_xy or calibration.enforce_z
        if bound:
            clearance = calibration.clearance_mm(snap.actual)
            if clearance is None:
                text = "SAMPLE BOUND"
            else:
                text = f"SAMPLE BOUND {clearance - calibration.margin_mm:+.3f} mm"
            self.sample_chip.setText(text)
            self._set_object_style(self.sample_chip, "chipSafe")
        elif calibration.is_usable:
            self.sample_chip.setText(
                f"SAMPLE MAPPED ({calibration.point_count})"
            )
            self._set_object_style(self.sample_chip, "chipLive")
        else:
            self.sample_chip.setText("SAMPLE UNCALIBRATED")
            self._set_object_style(self.sample_chip, "chipNeutral")
        self.sample_chip.setToolTip(calibration.summary())

    def _update_script_readouts(self) -> None:
        """Mirror script state onto the control screen and the run panel."""

        total = max(1, len(self.recipe.steps))
        if self._recipe_running:
            done = min(self._recipe_index, len(self.recipe.steps))
            percent = int(round(100.0 * done / total))
            elapsed = time.monotonic() - self._recipe_started_monotonic
            text = (
                f"step {min(done + 1, len(self.recipe.steps))}"
                f"/{len(self.recipe.steps)}  \u2022  {elapsed:.0f} s"
            )
            if self._recipe_write_block_open:
                text = "WRITING  \u2022  " + text
        else:
            percent = 0
            text = "idle" if self.recipe.steps else "no script"
        for bar in (self.home_script_progress, self.script_progress_bar):
            if bar.value() != percent:
                bar.setValue(percent)
            if bar.format() != text:
                bar.setFormat(text)

        runnable = bool(self.recipe.steps) and not self._recipe_running
        self.home_run_btn.setEnabled(runnable)
        self.run_recipe_btn.setEnabled(runnable)
        self.home_stop_btn.setEnabled(self._recipe_running)
        self.stop_recipe_btn.setEnabled(self._recipe_running)
        self.home_script_name.setText(
            self.recipe_progress.text()
            if self._recipe_running
            else (
                f"{self.recipe.name}  \u2014  {len(self.recipe.steps)} step(s)"
                if self.recipe.steps
                else "No script loaded"
            )
        )

    def log_internal_error(self, where: str, exc: BaseException) -> None:
        """Record an unexpected failure without losing the traceback.

        PySide6 terminates the process when an exception escapes a slot or a
        virtual, and the Windows launcher has no console, so an unguarded bug
        in the 20 Hz tick would make the application vanish silently.
        """

        text = (
            f"\n--- {where} failed {datetime.datetime.now().isoformat(timespec='seconds')} ---\n"
            + "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        )
        try:
            with CRASH_LOG.open("a", encoding="utf-8") as handle:
                handle.write(text)
        except Exception:
            pass
        print(text, file=sys.stderr)

    def _tick(self) -> None:
        try:
            self._tick_once()
        except Exception as exc:
            self._tick_failures += 1
            self.log_internal_error(
                f"tick (failure {self._tick_failures})",
                exc,
            )
            self.statusBar().showMessage(
                f"Internal error in the update loop: {exc} "
                f"(see {CRASH_LOG.name})",
                10000,
            )
            # A loop that cannot update its readouts must not keep a beam
            # open on the strength of a stale display.
            if self._tick_failures >= 5:
                self._tick_failures = 0
                try:
                    self._abort_all()
                except Exception as abort_exc:
                    self.log_internal_error("fail-safe abort", abort_exc)
                QtWidgets.QMessageBox.critical(
                    self,
                    "Update loop is failing",
                    "The controller could not update its state five times in "
                    "a row, so motion was aborted and beam closure requested.\n\n"
                    f"The traceback is in {CRASH_LOG}.",
                )
        else:
            self._tick_failures = 0

    def _tick_once(self) -> None:
        now = time.monotonic()
        dt = min(
            0.25,
            max(0.0, now - self._last_tick_monotonic),
        )
        self._last_tick_monotonic = now
        self._poll_stage(now, dt)
        self._manual_write_line_tick()
        self._recipe_tick()
        self._commission_selftest_tick()
        self._update_readouts()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.timer.stop()
        try:
            self._close_all_pockels()
            if self.real_stage is not None:
                try:
                    self.real_stage.disconnect()
                except Exception:
                    pass
            self.virtual_stage.disconnect()
            self.virtual_laser.disconnect()
            self.virtual_attenuator.disconnect()
            try:
                self.real_attenuator.disconnect()
            except Exception:
                pass
        finally:
            self._poll_pool.shutdown(
                wait=False,
                cancel_futures=True,
            )
            # pyvistaqt needs its render window released explicitly; leaving it
            # to garbage collection can fault during interpreter shutdown.
            try:
                self.viewer.close()
            except Exception as exc:
                self.log_internal_error("viewer shutdown", exc)
        super().closeEvent(event)
