"""Stable Qt controls for scrollable laboratory parameter pages."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QWidget


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """A normal spin box whose mouse wheel is reserved for page scrolling."""

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Integer spin box that ignores wheel changes but keeps buttons/keys."""

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()


class NoWheelComboBox(QComboBox):
    """Combo box that cannot cycle accidentally while a page is scrolled."""

    def wheelEvent(self, event):  # noqa: N802 - Qt API
        event.ignore()


@contextmanager
def blocked_signals(widgets: Iterable[QWidget]) -> Iterator[None]:
    """Block model-to-view update signals for the lifetime of the context."""

    blockers = [QSignalBlocker(widget) for widget in widgets]
    try:
        yield
    finally:
        # Keeping the objects alive until this point is what keeps signals blocked.
        blockers.clear()
