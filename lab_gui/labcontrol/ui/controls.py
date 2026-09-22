"""Stable Qt controls for scrollable laboratory parameter pages."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator

from PySide6.QtCore import QObject, QSignalBlocker, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QWidget,
)


def make_shrinkable(root: QWidget) -> None:
    """Let a card give back width instead of widening the whole page.

    Text inputs, spin boxes and combos advertise a minimum wide enough for their
    longest content, and a wrapped label still advertises its longest line, so a
    card full of them demands far more width than it needs.  Compressing them
    costs no function and is what keeps a page off a horizontal scrollbar.
    """

    for kind in (QLineEdit, QComboBox, QAbstractSpinBox):
        for child in root.findChildren(kind):
            child.setMinimumWidth(72)
    for combo in root.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
    for form in root.findChildren(QFormLayout):
        # Narrow columns put the label above its field instead of widening the page.
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    for tabs in root.findChildren(QTabWidget):
        tabs.setUsesScrollButtons(True)
    for label in root.findChildren(QLabel):
        if label.wordWrap():
            # An explicit minimum is only a floor; a wrapped label still
            # advertises a hint wide enough for its longest line, so the
            # horizontal hint has to be ignored for it to actually reflow.
            label.setMinimumWidth(120)
            # Modify the existing policy: building a fresh one drops the
            # height-for-width flag, so a wrapped label is given one line of
            # height and its neighbours are drawn on top of it.
            policy = label.sizePolicy()
            policy.setHorizontalPolicy(QSizePolicy.Ignored)
            policy.setHeightForWidth(True)
            label.setSizePolicy(policy)


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


class GuiThreadDispatcher(QObject):
    """Run a named method of its owner on the GUI thread, whatever thread asks.

    PySide6 6.11 delivers a queued connection in the *emitting* thread when the
    target slot is overridden in a subclass.  This window is a chain of
    subclasses, so handlers such as ``_virtual_task_finished`` and
    ``_live_stopped`` silently ran on worker threads and touched widgets there,
    which corrupts the heap and crashes the application intermittently.

    This object's own slot is never overridden, so its queued connection is
    honoured; the handler itself is looked up by name at delivery time, so
    subclass overrides still apply.
    """

    _deliver = Signal(str, object)

    def __init__(self, owner: QObject):
        super().__init__(owner)
        self._owner = owner
        self._deliver.connect(self._run, Qt.QueuedConnection)

    @Slot(str, object)
    def _run(self, name: str, args: object) -> None:
        getattr(self._owner, name)(*args)

    def forward(self, name: str):
        """A callable to connect with ``Qt.DirectConnection`` from any thread."""

        def relay(*args) -> None:
            if QThread.currentThread() is self.thread():
                getattr(self._owner, name)(*args)
            else:
                self._deliver.emit(name, args)

        return relay
