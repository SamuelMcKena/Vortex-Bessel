"""Sequence-list drag behaviour: edge autoscroll and a visible drop row."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyvista")

from hexapod_lab.main_window import RECIPE_MODULE_MIME, RecipeList  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def listing(app):
    widget = RecipeList()
    widget.resize(320, 240)
    for index in range(40):
        item = QtWidgets.QListWidgetItem(f"Block {index}\nstep {index}")
        item.setData(
            QtCore.Qt.ItemDataRole.UserRole,
            {"kind": "wait", "label": "Wait", "payload": {"seconds": 1.0}},
        )
        widget.addItem(item)
    widget.show()
    app.processEvents()
    return widget


# A drag event only borrows its mime data, so it must outlive the event.
_LIVE_MIME: list = []


def module_mime():
    mime = QtCore.QMimeData()
    mime.setData(RECIPE_MODULE_MIME, b"write_line")
    _LIVE_MIME.append(mime)
    return mime


def module_drag(x, y):
    return QtGui.QDragMoveEvent(
        QtCore.QPoint(int(x), int(y)),
        QtCore.Qt.DropAction.CopyAction,
        module_mime(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )


def test_dragging_to_the_bottom_edge_scrolls_the_list(listing, app):
    bar = listing.verticalScrollBar()
    assert bar.maximum() > 0, "list is not scrollable; test is meaningless"
    bar.setValue(0)

    height = listing.viewport().height()
    listing.dragMoveEvent(module_drag(60, height - 4))
    assert listing._scroll_timer.isActive(), "no autoscroll near the bottom edge"
    assert listing._scroll_speed > 0

    for _ in range(12):
        listing._auto_scroll_step()
    assert bar.value() > 0, "the view did not scroll while dragging"


def test_dragging_to_the_top_edge_scrolls_back(listing):
    bar = listing.verticalScrollBar()
    bar.setValue(bar.maximum())
    listing.dragMoveEvent(module_drag(60, 3))
    assert listing._scroll_speed < 0
    for _ in range(12):
        listing._auto_scroll_step()
    assert bar.value() < bar.maximum()


def test_no_autoscroll_in_the_middle_of_the_list(listing):
    height = listing.viewport().height()
    listing.dragMoveEvent(module_drag(60, height // 2))
    assert listing._scroll_speed == 0
    assert not listing._scroll_timer.isActive()


def test_autoscroll_stops_when_the_drag_leaves(listing):
    height = listing.viewport().height()
    listing.dragMoveEvent(module_drag(60, height - 4))
    assert listing._scroll_timer.isActive()
    listing.dragLeaveEvent(QtGui.QDragLeaveEvent())
    assert not listing._scroll_timer.isActive()
    assert listing._drop_row is None


def test_autoscroll_stops_at_the_end_of_the_range(listing):
    bar = listing.verticalScrollBar()
    bar.setValue(bar.maximum())
    listing._scroll_speed = 20
    listing._scroll_timer.start()
    listing._auto_scroll_step()
    assert not listing._scroll_timer.isActive()


def test_drop_row_tracks_the_pointer(listing):
    listing.verticalScrollBar().setValue(0)
    listing.dragMoveEvent(module_drag(60, 2))
    assert listing._drop_row == 0

    far = listing._row_at(QtCore.QPoint(60, listing.viewport().height() - 2))
    assert far > 0


def test_a_module_drag_is_accepted(listing):
    event = module_drag(60, 40)
    listing.dragMoveEvent(event)
    assert event.isAccepted()
    assert event.dropAction() == QtCore.Qt.DropAction.CopyAction


def test_module_drop_reports_the_insert_row(listing):
    seen = []
    listing.moduleDropped.connect(lambda key, row: seen.append((key, row)))
    drop = QtGui.QDropEvent(
        QtCore.QPointF(60, 2),
        QtCore.Qt.DropAction.CopyAction,
        module_mime(),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    listing.verticalScrollBar().setValue(0)
    listing.dropEvent(drop)
    assert seen == [("write_line", 0)]
    assert not listing._scroll_timer.isActive()


def test_alt_arrow_requests_a_move(listing):
    seen = []
    listing.moveRequested.connect(seen.append)
    listing.setCurrentRow(3)
    for key, expected in (
        (QtCore.Qt.Key.Key_Up, -1),
        (QtCore.Qt.Key.Key_Down, 1),
    ):
        listing.keyPressEvent(
            QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                key,
                QtCore.Qt.KeyboardModifier.AltModifier,
            )
        )
        assert seen[-1] == expected
