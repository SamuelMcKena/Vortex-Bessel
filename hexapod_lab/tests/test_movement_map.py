"""Path-map trace segmentation: beam-off travel is never drawn as written."""

import os
from dataclasses import replace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from hexapod_lab.movement_map import MovementMap2D  # noqa: E402
from hexapod_lab.sample import SampleCalibration, SamplePoint  # noqa: E402
from hexapod_lab.types import Pose6D  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def widget(app):
    widget = MovementMap2D()
    widget.resize(600, 320)
    widget.set_mode("hxp")
    return widget


def drive(widget, points):
    """Feed (x, laser_on) pairs as if the stage were moving along X."""
    for x, laser_on in points:
        widget.update_state(Pose6D(x=float(x)), laser_on, None)


def test_two_written_lines_stay_separate(widget):
    drive(widget, [(0.0, False)])
    drive(widget, [(x / 10.0, True) for x in range(0, 21)])
    drive(widget, [(2.0 + x / 10.0, False) for x in range(0, 21)])
    drive(widget, [(4.0 + x / 10.0, True) for x in range(0, 21)])

    assert widget.written_run_count() == 2
    # 2 mm written twice; the 2 mm repositioning move is not counted.
    assert widget.written_length_mm() == pytest.approx(4.0, abs=0.05)


def test_written_runs_are_continuous_with_the_travel_that_precedes_them(widget):
    drive(widget, [(0.0, False), (1.0, False)])
    drive(widget, [(2.0, True)])
    segments = widget._segments()
    assert [segment.laser_on for segment in segments] == [False, True]
    # The written run starts exactly where the beam switched on.
    assert segments[1].points[0] == segments[0].points[-1]


def test_beam_state_change_without_motion_still_splits_the_trace(widget):
    drive(widget, [(0.0, False), (0.0, True), (0.0, False)])
    assert [segment.laser_on for segment in widget._segments()] == [
        False,
        True,
        False,
    ]


def test_clear_resets_every_dataset(widget):
    drive(widget, [(0.0, True), (1.0, True)])
    widget.clear()
    assert widget.written_run_count() == 0
    assert widget.written_length_mm() == 0.0
    # Only the live carriage position remains, so the view collapses to it.
    assert widget.data_bounds_mm() == (0.0, 0.0, 0.0, 0.0)
    assert widget.suggested_view()[0] > 0.0


def test_suggested_view_frames_the_recorded_path(widget):
    drive(widget, [(-3.0, False), (3.0, True)])
    span, cx, cy = widget.suggested_view()
    assert span >= 6.0
    assert cx == pytest.approx(0.0)
    assert cy == pytest.approx(0.0)


def test_sample_outline_is_included_in_the_fit(widget):
    calibration = SampleCalibration(
        points=(
            SamplePoint("a", Pose6D(x=-10.0, y=-10.0)),
            SamplePoint("b", Pose6D(x=10.0, y=10.0)),
        )
    )
    widget.set_sample_calibration(calibration)
    drive(widget, [(0.0, False)])
    span, _, _ = widget.suggested_view()
    assert span >= 20.0


def test_sample_mode_uses_the_recorded_beam_footprint(widget):
    calibration = SampleCalibration(
        points=(
            SamplePoint("a", Pose6D(x=-1.0, y=-1.0), (1.0, 1.0)),
            SamplePoint("b", Pose6D(x=1.0, y=1.0), (-1.0, -1.0)),
        )
    )
    widget.set_sample_calibration(calibration)
    widget.set_mode("sample")
    assert widget._region(safe=False) is not None
    widget.set_mode("hxp")
    assert widget._region(safe=False) is not None


def test_painting_does_not_flood_the_plot_background(widget):
    """A stale fill brush once painted the whole plot light grey."""

    calibration = SampleCalibration(
        points=(
            SamplePoint("a", Pose6D(x=-2.0, y=-2.0)),
            SamplePoint("b", Pose6D(x=2.0, y=2.0)),
        ),
        margin_mm=0.2,
    )
    widget.set_sample_calibration(replace(calibration, enforce_xy=True))
    drive(widget, [(-1.0, False), (0.0, True), (1.0, False)])

    pixmap = widget.grab()
    image = pixmap.toImage()
    centre = QtGui.QColor(image.pixel(image.width() // 2, image.height() - 60))
    assert centre.lightness() < 90, centre.name()


def test_mode_must_be_known(widget):
    with pytest.raises(ValueError):
        widget.set_mode("elsewhere")
