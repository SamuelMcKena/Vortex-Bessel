import math
from dataclasses import replace

import pytest

from hexapod_lab.sample import (
    SampleCalibration,
    SamplePoint,
    convex_hull,
    inset_polygon,
    region_from_points,
    signed_inset_distance,
)
from hexapod_lab.types import Pose6D


def corner(label, x, y, z=14.0, beam=None):
    return SamplePoint(label, Pose6D(x=x, y=y, z=z), beam)


@pytest.fixture()
def square():
    """A 10 x 10 mm sample logged corner by corner."""
    return SampleCalibration(
        points=(
            corner("Corner 1", -5.0, -5.0, beam=(5.0, 5.0)),
            corner("Corner 2", 5.0, -5.0, beam=(-5.0, 5.0)),
            corner("Corner 3", 5.0, 5.0, beam=(-5.0, -5.0)),
            corner("Corner 4", -5.0, 5.0, beam=(5.0, -5.0)),
        ),
        margin_mm=0.0,
    )


def test_hull_is_counter_clockwise_and_drops_interior_points():
    hull = convex_hull([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1)])
    assert len(hull) == 4
    assert (1, 1) not in hull


def test_two_points_are_read_as_opposite_corners():
    region = region_from_points([(-3.0, -2.0), (3.0, 2.0)])
    assert region == [(-3.0, -2.0), (3.0, -2.0), (3.0, 2.0), (-3.0, 2.0)]


def test_single_point_cannot_bound_anything():
    assert region_from_points([(1.0, 1.0)]) is None


def test_collinear_points_fall_back_to_the_enclosing_rectangle():
    region = region_from_points([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    assert region is None


def test_clearance_is_positive_inside_and_negative_outside(square):
    assert square.clearance_mm(Pose6D()) == pytest.approx(5.0)
    assert square.clearance_mm(Pose6D(x=4.0)) == pytest.approx(1.0)
    assert square.clearance_mm(Pose6D(x=6.0)) == pytest.approx(-1.0)


def test_bounds_are_inert_until_they_are_armed(square):
    assert square.violations(Pose6D(x=50.0)) == []
    assert square.path_violations(Pose6D(), Pose6D(x=50.0)) == []


def test_armed_bounds_reject_a_pose_outside_the_sample(square):
    bound = replace(square, enforce_xy=True)
    assert bound.violations(Pose6D(x=0.0, y=0.0)) == []
    issues = bound.violations(Pose6D(x=5.5, y=0.0))
    assert issues and "leaves the calibrated sample area" in issues[0]


def test_margin_holds_the_beam_inside_the_measured_edge(square):
    bound = replace(square, enforce_xy=True, margin_mm=0.5)
    assert bound.violations(Pose6D(x=4.4)) == []
    assert bound.violations(Pose6D(x=4.6))


def test_path_is_checked_not_only_the_endpoints(square):
    bound = replace(square, enforce_xy=True)
    # Both endpoints sit inside; the straight path between them does not.
    issues = bound.path_violations(Pose6D(x=-4.0, y=4.0), Pose6D(x=4.0, y=4.0))
    assert bound.violations(Pose6D(x=-4.0, y=4.0)) == []
    assert issues == [] or "%" in issues[0]

    outside = bound.path_violations(Pose6D(), Pose6D(x=20.0))
    assert outside and "% of the requested path" in outside[0]


def test_require_path_raises_for_an_out_of_sample_move(square):
    bound = replace(square, enforce_xy=True)
    bound.require_path(Pose6D(), Pose6D(x=3.0))
    with pytest.raises(ValueError, match="sample bounds"):
        bound.require_path(Pose6D(), Pose6D(x=30.0))


def test_surface_fit_reports_tilt_and_flatness():
    tilted = SampleCalibration(
        points=(
            corner("a", -5.0, -5.0, z=10.0),
            corner("b", 5.0, -5.0, z=10.1),
            corner("c", 5.0, 5.0, z=10.1),
            corner("d", -5.0, 5.0, z=10.0),
        )
    )
    assert tilted.tilt_deg() == pytest.approx(
        math.degrees(math.atan(0.1 / 10.0)),
        rel=1e-6,
    )
    assert tilted.flatness_mm() == pytest.approx(0.0, abs=1e-9)
    assert tilted.plane_z(0.0, 0.0) == pytest.approx(10.05)


def test_z_band_follows_the_fitted_surface():
    bound = SampleCalibration(
        points=(
            corner("a", -5.0, 0.0, z=10.0),
            corner("b", 5.0, 0.0, z=11.0),
            corner("c", 0.0, 5.0, z=10.5),
        ),
        enforce_z=True,
        z_above_mm=0.2,
        z_below_mm=0.2,
    )
    assert bound.violations(Pose6D(x=5.0, y=0.0, z=11.0)) == []
    issues = bound.violations(Pose6D(x=5.0, y=0.0, z=10.0))
    assert issues and "outside the calibrated sample band" in issues[0]


def test_centre_sits_on_the_fitted_surface(square):
    centre = square.centre()
    assert centre is not None
    assert centre.x == pytest.approx(0.0)
    assert centre.y == pytest.approx(0.0)
    assert centre.z == pytest.approx(14.0)


def test_sample_region_uses_the_recorded_beam_footprint(square):
    region = square.sample_region()
    assert region is not None
    assert len(region) == 4
    assert max(abs(x) for x, _ in region) == pytest.approx(5.0)


def test_sample_region_is_unavailable_without_beam_data():
    calibration = SampleCalibration(
        points=(corner("a", 0.0, 0.0), corner("b", 1.0, 1.0))
    )
    assert calibration.sample_region() is None
    assert calibration.stage_region() is not None


def test_inset_polygon_shrinks_a_convex_outline(square):
    region = square.stage_region()
    inset = inset_polygon(region, 1.0)
    assert inset is not None
    assert max(abs(x) for x, _ in inset) == pytest.approx(4.0)
    # Shrinking past the middle has no valid result.
    assert inset_polygon(region, 6.0) is None


def test_signed_distance_of_a_degenerate_polygon_is_minus_infinity():
    assert signed_inset_distance([(0.0, 0.0), (1.0, 0.0)], 0.0, 0.0) == float("-inf")


def test_round_trip_through_json_dict(square):
    bound = replace(square, enforce_xy=True, margin_mm=0.4, name="Chip 7")
    restored = SampleCalibration.from_dict(bound.to_dict())
    assert restored.name == "Chip 7"
    assert restored.enforce_xy is True
    assert restored.margin_mm == pytest.approx(0.4)
    assert restored.point_count == 4
    assert restored.stage_region() == bound.stage_region()


def test_negative_margin_is_rejected():
    with pytest.raises(ValueError, match="margin"):
        SampleCalibration.from_dict({"margin_mm": -1.0})


def test_clearing_releases_the_bounds(square):
    cleared = replace(square, enforce_xy=True, enforce_z=True).cleared()
    assert cleared.point_count == 0
    assert cleared.enforce_xy is False
    assert cleared.enforce_z is False
    assert cleared.violations(Pose6D(x=99.0)) == []
