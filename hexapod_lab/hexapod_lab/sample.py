"""Sample edge calibration for the hexapod lab controller.

The laboratory beam is fixed and the sample rides on the hexapod, so the
question "where is the sample?" is answered in *hexapod pose* coordinates: the
operator drives the stage until the beam sits on a physical edge/corner of the
sample and logs that pose. Two or more logged points define a region of hexapod
XY poses for which the beam still lands on the sample.

Nothing here is inferred from CAD. The region is exactly the measured points,
optionally shrunk by an operator-chosen safety margin.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Sequence

from .types import Pose6D


Point2D = tuple[float, float]


def _shoelace_area(polygon: Sequence[Point2D]) -> float:
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        total += x0 * y1 - x1 * y0
    return total / 2.0


def convex_hull(points: Sequence[Point2D]) -> list[Point2D]:
    """Counter-clockwise monotone-chain hull; duplicate points removed."""

    unique = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in points})
    if len(unique) <= 2:
        return list(unique)

    def cross(o: Point2D, a: Point2D, b: Point2D) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[Point2D] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[Point2D] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _rectangle(points: Sequence[Point2D]) -> list[Point2D]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def region_from_points(points: Sequence[Point2D]) -> list[Point2D] | None:
    """Return the CCW bounding polygon implied by the captured points.

    Two points are read as opposite corners of an axis-aligned rectangle, which
    is the fastest useful calibration for a rectangular chip. Three or more use
    the convex hull, falling back to the enclosing rectangle when the captured
    points are (near-)collinear and enclose no area.
    """

    cleaned = [
        (float(x), float(y))
        for x, y in points
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(cleaned) < 2:
        return None
    if len(cleaned) == 2:
        polygon = _rectangle(cleaned)
    else:
        polygon = convex_hull(cleaned)
        if len(polygon) < 3 or abs(_shoelace_area(polygon)) < 1e-6:
            polygon = _rectangle(cleaned)
    if abs(_shoelace_area(polygon)) < 1e-9:
        return None
    if _shoelace_area(polygon) < 0:
        polygon = list(reversed(polygon))
    return polygon


def signed_inset_distance(polygon: Sequence[Point2D], x: float, y: float) -> float:
    """Distance from (x, y) to the nearest polygon edge; negative when outside.

    The polygon is convex and counter-clockwise, so the minimum of the per-edge
    left-hand distances is the inward clearance.
    """

    if len(polygon) < 3:
        return float("-inf")
    clearance = float("inf")
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        ex, ey = x1 - x0, y1 - y0
        length = math.hypot(ex, ey)
        if length < 1e-12:
            continue
        # Positive on the interior (left) side of a CCW edge.
        clearance = min(clearance, ((x - x0) * ey - (y - y0) * ex) / -length)
    return clearance if math.isfinite(clearance) else float("-inf")


def inset_polygon(
    polygon: Sequence[Point2D],
    margin_mm: float,
) -> list[Point2D] | None:
    """Shrink a convex CCW polygon by ``margin_mm`` for display purposes."""

    if len(polygon) < 3:
        return None
    if margin_mm <= 1e-9:
        return list(polygon)

    lines: list[tuple[float, float, float]] = []
    for i, (x0, y0) in enumerate(polygon):
        x1, y1 = polygon[(i + 1) % len(polygon)]
        ex, ey = x1 - x0, y1 - y0
        length = math.hypot(ex, ey)
        if length < 1e-12:
            return None
        # The inward normal of a CCW edge points to its left.
        nx, ny = -ey / length, ex / length
        lines.append((nx, ny, nx * x0 + ny * y0 + margin_mm))

    result: list[Point2D] = []
    for i in range(len(lines)):
        a1, b1, c1 = lines[i - 1]
        a2, b2, c2 = lines[i]
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-12:
            return None
        result.append(((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det))
    if _shoelace_area(result) <= 1e-9:
        return None
    # Offset edges cross once the margin exceeds the half-width, which yields a
    # turned-inside-out polygon whose area is still positive. Every corner of a
    # real inset keeps at least the requested clearance from the original edges.
    if any(
        signed_inset_distance(polygon, x, y) < margin_mm - 1e-6
        for x, y in result
    ):
        return None
    return result


@dataclass(frozen=True, slots=True)
class SamplePoint:
    """One logged sample edge/corner."""

    label: str
    pose: Pose6D
    beam_xy_mm: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "pose": list(self.pose.as_tuple()),
            "beam_xy_mm": (
                None
                if self.beam_xy_mm is None
                else [float(self.beam_xy_mm[0]), float(self.beam_xy_mm[1])]
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SamplePoint":
        beam = data.get("beam_xy_mm")
        return cls(
            label=str(data.get("label", "edge point")),
            pose=Pose6D.from_iterable(data["pose"]),
            beam_xy_mm=(
                None
                if not beam
                else (float(beam[0]), float(beam[1]))
            ),
        )


@dataclass(frozen=True, slots=True)
class SampleCalibration:
    """Measured sample extent plus the operator's chosen safety margin."""

    name: str = "Sample"
    points: tuple[SamplePoint, ...] = ()
    margin_mm: float = 0.25
    enforce_xy: bool = False
    enforce_z: bool = False
    z_above_mm: float = 0.50
    z_below_mm: float = 0.50
    path_samples: int = 25

    # ------------------------------------------------------------ geometry
    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def is_usable(self) -> bool:
        return self.stage_region() is not None

    def stage_xy(self) -> list[Point2D]:
        return [(p.pose.x, p.pose.y) for p in self.points]

    def beam_xy(self) -> list[Point2D] | None:
        values = [p.beam_xy_mm for p in self.points]
        if not values or any(v is None for v in values):
            return None
        return [(float(v[0]), float(v[1])) for v in values]  # type: ignore[index]

    def stage_region(self) -> list[Point2D] | None:
        """Allowed hexapod XY polygon implied by the captured poses."""
        return region_from_points(self.stage_xy())

    def sample_region(self) -> list[Point2D] | None:
        """The same outline expressed in sample-surface coordinates."""
        beam = self.beam_xy()
        return None if beam is None else region_from_points(beam)

    def safe_region(self) -> list[Point2D] | None:
        region = self.stage_region()
        if region is None:
            return None
        return inset_polygon(region, max(0.0, float(self.margin_mm)))

    def area_mm2(self) -> float:
        region = self.stage_region()
        return 0.0 if region is None else abs(_shoelace_area(region))

    def extent_mm(self) -> tuple[float, float] | None:
        region = self.stage_region()
        if region is None:
            return None
        xs = [p[0] for p in region]
        ys = [p[1] for p in region]
        return (max(xs) - min(xs), max(ys) - min(ys))

    def centre(self) -> Pose6D | None:
        """Centroid pose: the mean captured pose, placed on the fitted surface."""
        if not self.points:
            return None
        count = float(len(self.points))
        values = [
            sum(p.pose.as_tuple()[i] for p in self.points) / count
            for i in range(6)
        ]
        centre = Pose6D(*values)
        plane_z = self.plane_z(centre.x, centre.y)
        if plane_z is not None:
            centre = replace(centre, z=plane_z)
        return centre

    # --------------------------------------------------------- surface fit
    def plane_coefficients(self) -> tuple[float, float, float] | None:
        """Least-squares fit of stage Z over stage XY: z = a + b*x + c*y."""
        if len(self.points) < 3:
            return None
        xs = [p.pose.x for p in self.points]
        ys = [p.pose.y for p in self.points]
        zs = [p.pose.z for p in self.points]
        n = float(len(xs))
        mx = sum(xs) / n
        my = sum(ys) / n
        mz = sum(zs) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxz = sum((x - mx) * (z - mz) for x, z in zip(xs, zs))
        syz = sum((y - my) * (z - mz) for y, z in zip(ys, zs))
        det = sxx * syy - sxy * sxy
        if abs(det) < 1e-9:
            return None
        b = (sxz * syy - syz * sxy) / det
        c = (syz * sxx - sxz * sxy) / det
        return (mz - b * mx - c * my, b, c)

    def plane_z(self, x: float, y: float) -> float | None:
        coefficients = self.plane_coefficients()
        if coefficients is None:
            if not self.points:
                return None
            return sum(p.pose.z for p in self.points) / float(len(self.points))
        a, b, c = coefficients
        return a + b * float(x) + c * float(y)

    def tilt_deg(self) -> float | None:
        coefficients = self.plane_coefficients()
        if coefficients is None:
            return None
        _, b, c = coefficients
        return math.degrees(math.atan(math.hypot(b, c)))

    def flatness_mm(self) -> float | None:
        """Worst residual of the captured Z values about the fitted plane."""
        if len(self.points) < 3:
            return None
        worst = 0.0
        for point in self.points:
            fitted = self.plane_z(point.pose.x, point.pose.y)
            if fitted is None:
                return None
            worst = max(worst, abs(point.pose.z - fitted))
        return worst

    # ------------------------------------------------------------- limits
    def clearance_mm(self, pose: Pose6D) -> float | None:
        """Signed XY clearance to the calibrated edge; the margin is not applied."""
        region = self.stage_region()
        if region is None:
            return None
        return signed_inset_distance(region, pose.x, pose.y)

    def violations(self, pose: Pose6D) -> list[str]:
        issues: list[str] = []
        if self.enforce_xy:
            region = self.stage_region()
            if region is None:
                issues.append(
                    "sample bounds are enabled but fewer than two edge points "
                    "are calibrated"
                )
            else:
                clearance = signed_inset_distance(region, pose.x, pose.y)
                margin = max(0.0, float(self.margin_mm))
                if clearance < margin - 1e-9:
                    issues.append(
                        f"X={pose.x:.4f}, Y={pose.y:.4f} mm leaves the "
                        f"calibrated sample area (clearance {clearance:+.4f} mm, "
                        f"required margin {margin:.4f} mm)"
                    )
        if self.enforce_z:
            surface = self.plane_z(pose.x, pose.y)
            if surface is None:
                issues.append(
                    "the sample Z band is enabled but no edge point is calibrated"
                )
            else:
                low = surface - abs(float(self.z_below_mm))
                high = surface + abs(float(self.z_above_mm))
                if pose.z < low - 1e-9 or pose.z > high + 1e-9:
                    issues.append(
                        f"Z={pose.z:.4f} mm is outside the calibrated sample "
                        f"band [{low:.4f}, {high:.4f}] mm"
                    )
        return issues

    def path_violations(self, start: Pose6D, target: Pose6D) -> list[str]:
        if not (self.enforce_xy or self.enforce_z):
            return []
        samples = max(2, int(self.path_samples))
        for index in range(samples + 1):
            fraction = index / samples
            issues = self.violations(start.lerp(target, fraction))
            if issues:
                where = f" (at {fraction * 100:.1f}% of the requested path)"
                return [issue + where for issue in issues]
        return []

    def require_path(self, start: Pose6D, target: Pose6D) -> None:
        issues = self.path_violations(start, target)
        if issues:
            raise ValueError("Motion blocked by sample bounds: " + issues[0])

    # ------------------------------------------------------------ editing
    def with_point(self, point: SamplePoint) -> "SampleCalibration":
        return replace(self, points=(*self.points, point))

    def without_point(self, index: int) -> "SampleCalibration":
        points = list(self.points)
        if 0 <= index < len(points):
            points.pop(index)
        return replace(self, points=tuple(points))

    def cleared(self) -> "SampleCalibration":
        return replace(self, points=(), enforce_xy=False, enforce_z=False)

    def summary(self) -> str:
        if not self.points:
            return "No sample edge points captured"
        extent = self.extent_mm()
        if extent is None:
            return (
                f"{len(self.points)} point captured - at least two distinct "
                "corners are needed to bound the sample"
            )
        text = (
            f"{len(self.points)} edge points  -  "
            f"{extent[0]:.3f} x {extent[1]:.3f} mm  -  "
            f"{self.area_mm2():.3f} mm2"
        )
        tilt = self.tilt_deg()
        if tilt is not None:
            text += f"  -  surface tilt {tilt:.3f} deg"
        flatness = self.flatness_mm()
        if flatness is not None:
            text += f"  -  flatness +/-{flatness:.4f} mm"
        return text

    # ---------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "margin_mm": float(self.margin_mm),
            "enforce_xy": bool(self.enforce_xy),
            "enforce_z": bool(self.enforce_z),
            "z_above_mm": float(self.z_above_mm),
            "z_below_mm": float(self.z_below_mm),
            "path_samples": int(self.path_samples),
            "points": [point.to_dict() for point in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SampleCalibration":
        if not data:
            return cls()
        points = tuple(
            SamplePoint.from_dict(item)
            for item in data.get("points", [])
        )
        margin = float(data.get("margin_mm", 0.25))
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("the sample margin must be finite and non-negative")
        samples = int(data.get("path_samples", 25))
        if samples < 2 or samples > 501:
            raise ValueError("sample path_samples must be between 2 and 501")
        return cls(
            name=str(data.get("name", "Sample")),
            points=points,
            margin_mm=margin,
            enforce_xy=bool(data.get("enforce_xy", False)),
            enforce_z=bool(data.get("enforce_z", False)),
            z_above_mm=float(data.get("z_above_mm", 0.5)),
            z_below_mm=float(data.get("z_below_mm", 0.5)),
            path_samples=samples,
        )


def next_point_label(index: int) -> str:
    """Default label for the ``index``-th captured point (0-based)."""
    corners = ("Corner 1", "Corner 2", "Corner 3", "Corner 4")
    if index < len(corners):
        return corners[index]
    return f"Edge point {index + 1}"


def default_corner_labels() -> Iterable[str]:
    index = 0
    while True:
        yield next_point_label(index)
        index += 1
