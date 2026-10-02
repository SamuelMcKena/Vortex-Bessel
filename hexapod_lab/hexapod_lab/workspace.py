from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from .kinematics import RigKinematics
from .recipe import Recipe, StepKind
from .sample import SampleCalibration
from .types import Pose6D


AXES = "XYZUVW"


@dataclass(frozen=True, slots=True)
class LimitViolation:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class RecipeWorkspaceIssue:
    step_index: int
    message: str

@dataclass(frozen=True, slots=True)
class WorkspaceLimits:
    """Coupled software envelope for the configured Stewart platform.

    Cartesian bounds are only the first check. Every pose is also converted to
    six strut lengths and changes in strut direction. Paths are sampled because
    a valid endpoint does not guarantee that every intermediate pose is valid.
    """

    minimum_pose: Pose6D
    maximum_pose: Pose6D
    minimum_leg_lengths_mm: tuple[float, float, float, float, float, float]
    maximum_leg_lengths_mm: tuple[float, float, float, float, float, float]
    maximum_leg_tilt_deg: float = 12.0
    maximum_line_velocity_mm_s: float = 2.1
    path_samples: int = 25
    source: str = "unconfigured"
    controller_verified: bool = False

    @classmethod
    def cad_conservative(cls, kinematics: RigKinematics) -> "WorkspaceLimits":
        """Return a deliberately conservative commissioning envelope.

        The supplied STEP identifies geometry and the neutral strut length, but
        it does not certify actuator stroke, joint travel, payload or collision
        limits. The +/-50 mm strut window and Cartesian caps therefore remain
        unverified until replaced by controller/manual values.
        """

        home = tuple(float(v) for v in kinematics.leg_lengths_mm(Pose6D()))
        return cls(
            minimum_pose=Pose6D(-25.0, -25.0, 0.0, -8.0, -8.0, -10.0),
            maximum_pose=Pose6D(25.0, 25.0, 28.0, 8.0, 8.0, 10.0),
            minimum_leg_lengths_mm=tuple(v - 50.0 for v in home),
            maximum_leg_lengths_mm=tuple(v + 50.0 for v in home),
            maximum_leg_tilt_deg=12.0,
            maximum_line_velocity_mm_s=2.1,
            path_samples=25,
            source=(
                "provisional CAD/HXP-manual envelope; 2.1 mm/s ceiling from "
                "legacy TCL operating evidence"
            ),
            controller_verified=False,
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        kinematics: RigKinematics,
    ) -> "WorkspaceLimits":
        default = cls.cad_conservative(kinematics)
        if not data:
            return default

        axis_data = dict(data.get("axes", {}))
        minimum = []
        maximum = []
        for index, axis in enumerate(AXES):
            fallback = (
                default.minimum_pose.as_tuple()[index],
                default.maximum_pose.as_tuple()[index],
            )
            values = axis_data.get(axis, fallback)
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ValueError(f"workspace axis {axis} must contain [min, max]")
            lo, hi = map(float, values)
            if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
                raise ValueError(f"workspace axis {axis} has invalid limits")
            minimum.append(lo)
            maximum.append(hi)

        leg_data = dict(data.get("struts", {}))
        leg_min = cls._six_values(
            leg_data.get("minimum_length_mm"),
            default.minimum_leg_lengths_mm,
            "minimum strut lengths",
        )
        leg_max = cls._six_values(
            leg_data.get("maximum_length_mm"),
            default.maximum_leg_lengths_mm,
            "maximum strut lengths",
        )
        if any(lo >= hi for lo, hi in zip(leg_min, leg_max)):
            raise ValueError("every strut minimum must be below its maximum")

        tilt = float(
            leg_data.get("maximum_tilt_from_home_deg", default.maximum_leg_tilt_deg)
        )
        velocity = float(
            data.get(
                "maximum_line_velocity_mm_s",
                default.maximum_line_velocity_mm_s,
            )
        )
        samples = int(data.get("path_samples", default.path_samples))
        if not math.isfinite(tilt) or tilt <= 0:
            raise ValueError("maximum strut tilt must be finite and positive")
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError("maximum line velocity must be finite and positive")
        if samples < 2 or samples > 501:
            raise ValueError("workspace path_samples must be between 2 and 501")

        return cls(
            minimum_pose=Pose6D(*minimum),
            maximum_pose=Pose6D(*maximum),
            minimum_leg_lengths_mm=leg_min,
            maximum_leg_lengths_mm=leg_max,
            maximum_leg_tilt_deg=tilt,
            maximum_line_velocity_mm_s=velocity,
            path_samples=samples,
            source=str(data.get("source", default.source)),
            controller_verified=bool(data.get("controller_verified", False)),
        )

    @staticmethod
    def _six_values(
        values: Any,
        fallback: Iterable[float],
        label: str,
    ) -> tuple[float, float, float, float, float, float]:
        raw = tuple(fallback) if values is None else tuple(float(v) for v in values)
        if len(raw) != 6 or not all(math.isfinite(v) for v in raw):
            raise ValueError(f"{label} must contain six finite values")
        return raw  # type: ignore[return-value]

    def to_dict(self) -> dict[str, Any]:
        minimum = self.minimum_pose.as_tuple()
        maximum = self.maximum_pose.as_tuple()
        return {
            "source": self.source,
            "controller_verified": self.controller_verified,
            "axes": {
                axis: [minimum[i], maximum[i]]
                for i, axis in enumerate(AXES)
            },
            "struts": {
                "minimum_length_mm": list(self.minimum_leg_lengths_mm),
                "maximum_length_mm": list(self.maximum_leg_lengths_mm),
                "maximum_tilt_from_home_deg": self.maximum_leg_tilt_deg,
            },
            "maximum_line_velocity_mm_s": self.maximum_line_velocity_mm_s,
            "path_samples": self.path_samples,
        }

    def axis_bounds(self, axis: str) -> tuple[float, float]:
        index = AXES.index(axis.upper())
        return (
            self.minimum_pose.as_tuple()[index],
            self.maximum_pose.as_tuple()[index],
        )

    def violations(
        self,
        kinematics: RigKinematics,
        pose: Pose6D,
    ) -> list[LimitViolation]:
        values = pose.as_tuple()
        if not all(math.isfinite(float(v)) for v in values):
            return [LimitViolation("non_finite", "pose contains a non-finite value")]

        issues: list[LimitViolation] = []
        minimum = self.minimum_pose.as_tuple()
        maximum = self.maximum_pose.as_tuple()
        for axis, value, lo, hi in zip(AXES, values, minimum, maximum):
            if value < lo - 1e-9 or value > hi + 1e-9:
                unit = "mm" if axis in "XYZ" else "deg"
                issues.append(
                    LimitViolation(
                        f"axis_{axis.lower()}",
                        f"{axis}={value:.4f} {unit} is outside [{lo:.4f}, {hi:.4f}] {unit}",
                    )
                )

        lengths = kinematics.leg_lengths_mm(pose)
        for index, (length, lo, hi) in enumerate(
            zip(
                lengths,
                self.minimum_leg_lengths_mm,
                self.maximum_leg_lengths_mm,
            ),
            start=1,
        ):
            if length < lo - 1e-9 or length > hi + 1e-9:
                issues.append(
                    LimitViolation(
                        f"strut_{index}",
                        f"strut {index} length {length:.3f} mm is outside "
                        f"[{lo:.3f}, {hi:.3f}] mm",
                    )
                )

        bottom = kinematics.bottom_joint_positions()
        home_top = kinematics.top_joint_positions(Pose6D())
        moved_top = kinematics.top_joint_positions(pose)
        home_vectors = home_top - bottom
        moved_vectors = moved_top - bottom
        dots = np.sum(home_vectors * moved_vectors, axis=1)
        denom = np.linalg.norm(home_vectors, axis=1) * np.linalg.norm(
            moved_vectors,
            axis=1,
        )
        angles = np.degrees(
            np.arccos(np.clip(dots / np.maximum(denom, 1e-12), -1.0, 1.0))
        )
        for index, angle in enumerate(angles, start=1):
            if angle > self.maximum_leg_tilt_deg + 1e-9:
                issues.append(
                    LimitViolation(
                        f"joint_{index}",
                        f"strut {index} direction changes {angle:.3f} deg; "
                        f"limit is {self.maximum_leg_tilt_deg:.3f} deg",
                    )
                )
        return issues

    def cartesian_violations(self, pose: Pose6D) -> list[LimitViolation]:
        """Check only XYZUVW bounds, without using the decorative STEP geometry.

        REAL LAB uses this for general motion because the supplied STEP model is
        not the Newport controller's physical kinematic model. Coupled Line
        feasibility is instead checked by the HXP itself.
        """
        values = pose.as_tuple()
        if not all(math.isfinite(float(v)) for v in values):
            return [LimitViolation("non_finite", "pose contains a non-finite value")]
        issues: list[LimitViolation] = []
        minimum = self.minimum_pose.as_tuple()
        maximum = self.maximum_pose.as_tuple()
        for axis, value, lo, hi in zip(AXES, values, minimum, maximum):
            if value < lo - 1e-9 or value > hi + 1e-9:
                unit = "mm" if axis in "XYZ" else "deg"
                issues.append(
                    LimitViolation(
                        f"axis_{axis.lower()}",
                        f"{axis}={value:.4f} {unit} is outside "
                        f"[{lo:.4f}, {hi:.4f}] {unit}",
                    )
                )
        return issues

    def require_cartesian_pose(self, pose: Pose6D) -> None:
        issues = self.cartesian_violations(pose)
        if issues:
            raise ValueError("Motion blocked by Cartesian limits: " + issues[0].message)

    def path_violations(
        self,
        kinematics: RigKinematics,
        start: Pose6D,
        target: Pose6D,
    ) -> list[LimitViolation]:
        for sample in range(self.path_samples + 1):
            fraction = sample / self.path_samples
            issues = self.violations(kinematics, start.lerp(target, fraction))
            if issues:
                location = f"at {fraction * 100:.1f}% of the requested path"
                return [
                    LimitViolation(issue.code, f"{issue.message} ({location})")
                    for issue in issues
                ]
        return []

    def require_path(
        self,
        kinematics: RigKinematics,
        start: Pose6D,
        target: Pose6D,
    ) -> None:
        issues = self.path_violations(kinematics, start, target)
        if issues:
            raise ValueError("Motion blocked by workspace limits: " + issues[0].message)

    def require_line_velocity(self, velocity_mm_s: float) -> None:
        velocity = float(velocity_mm_s)
        if not math.isfinite(velocity) or velocity <= 0:
            raise ValueError("line velocity must be finite and positive")
        if velocity > self.maximum_line_velocity_mm_s + 1e-9:
            raise ValueError(
                f"line velocity {velocity:.3f} mm/s exceeds configured limit "
                f"{self.maximum_line_velocity_mm_s:.3f} mm/s"
            )


def validate_recipe_workspace(
    recipe: Recipe,
    start_pose: Pose6D,
    limits: WorkspaceLimits,
    kinematics: RigKinematics,
    sample: SampleCalibration | None = None,
) -> list[RecipeWorkspaceIssue]:
    """Simulate recipe motion and validate every sampled segment.

    When a sample calibration with active bounds is supplied, each segment is
    also checked against the measured sample extent.
    """

    issues: list[RecipeWorkspaceIssue] = []
    current = start_pose
    for index, step in enumerate(recipe.steps):
        try:
            target: Pose6D | None = None
            if step.kind == StepKind.MOVE_ABSOLUTE:
                target = Pose6D.from_iterable(step.payload["pose"])
            elif step.kind == StepKind.MOVE_INCREMENTAL:
                target = current.plus(Pose6D.from_iterable(step.payload["delta"]))
            elif step.kind in (
                StepKind.MOVE_LINE_VELOCITY,
                StepKind.WRITE_LINE,
            ):
                delta = list(step.payload["delta_xyz_mm"])
                target = current.plus(Pose6D(*map(float, delta)))
                limits.require_line_velocity(float(step.payload["velocity_mm_s"]))

            if target is not None:
                segment_issues = limits.path_violations(
                    kinematics,
                    current,
                    target,
                )
                if segment_issues:
                    issues.append(
                        RecipeWorkspaceIssue(index, segment_issues[0].message)
                    )
                elif sample is not None:
                    sample_issues = sample.path_violations(current, target)
                    if sample_issues:
                        issues.append(
                            RecipeWorkspaceIssue(index, sample_issues[0])
                        )
                current = target
        except Exception as exc:
            issues.append(RecipeWorkspaceIssue(index, str(exc)))
    return issues
