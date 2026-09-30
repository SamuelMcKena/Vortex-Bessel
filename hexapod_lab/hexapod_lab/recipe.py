from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any

from .types import Pose6D


class StepKind(str, Enum):
    MOVE_ABSOLUTE = "move_absolute"
    MOVE_INCREMENTAL = "move_incremental"
    MOVE_LINE_VELOCITY = "move_line_velocity"
    WRITE_LINE = "write_line"
    POCKELS_CELL = "pockels_cell"
    ATTENUATOR_SET = "attenuator_set"
    WAIT = "wait"


#: Steps that command stage motion.
MOTION_KINDS = frozenset(
    {
        StepKind.MOVE_ABSOLUTE,
        StepKind.MOVE_INCREMENTAL,
        StepKind.MOVE_LINE_VELOCITY,
        StepKind.WRITE_LINE,
    }
)

#: Steps that command the Pockels cell.
POCKELS_KINDS = frozenset({StepKind.POCKELS_CELL, StepKind.WRITE_LINE})


@dataclass(slots=True)
class RecipeStep:
    kind: StepKind
    label: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def move_absolute(cls, pose: Pose6D) -> "RecipeStep":
        if not all(math.isfinite(v) for v in pose.as_tuple()):
            raise ValueError("absolute pose values must be finite")
        return cls(
            StepKind.MOVE_ABSOLUTE,
            "Move absolute",
            {"pose": list(pose.as_tuple())},
        )

    @classmethod
    def move_incremental(cls, delta: Pose6D) -> "RecipeStep":
        if not all(math.isfinite(v) for v in delta.as_tuple()):
            raise ValueError("incremental move values must be finite")
        return cls(
            StepKind.MOVE_INCREMENTAL,
            "Move incremental",
            {"delta": list(delta.as_tuple())},
        )

    @classmethod
    def move_line_velocity(
        cls,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        velocity_mm_s: float,
    ) -> "RecipeStep":
        delta = [float(dx_mm), float(dy_mm), float(dz_mm)]
        velocity = float(velocity_mm_s)
        if not all(math.isfinite(v) for v in [*delta, velocity]):
            raise ValueError("line move values must be finite")
        if velocity <= 0:
            raise ValueError("line target velocity must be > 0 mm/s")
        return cls(
            StepKind.MOVE_LINE_VELOCITY,
            "Line move at target velocity",
            {
                "delta_xyz_mm": delta,
                "velocity_mm_s": velocity,
            },
        )

    @classmethod
    def write_line(
        cls,
        dx_mm: float,
        dy_mm: float,
        dz_mm: float,
        velocity_mm_s: float,
    ) -> "RecipeStep":
        """Move while writing: Pockels OPEN for the move, CLOSED afterwards.

        This mirrors the recovered LabVIEW ``LINE Move_While Write`` behaviour as
        one indivisible block, so a sequence cannot be reordered into a state
        where the beam is left open across a repositioning move.
        """

        delta = [float(dx_mm), float(dy_mm), float(dz_mm)]
        velocity = float(velocity_mm_s)
        if not all(math.isfinite(v) for v in [*delta, velocity]):
            raise ValueError("write-line values must be finite")
        if velocity <= 0:
            raise ValueError("write-line target velocity must be > 0 mm/s")
        if not any(abs(v) > 0 for v in delta):
            raise ValueError("a write line needs a non-zero dX, dY or dZ")
        return cls(
            StepKind.WRITE_LINE,
            "Move while write",
            {
                "delta_xyz_mm": delta,
                "velocity_mm_s": velocity,
            },
        )

    @classmethod
    def pockels_cell(cls, open_: bool) -> "RecipeStep":
        return cls(
            StepKind.POCKELS_CELL,
            "Pockels OPEN" if open_ else "Pockels CLOSED",
            {"open": bool(open_)},
        )

    @classmethod
    def laser_gate(cls, enabled: bool) -> "RecipeStep":
        """Backward-compatible alias used by v0.1 callers."""
        return cls.pockels_cell(bool(enabled))

    @classmethod
    def attenuator_set(cls, transmission_percent: float) -> "RecipeStep":
        value = float(transmission_percent)
        if not math.isfinite(value):
            raise ValueError("attenuator transmission must be finite")
        if not 0.0 <= value <= 100.0:
            raise ValueError("attenuator transmission must be between 0 and 100 %")
        return cls(
            StepKind.ATTENUATOR_SET,
            f"Attenuator {value:g} %",
            {"transmission_percent": value},
        )

    @classmethod
    def wait(cls, seconds: float) -> "RecipeStep":
        seconds = float(seconds)
        if not math.isfinite(seconds):
            raise ValueError("wait time must be finite")
        seconds = max(0.0, seconds)
        return cls(
            StepKind.WAIT,
            f"Wait {seconds:g} s",
            {"seconds": seconds},
        )

    def describe(self) -> str:
        if self.kind == StepKind.MOVE_ABSOLUTE:
            p = Pose6D.from_iterable(self.payload["pose"])
            return "ABS  " + "  ".join(
                f"{name}={value:.3f}"
                for name, value in zip("XYZUVW", p.as_tuple())
            )
        if self.kind == StepKind.MOVE_INCREMENTAL:
            p = Pose6D.from_iterable(self.payload["delta"])
            return "REL  " + "  ".join(
                f"d{name}={value:.3f}"
                for name, value in zip("XYZUVW", p.as_tuple())
            )
        if self.kind == StepKind.MOVE_LINE_VELOCITY:
            delta = [
                float(v)
                for v in self.payload.get(
                    "delta_xyz_mm",
                    [0.0, 0.0, 0.0],
                )
            ]
            velocity = float(self.payload.get("velocity_mm_s", 0.0))
            return (
                "LINE  "
                f"dX={delta[0]:.3f}  dY={delta[1]:.3f}  "
                f"dZ={delta[2]:.3f}  @ {velocity:.3f} mm/s"
            )
        if self.kind == StepKind.WRITE_LINE:
            delta = [
                float(v)
                for v in self.payload.get(
                    "delta_xyz_mm",
                    [0.0, 0.0, 0.0],
                )
            ]
            velocity = float(self.payload.get("velocity_mm_s", 0.0))
            return (
                "WRITE  "
                f"dX={delta[0]:.3f}  dY={delta[1]:.3f}  "
                f"dZ={delta[2]:.3f}  @ {velocity:.3f} mm/s  "
                "(beam OPEN during the move only)"
            )
        if self.kind == StepKind.POCKELS_CELL:
            return (
                "LASER ON  •  POCKELS OPEN"
                if bool(self.payload.get("open"))
                else "LASER OFF  •  POCKELS CLOSED"
            )
        if self.kind == StepKind.ATTENUATOR_SET:
            return (
                "ATTENUATOR  "
                f"{float(self.payload.get('transmission_percent', 0.0)):.2f} %"
            )
        if self.kind == StepKind.WAIT:
            return f"WAIT  {float(self.payload.get('seconds', 0.0)):.3f} s"
        return self.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipeStep":
        raw_kind = str(data["kind"])
        payload = dict(data.get("payload", {}))

        # v0.1 recipes serialized the Pockels/LX13 state as "laser_gate".
        if raw_kind == "laser_gate":
            return cls.pockels_cell(bool(payload.get("enabled")))

        kind = StepKind(raw_kind)
        return cls(
            kind,
            str(data.get("label", raw_kind)),
            payload,
        )


@dataclass(slots=True)
class Recipe:
    name: str = "Untitled recipe"
    steps: list[RecipeStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # v3 adds the combined write-line ("move while write") block. Earlier
        # files remain loadable.
        return {
            "version": 3,
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = int(data.get("version", 1))
        if version not in (1, 2, 3):
            raise ValueError(f"unsupported recipe version {version}")
        return cls(
            name=str(data.get("name", "Recipe")),
            steps=[
                RecipeStep.from_dict(item)
                for item in data.get("steps", [])
            ],
        )


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    severity: str
    message: str
    step_index: int | None = None


def preflight_recipe(recipe: Recipe) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    pockels_open = False

    for i, step in enumerate(recipe.steps):
        try:
            if step.kind == StepKind.MOVE_ABSOLUTE:
                if pockels_open:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "ordinary motion while the Pockels cell is OPEN is "
                            "not allowed; use MOVE WHILE WRITE or close the beam first",
                            i,
                        )
                    )
                pose = Pose6D.from_iterable(step.payload["pose"])
                if not all(math.isfinite(v) for v in pose.as_tuple()):
                    raise ValueError("absolute pose values must be finite")

            elif step.kind == StepKind.MOVE_INCREMENTAL:
                if pockels_open:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "ordinary motion while the Pockels cell is OPEN is "
                            "not allowed; use MOVE WHILE WRITE or close the beam first",
                            i,
                        )
                    )
                delta_pose = Pose6D.from_iterable(step.payload["delta"])
                if not all(math.isfinite(v) for v in delta_pose.as_tuple()):
                    raise ValueError("incremental move values must be finite")

            elif step.kind == StepKind.MOVE_LINE_VELOCITY:
                if pockels_open:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "ordinary Line motion while the Pockels cell is OPEN is "
                            "not allowed; use MOVE WHILE WRITE or close the beam first",
                            i,
                        )
                    )
                delta = list(step.payload["delta_xyz_mm"])
                if len(delta) != 3:
                    raise ValueError(
                        "line move requires dX, dY and dZ"
                    )
                numeric_delta = [float(v) for v in delta]
                velocity = float(step.payload["velocity_mm_s"])
                if not all(
                    math.isfinite(v)
                    for v in [*numeric_delta, velocity]
                ):
                    raise ValueError("line move values must be finite")
                if velocity <= 0:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "line target velocity must be > 0 mm/s",
                            i,
                        )
                    )

            elif step.kind == StepKind.WRITE_LINE:
                delta = list(step.payload["delta_xyz_mm"])
                if len(delta) != 3:
                    raise ValueError("write line requires dX, dY and dZ")
                numeric_delta = [float(v) for v in delta]
                velocity = float(step.payload["velocity_mm_s"])
                if not all(
                    math.isfinite(v)
                    for v in [*numeric_delta, velocity]
                ):
                    raise ValueError("write-line values must be finite")
                if velocity <= 0:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "write-line target velocity must be > 0 mm/s",
                            i,
                        )
                    )
                if not any(abs(v) > 0 for v in numeric_delta):
                    issues.append(
                        PreflightIssue(
                            "error",
                            "write line has zero length",
                            i,
                        )
                    )
                if pockels_open:
                    issues.append(
                        PreflightIssue(
                            "warning",
                            "the Pockels cell is already OPEN; this write block "
                            "closes it when the line finishes",
                            i,
                        )
                    )
                # The block opens the cell for the move and closes it again.
                pockels_open = False

            elif step.kind == StepKind.POCKELS_CELL:
                requested = bool(step.payload.get("open"))
                if requested == pockels_open:
                    issues.append(
                        PreflightIssue(
                            "warning",
                            "Pockels cell is already "
                            + ("OPEN" if requested else "CLOSED"),
                            i,
                        )
                    )
                pockels_open = requested

            elif step.kind == StepKind.ATTENUATOR_SET:
                value = float(step.payload["transmission_percent"])
                if not math.isfinite(value):
                    raise ValueError("attenuator transmission must be finite")
                if not 0.0 <= value <= 100.0:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "attenuator transmission must be between 0 and 100 %",
                            i,
                        )
                    )
                if pockels_open:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "attenuator changes while the Pockels cell is OPEN are "
                            "blocked; close the beam before changing transmission",
                            i,
                        )
                    )

            elif step.kind == StepKind.WAIT:
                if pockels_open:
                    issues.append(
                        PreflightIssue(
                            "warning",
                            "WAIT occurs while the Pockels cell is OPEN; this creates "
                            "a stationary exposure/dwell",
                            i,
                        )
                    )
                seconds = float(step.payload.get("seconds", 0.0))
                if not math.isfinite(seconds):
                    raise ValueError("wait time must be finite")
                if seconds < 0:
                    issues.append(
                        PreflightIssue(
                            "error",
                            "negative wait time",
                            i,
                        )
                    )

        except Exception as exc:
            issues.append(
                PreflightIssue(
                    "error",
                    f"invalid step: {exc}",
                    i,
                )
            )

    if pockels_open:
        issues.append(
            PreflightIssue(
                "error",
                "recipe ends with the Pockels cell OPEN; add an explicit LASER OFF step",
            )
        )

    if not recipe.steps:
        issues.append(
            PreflightIssue(
                "warning",
                "recipe contains no steps",
            )
        )

    return issues
