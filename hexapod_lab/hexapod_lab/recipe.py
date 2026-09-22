from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any

from .types import Pose6D


class StepKind(str, Enum):
    MOVE_ABSOLUTE = "move_absolute"
    MOVE_INCREMENTAL = "move_incremental"
    POCKELS_CELL = "pockels_cell"
    ATTENUATOR_SET = "attenuator_set"
    WAIT = "wait"


@dataclass(slots=True)
class RecipeStep:
    kind: StepKind
    label: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def move_absolute(cls, pose: Pose6D) -> "RecipeStep":
        return cls(
            StepKind.MOVE_ABSOLUTE,
            "Move absolute",
            {"pose": list(pose.as_tuple())},
        )

    @classmethod
    def move_incremental(cls, delta: Pose6D) -> "RecipeStep":
        return cls(
            StepKind.MOVE_INCREMENTAL,
            "Move incremental",
            {"delta": list(delta.as_tuple())},
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
        if not 0.0 <= value <= 100.0:
            raise ValueError("attenuator transmission must be between 0 and 100 %")
        return cls(
            StepKind.ATTENUATOR_SET,
            f"Attenuator {value:g} %",
            {"transmission_percent": value},
        )

    @classmethod
    def wait(cls, seconds: float) -> "RecipeStep":
        seconds = max(0.0, float(seconds))
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
        return {
            "version": 2,
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
        if version not in (1, 2):
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
                Pose6D.from_iterable(step.payload["pose"])

            elif step.kind == StepKind.MOVE_INCREMENTAL:
                Pose6D.from_iterable(step.payload["delta"])

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
                            "warning",
                            "attenuator is changed while the Pockels cell is OPEN",
                            i,
                        )
                    )

            elif step.kind == StepKind.WAIT:
                if float(step.payload.get("seconds", 0.0)) < 0:
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
