from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable

from .types import Pose6D


class StepKind(str, Enum):
    MOVE_ABSOLUTE = "move_absolute"
    MOVE_INCREMENTAL = "move_incremental"
    LASER_GATE = "laser_gate"
    WAIT = "wait"


@dataclass(slots=True)
class RecipeStep:
    kind: StepKind
    label: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def move_absolute(cls, pose: Pose6D) -> "RecipeStep":
        return cls(StepKind.MOVE_ABSOLUTE, "Move absolute", {"pose": list(pose.as_tuple())})

    @classmethod
    def move_incremental(cls, delta: Pose6D) -> "RecipeStep":
        return cls(StepKind.MOVE_INCREMENTAL, "Move incremental", {"delta": list(delta.as_tuple())})

    @classmethod
    def laser_gate(cls, enabled: bool) -> "RecipeStep":
        return cls(StepKind.LASER_GATE, "Laser gate ON" if enabled else "Laser gate OFF", {"enabled": bool(enabled)})

    @classmethod
    def wait(cls, seconds: float) -> "RecipeStep":
        seconds = max(0.0, float(seconds))
        return cls(StepKind.WAIT, f"Wait {seconds:g} s", {"seconds": seconds})

    def describe(self) -> str:
        if self.kind == StepKind.MOVE_ABSOLUTE:
            p = Pose6D.from_iterable(self.payload["pose"])
            return "ABS  " + "  ".join(f"{n}={v:.3f}" for n, v in zip("XYZUVW", p.as_tuple()))
        if self.kind == StepKind.MOVE_INCREMENTAL:
            p = Pose6D.from_iterable(self.payload["delta"])
            return "REL  " + "  ".join(f"d{n}={v:.3f}" for n, v in zip("XYZUVW", p.as_tuple()))
        if self.kind == StepKind.LASER_GATE:
            return "LASER GATE ON" if self.payload.get("enabled") else "LASER GATE OFF"
        if self.kind == StepKind.WAIT:
            return f"WAIT {float(self.payload.get('seconds', 0.0)):.3f} s"
        return self.label

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "label": self.label, "payload": self.payload}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipeStep":
        return cls(StepKind(data["kind"]), str(data.get("label", data["kind"])), dict(data.get("payload", {})))


@dataclass(slots=True)
class Recipe:
    name: str = "Untitled recipe"
    steps: list[RecipeStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "name": self.name, "steps": [s.to_dict() for s in self.steps]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Recipe":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(data.get("version", 1)) != 1:
            raise ValueError("unsupported recipe version")
        return cls(name=str(data.get("name", "Recipe")), steps=[RecipeStep.from_dict(x) for x in data.get("steps", [])])


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    severity: str
    message: str
    step_index: int | None = None


def preflight_recipe(recipe: Recipe) -> list[PreflightIssue]:
    issues: list[PreflightIssue] = []
    laser_on = False
    for i, step in enumerate(recipe.steps):
        try:
            if step.kind == StepKind.MOVE_ABSOLUTE:
                Pose6D.from_iterable(step.payload["pose"])
            elif step.kind == StepKind.MOVE_INCREMENTAL:
                Pose6D.from_iterable(step.payload["delta"])
            elif step.kind == StepKind.LASER_GATE:
                enabled = bool(step.payload.get("enabled"))
                if enabled == laser_on:
                    issues.append(PreflightIssue("warning", f"laser gate is already {'ON' if enabled else 'OFF'}", i))
                laser_on = enabled
            elif step.kind == StepKind.WAIT:
                if float(step.payload.get("seconds", 0.0)) < 0:
                    issues.append(PreflightIssue("error", "negative wait time", i))
        except Exception as exc:
            issues.append(PreflightIssue("error", f"invalid step: {exc}", i))

    if laser_on:
        issues.append(PreflightIssue("error", "recipe ends with the laser gate ON; add an explicit gate-OFF step"))
    if not recipe.steps:
        issues.append(PreflightIssue("warning", "recipe contains no steps"))
    return issues
