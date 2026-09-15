"""Qt-independent, serialisable and resumable experiment recipes."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .calibration import CalibrationKey, CalibrationRegistry, CalibrationStatus
from .state import ExperimentStore, utc_now


class StepKind(str, Enum):
    CONFIGURE = "CONFIGURE"
    GUIDED_MOVE = "GUIDED_MOVE"
    CAPTURE = "CAPTURE"
    ANALYSE = "ANALYSE"
    OPTIMISE = "OPTIMISE"
    VERIFY = "VERIFY"
    ACCEPT_OR_ROLLBACK = "ACCEPT_OR_ROLLBACK"
    REPORT = "REPORT"


class RecipeRunStatus(str, Enum):
    RUNNING = "RUNNING"
    WAITING_FOR_OPERATOR = "WAITING_FOR_OPERATOR"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RecipeStep:
    step_id: str
    title: str
    kind: StepKind
    instructions: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecipeDefinition:
    name: str
    description: str
    prerequisites: tuple[CalibrationKey, ...]
    steps: tuple[RecipeStep, ...]


@dataclass
class RecipeRun:
    run_id: str
    recipe_name: str
    state_revision_started: int
    parameters: dict[str, Any]
    step_index: int = 0
    status: RecipeRunStatus = RecipeRunStatus.RUNNING
    completed: list[dict[str, Any]] = field(default_factory=list)
    created_utc: str = field(default_factory=utc_now)
    updated_utc: str = field(default_factory=utc_now)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "recipe_name": self.recipe_name,
            "state_revision_started": self.state_revision_started,
            "parameters": self.parameters,
            "step_index": self.step_index,
            "status": self.status.value,
            "completed": self.completed,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RecipeRun":
        if int(data.get("schema_version", 1)) != 1:
            raise ValueError("Unsupported recipe-run schema.")
        return cls(
            run_id=data["run_id"],
            recipe_name=data["recipe_name"],
            state_revision_started=int(data["state_revision_started"]),
            parameters=dict(data.get("parameters", {})),
            step_index=int(data.get("step_index", 0)),
            status=RecipeRunStatus(data.get("status", "RUNNING")),
            completed=list(data.get("completed", [])),
            created_utc=data.get("created_utc", utc_now()),
            updated_utc=data.get("updated_utc", utc_now()),
            error=data.get("error"),
        )


class RecipePrerequisiteError(RuntimeError):
    pass


class RecipeEngine:
    def __init__(
        self,
        store: ExperimentStore,
        calibration: CalibrationRegistry,
        definitions: Iterable[RecipeDefinition] | None = None,
    ):
        self.store = store
        self.calibration = calibration
        self.definitions = {
            definition.name: definition
            for definition in (definitions or builtin_recipes().values())
        }

    def start(self, recipe_name: str, parameters: dict[str, Any] | None = None) -> RecipeRun:
        definition = self.definitions.get(recipe_name)
        if definition is None:
            raise KeyError(f"Unknown recipe {recipe_name!r}")
        missing = [
            key.value
            for key in definition.prerequisites
            if self.calibration.status(key) != CalibrationStatus.VALID
        ]
        if missing:
            raise RecipePrerequisiteError(
                f"{recipe_name} prerequisites are not VALID: {', '.join(missing)}"
            )
        snapshot = self.store.snapshot()
        run = RecipeRun(
            run_id=f"recipe-{recipe_name}-{uuid.uuid4().hex[:10]}",
            recipe_name=recipe_name,
            state_revision_started=snapshot.revision,
            parameters=dict(parameters or {}),
        )
        self._publish(run)
        return run

    def definition(self, run: RecipeRun) -> RecipeDefinition:
        if run.recipe_name not in self.definitions:
            raise KeyError(f"Recipe definition {run.recipe_name!r} is unavailable.")
        return self.definitions[run.recipe_name]

    def current_step(self, run: RecipeRun) -> RecipeStep | None:
        definition = self.definition(run)
        return definition.steps[run.step_index] if run.step_index < len(definition.steps) else None

    def complete_step(self, run: RecipeRun, result: dict[str, Any] | None = None) -> RecipeStep | None:
        if run.status == RecipeRunStatus.COMPLETE:
            return None
        if run.status == RecipeRunStatus.FAILED:
            raise ValueError("Cannot advance a failed recipe run.")
        step = self.current_step(run)
        if step is None:
            run.status = RecipeRunStatus.COMPLETE
            return None
        run.completed.append(
            {
                "step_id": step.step_id,
                "kind": step.kind.value,
                "completed_utc": utc_now(),
                "result": dict(result or {}),
            }
        )
        run.step_index += 1
        run.updated_utc = utc_now()
        next_step = self.current_step(run)
        run.status = (
            RecipeRunStatus.COMPLETE
            if next_step is None
            else RecipeRunStatus.WAITING_FOR_OPERATOR
            if next_step.kind == StepKind.GUIDED_MOVE
            else RecipeRunStatus.RUNNING
        )
        self._publish(run)
        return next_step

    def fail(self, run: RecipeRun, error: str) -> None:
        run.status = RecipeRunStatus.FAILED
        run.error = str(error)
        run.updated_utc = utc_now()
        self._publish(run)

    def _publish(self, run: RecipeRun) -> None:
        step = self.current_step(run)
        self.store.update(
            lambda state: (
                setattr(state.session, "recipe", run.recipe_name),
                setattr(state.session, "recipe_run_id", run.run_id),
                setattr(state.session, "current_step", step.step_id if step else None),
            ),
            source="recipe_engine",
            reason=f"Recipe {run.recipe_name}: {run.status.value}",
        )

    @staticmethod
    def save(run: RecipeRun, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @staticmethod
    def load(path: str | Path) -> RecipeRun:
        return RecipeRun.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _steps(*values: tuple[str, str, StepKind, str, dict[str, Any] | None]) -> tuple[RecipeStep, ...]:
    return tuple(
        RecipeStep(step_id, title, kind, instructions, parameters or {})
        for step_id, title, kind, instructions, parameters in values
    )


def builtin_recipes() -> dict[str, RecipeDefinition]:
    camera_ready = (
        CalibrationKey.CAMERA_PIXEL_SCALE,
        CalibrationKey.CAMERA_SENSOR_GEOMETRY,
    )
    phase_camera_ready = (CalibrationKey.SLM_PHASE_PATH,) + camera_ready
    definitions = [
        RecipeDefinition(
            "measure_beam_walk",
            "Acquire repeats at physical z coordinates and fit beam-camera relative walk.",
            camera_ready,
            _steps(
                ("configure", "Configure z planes", StepKind.CONFIGURE, "Confirm z reference, planes and repeats.", None),
                ("capture", "Guided z captures", StepKind.GUIDED_MOVE, "Move/confirm each plane, then capture fresh quantitative frames.", None),
                ("analyse", "Fit propagation", StepKind.ANALYSE, "Fit x(z), y(z), repeat scatter and relative slopes.", None),
                ("report", "Save report", StepKind.REPORT, "Save plots, metrics and provenance.", None),
            ),
        ),
        RecipeDefinition(
            "z_stack",
            "Reusable quantitative z-stack acquisition.",
            camera_ready,
            _steps(
                ("configure", "Configure stack", StepKind.CONFIGURE, "Choose z reference, range, planes and repeats.", None),
                ("capture", "Capture stack", StepKind.GUIDED_MOVE, "Move or prompt at every z plane and acquire fresh frames.", None),
                ("report", "Save stack", StepKind.REPORT, "Seal numeric frames, state and report.", None),
            ),
        ),
        RecipeDefinition(
            "vortex_charge_scan",
            "Scan arbitrary integer vortex charges using the shared phase engine.",
            phase_camera_ready,
            _steps(
                ("configure", "Choose charges", StepKind.CONFIGURE, "Record panel, charges, repeats and fixed camera settings.", None),
                ("capture", "Scan charges", StepKind.CAPTURE, "Cast and capture every requested charge.", None),
                ("analyse", "Compare families", StepKind.ANALYSE, "Compare state-aware beam metrics across charges.", None),
                ("report", "Save report", StepKind.REPORT, "Write provenance-linked results.", None),
            ),
        ),
        RecipeDefinition(
            "correction_gain_scan",
            "Scan retrieved-map gain without duplicating carrier or correction layers.",
            phase_camera_ready,
            _steps(
                ("configure", "Choose gain values", StepKind.CONFIGURE, "Confirm correction map and gain range.", None),
                ("capture", "Capture gains", StepKind.CAPTURE, "Cast each complete phase and capture repeats.", None),
                ("verify", "Fresh verification", StepKind.VERIFY, "Recapture the recommended gain independently.", None),
                ("decide", "Accept or roll back", StepKind.ACCEPT_OR_ROLLBACK, "Accept only verified improvement.", None),
            ),
        ),
        RecipeDefinition(
            "optimise_zernike",
            "Signed, drift-bracketed low-order command optimisation.",
            phase_camera_ready,
            _steps(
                ("configure", "Choose mode and sweep", StepKind.CONFIGURE, "Choose a supported command mode and signed deltas.", None),
                ("optimise", "Run candidate sweep", StepKind.OPTIMISE, "Randomised candidates bracketed by start/end controls.", None),
                ("verify", "Fresh verification", StepKind.VERIFY, "Acquire fresh repeats at the recommendation.", None),
                ("decide", "Accept or roll back", StepKind.ACCEPT_OR_ROLLBACK, "Retain only a verified correction command.", None),
            ),
        ),
        RecipeDefinition(
            "gaussian_characterisation",
            "q=0 central-beam characterisation.",
            phase_camera_ready,
            _steps(
                ("configure_q0", "Configure q=0", StepKind.CONFIGURE, "Disable vortex contributions while preserving other phase layers.", None),
                ("capture", "Capture central beam", StepKind.CAPTURE, "Capture quantitative repeats at configured planes.", None),
                ("analyse", "Central-beam metrics", StepKind.ANALYSE, "Calculate centre, width, ellipticity and propagation.", None),
                ("report", "Save report", StepKind.REPORT, "Write provenance-linked characterisation.", None),
            ),
        ),
        RecipeDefinition(
            "commission_q20",
            "Full q=20 reference commissioning built from reusable steps.",
            phase_camera_ready + (CalibrationKey.CAMERA_TRAVEL_AXIS,),
            _steps(
                ("q0_dense", "Dense q=0 reference", StepKind.GUIDED_MOVE, "Capture a dense q=0 propagation reference.", None),
                ("q20_dense", "Dense q=20 reference", StepKind.GUIDED_MOVE, "Capture the same planes at q=20.", None),
                ("optimise", "Low-order correction", StepKind.OPTIMISE, "Run configured low-order command sweeps.", None),
                ("verify", "Independent verification", StepKind.VERIFY, "Recapture dense verification data.", None),
                ("accept", "Create golden reference", StepKind.ACCEPT_OR_ROLLBACK, "Accept or restore; update calibration only from evidence.", None),
                ("report", "Commissioning report", StepKind.REPORT, "Write the complete commissioning record.", None),
            ),
        ),
        RecipeDefinition(
            "recover_q20",
            "Quick q=20 recovery after routine axicon disturbance.",
            phase_camera_ready,
            _steps(
                ("configure", "Confirm recovery geometry", StepKind.CONFIGURE, "Confirm physical axicon IN, z reference, planes and repeats.", {"default_z_mm": [31.0, 38.0, 45.0], "charge": 20}),
                ("q0_walk", "Measure q=0 walk", StepKind.GUIDED_MOVE, "Capture identical near/middle/far q=0 planes.", None),
                ("q20_walk", "Measure q=20 walk", StepKind.GUIDED_MOVE, "Capture identical planes with q=20 on SLM1.", None),
                ("compare", "Compare walk", StepKind.ANALYSE, "Separate common propagation mismatch from possible vortex/SLM asymmetry.", None),
                ("optimise", "Low-order command sweep", StepKind.OPTIMISE, "Sweep configured modes; do not infer physical aberration coefficients.", {"default_modes": ["astig_x", "coma_y", "coma_x", "astig_xy"]}),
                ("verify", "Fresh verification", StepKind.VERIFY, "Acquire new data at the recommended command.", None),
                ("decide", "Accept or roll back", StepKind.ACCEPT_OR_ROLLBACK, "Accept only if verification passes; otherwise restore baseline.", None),
                ("readiness", "Update readiness", StepKind.REPORT, "Update calibration evidence and readiness without fake confidence.", None),
            ),
        ),
    ]
    return {definition.name: definition for definition in definitions}
