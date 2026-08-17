"""Shared editable-control helpers for study notebooks.

The notebooks in this repository have two different roles:

* locked stage notebooks, which regenerate canonical outputs and should not hide
  fail/marginal QA labels; and
* exploratory notebook sessions, where the user may change parameters to inspect
  qualitative and metric changes.

This module provides a small, explicit control object that notebooks can import
at the top.  It does not change the locked physics by itself; it standardises
where editable parameters live and records whether a notebook is being used for
quick exploration, balanced inspection, or publication-grade regeneration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import pandas as pd

ALLOWED_RUN_MODES = ("quick_preview", "balanced", "publication", "custom")

STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "overview": {"run_mode": "quick_preview", "save_outputs": False},
    "scalar": {"run_mode": "balanced", "save_outputs": False, "ell": 3, "target_core_diameter_um": 3.0, "target_bessel_length_um": 150.0},
    "lab_realism": {"run_mode": "balanced", "save_outputs": False, "ell": 3, "target_core_diameter_um": 3.0, "target_bessel_length_um": 150.0, "objective_NA": 0.45, "blaze_period_px": 20},
    "vector": {"run_mode": "balanced", "save_outputs": False, "ell": 1, "vector_mode": "radial_reference"},
    "materials": {"run_mode": "balanced", "save_outputs": False, "pulse_energy_uJ": 10.0, "threshold_fluence_J_cm2": 1.0, "pulse_count": 1},
    "capsule": {"run_mode": "balanced", "save_outputs": False, "target_width_um": 4.2, "target_length_um": 150.0},
    "advanced": {"run_mode": "balanced", "save_outputs": False, "symmetry_order": 6, "propagation_tested": True},
    "publication_exports": {"run_mode": "publication", "save_outputs": True},
    "quicklook": {"run_mode": "balanced", "save_outputs": False, "ell": 3, "vortex_phase_on": True, "flatten_phase_before_axicon": False},
}


@dataclass(frozen=True)
class NotebookControls:
    """Human-editable notebook control block.

    These fields are deliberately broad and optional.  Individual notebooks may
    use only the subset relevant to that study.  The important thing is that the
    editable values are visible near the top of the notebook and that the run
    mode/save behaviour is not hidden inside the body of the study.
    """

    stage: str
    run_mode: str = "balanced"
    save_outputs: bool = False
    use_canonical_outputs: bool = True
    allow_publication_export: bool = False
    notes: str = "Edit this object or make a copy for exploratory runs; do not hide QA labels."
    parameters: dict[str, Any] | None = None

    def validate(self) -> None:
        if self.run_mode not in ALLOWED_RUN_MODES:
            raise ValueError(f"run_mode must be one of {ALLOWED_RUN_MODES!r}")
        if self.stage not in STAGE_DEFAULTS:
            # New stages are allowed, but they should still be explicit.
            if not self.stage:
                raise ValueError("stage must be a non-empty string")

    def with_updates(self, **updates: Any) -> "NotebookControls":
        values = asdict(self)
        params = dict(values.pop("parameters") or {})
        for key, value in updates.items():
            if key in values:
                values[key] = value
            else:
                params[key] = value
        return replace(self, **values, parameters=params)

    def to_flat_dict(self) -> dict[str, Any]:
        values = asdict(self)
        params = values.pop("parameters") or {}
        return {**values, **params}

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for key, value in self.to_flat_dict().items():
            rows.append({"control": key, "value": value})
        return pd.DataFrame(rows)


def make_notebook_controls(stage: str, **updates: Any) -> NotebookControls:
    """Return a validated editable-control object for a notebook stage."""

    stage_key = str(stage).strip().lower()
    defaults = dict(STAGE_DEFAULTS.get(stage_key, {}))
    defaults.update(updates)
    run_mode = defaults.pop("run_mode", defaults.pop("computational_mode", "balanced"))
    save_outputs = bool(defaults.pop("save_outputs", False))
    allow_publication_export = bool(defaults.pop("allow_publication_export", stage_key == "publication_exports"))
    use_canonical_outputs = bool(defaults.pop("use_canonical_outputs", True))
    notes = str(defaults.pop("notes", "Edit for exploration; keep QA labels/caveats visible."))
    controls = NotebookControls(
        stage=stage_key,
        run_mode=str(run_mode),
        save_outputs=save_outputs,
        use_canonical_outputs=use_canonical_outputs,
        allow_publication_export=allow_publication_export,
        notes=notes,
        parameters=defaults,
    )
    controls.validate()
    return controls


def controls_from_mapping(stage: str, mapping: Mapping[str, Any]) -> NotebookControls:
    """Create notebook controls from a dictionary-like object."""

    return make_notebook_controls(stage, **dict(mapping))


def describe_controls(controls: NotebookControls) -> pd.DataFrame:
    """Return a compact DataFrame suitable for display in notebooks."""

    controls.validate()
    return controls.summary_frame()


__all__ = [
    "ALLOWED_RUN_MODES",
    "STAGE_DEFAULTS",
    "NotebookControls",
    "controls_from_mapping",
    "describe_controls",
    "make_notebook_controls",
]
