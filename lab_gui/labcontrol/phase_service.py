"""Single reusable entry point for complete dual-SLM phase generation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from slm_lab_control.phase import PhaseResult, compose_phase

from .state import ExperimentState


def phase_sha256(phase_rad: np.ndarray) -> str:
    """Hash the exact float32 wrapped-radians command sent by HEDS."""

    canonical = np.ascontiguousarray(np.mod(phase_rad, 2.0 * np.pi), dtype="<f4")
    canonical[canonical >= np.float32(2.0 * np.pi)] = 0.0
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@dataclass(frozen=True)
class PhaseBundle:
    """Generated complete phases and their canonical content hashes."""

    results: Mapping[str, PhaseResult]
    hashes: Mapping[str, str]
    source_revision: int

    def result(self, name: str) -> PhaseResult:
        return self.results[name.upper()]


class PhaseService:
    """Adapts ExperimentState to the proven v0.6 composer.

    No phase term is reimplemented here.  The adapter exists so every GUI,
    recipe and CLI client uses the same complete-phase path and one final wrap.
    """

    def generate(self, state: ExperimentState) -> PhaseBundle:
        state.validate()
        results = {
            "SLM1": compose_phase(state.slm1.phase),
            "SLM2": compose_phase(state.slm2.phase),
        }
        return PhaseBundle(
            results=results,
            hashes={name: phase_sha256(result.phase_rad) for name, result in results.items()},
            source_revision=state.revision,
        )
