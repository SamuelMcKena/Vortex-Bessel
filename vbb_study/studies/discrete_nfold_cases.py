"""Stage 8 discrete N-fold case registry."""

from __future__ import annotations

from typing import Any


def discrete_nfold_stage8_cases() -> list[dict[str, Any]]:
    """Return canonical discrete N-fold patterns and route labels."""

    base_patterns = [
        ("triangular", 3, 0),
        ("square", 4, 0),
        ("hexagonal", 6, 0),
        ("octagonal", 8, 0),
        ("dodecagonal", 12, 0),
        ("hexagonal_vortex", 6, 1),
    ]
    rows: list[dict[str, Any]] = []
    for name, order, charge in base_patterns:
        rows.append(
            {
                "case_id": f"{name}_ideal_discrete_superposition",
                "preset": name,
                "path": "ideal",
                "beam_family": "nfold_vortex_ring" if charge else "discrete_nfold",
                "model_level": "numerical_propagation",
                "generation_method": "discrete_superposition",
                "target_symmetry_order": order,
                "target_polygon_sides": order,
                "ell": charge,
                "phase_only_compatible": False,
                "complex_amplitude_required": True,
                "propagation_tested": True,
            }
        )
        rows.append(
            {
                "case_id": f"{name}_phase_only_approximation",
                "preset": name,
                "path": "phase_only_proxy",
                "beam_family": "nfold_vortex_ring" if charge else "discrete_nfold",
                "model_level": "lab_realistic",
                "generation_method": "phase_only_slm",
                "target_symmetry_order": order,
                "target_polygon_sides": order,
                "ell": charge,
                "phase_only_compatible": True,
                "complex_amplitude_required": False,
                "propagation_tested": True,
            }
        )
    return rows


__all__ = ["discrete_nfold_stage8_cases"]
