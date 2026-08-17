"""Stage 8 polygonal/hexagonal case registry."""

from __future__ import annotations

from typing import Any


def polygonal_stage8_cases() -> list[dict[str, Any]]:
    """Return the canonical Stage 8 hexagonal/polygonal optical cases."""

    return [
        {
            "case_id": "hexagonal_focal_plane_target",
            "beam_family": "hexagonal_polygonal",
            "model_level": "focal_plane_target",
            "generation_method": "amplitude_phase_target",
            "target_polygon_sides": 6,
            "target_symmetry_order": 6,
            "focal_plane_only": True,
            "description": "filled hexagonal focal-plane target mask",
        },
        {
            "case_id": "hollow_hexagonal_outline_target",
            "beam_family": "hollow_polygon",
            "model_level": "focal_plane_target",
            "generation_method": "amplitude_phase_target",
            "target_polygon_sides": 6,
            "target_symmetry_order": 6,
            "focal_plane_only": True,
            "description": "hollow regular-hexagon outline target",
        },
        {
            "case_id": "phase_only_polygonal_approximation",
            "beam_family": "hexagonal_polygonal",
            "model_level": "hardware_route",
            "generation_method": "phase_only_slm",
            "target_polygon_sides": 6,
            "target_symmetry_order": 6,
            "focal_plane_only": True,
            "phase_only_compatible": True,
            "description": "phase-only approximation scored at the focal plane only",
        },
        {
            "case_id": "propagation_tested_hollow_polygon_candidate",
            "beam_family": "hollow_polygon",
            "model_level": "numerical_propagation",
            "generation_method": "amplitude_phase_target",
            "target_polygon_sides": 6,
            "target_symmetry_order": 6,
            "focal_plane_only": False,
            "propagation_tested": True,
            "description": "simulation-only hollow outline propagated over z",
        },
    ]


__all__ = ["polygonal_stage8_cases"]
