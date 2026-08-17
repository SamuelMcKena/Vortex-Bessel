"""Plane-tagged length helpers for the VBB publication study.

The scalar study has two natural transverse scales: millimetres before the
objective and microns at the focused writing plane.  These helpers make that
boundary explicit without forcing every legacy float field to migrate at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

Plane = Literal["slm", "free_space", "pupil", "surface", "sample"]
LengthUnit = Literal["m", "mm", "um", "nm"]

_UNIT_TO_M = {"m": 1.0, "mm": 1.0e-3, "um": 1.0e-6, "nm": 1.0e-9}


@dataclass(frozen=True)
class Length:
    """A length that always knows its plane and unit."""

    value: float
    unit: LengthUnit
    plane: Plane

    def to_m(self) -> float:
        return float(self.value) * _UNIT_TO_M[self.unit]


@dataclass(frozen=True)
class ObjectiveMap:
    """Links pre-objective transverse lengths to focused surface/sample lengths.

    ``demag`` is the focused-plane size divided by the pre-objective size.  For
    the beam study the focused plane is the air-side sample surface.  Through-
    sample studies can still use the same object when explicitly converting to
    an in-medium sample plane.
    """

    demag: float
    n_sample: float = 2.44
    source: str = "effective_objective_relay"
    mapping_mode: str = "fixed_physical_optics"

    def __post_init__(self) -> None:
        if float(self.demag) <= 0.0:
            raise ValueError("ObjectiveMap.demag must be positive.")

    def pre_to_sample_m(self, x_pre_m: float) -> float:
        return float(x_pre_m) * float(self.demag)

    def sample_to_pre_m(self, x_sample_m: float) -> float:
        return float(x_sample_m) / float(self.demag)

    def sample_to_pre_spatial_frequency_m_inv(self, k_sample_m_inv: float) -> float:
        """Convert transverse spatial frequency from sample to pre-objective."""

        return float(k_sample_m_inv) * float(self.demag)

    def pre_to_sample_spatial_frequency_m_inv(self, k_pre_m_inv: float) -> float:
        """Convert transverse spatial frequency from pre-objective to sample."""

        return float(k_pre_m_inv) / float(self.demag)

    def convert_length(self, length: Length, target_plane: Plane, unit: LengthUnit = "m") -> Length:
        """Convert a transverse length between pre-objective and sample planes."""

        source_plane = length.plane
        value_m = length.to_m()
        pre_planes = {"slm", "free_space", "pupil"}
        focused_planes = {"surface", "sample"}
        if source_plane == target_plane:
            converted_m = value_m
        elif source_plane in pre_planes and target_plane in focused_planes:
            converted_m = self.pre_to_sample_m(value_m)
        elif source_plane in focused_planes and target_plane in pre_planes:
            converted_m = self.sample_to_pre_m(value_m)
        else:
            raise ValueError(f"Unsupported plane conversion: {source_plane!r} -> {target_plane!r}")
        return Length(converted_m / _UNIT_TO_M[unit], unit, target_plane)


def length_from_m(value_m: float, *, plane: Plane, unit: LengthUnit) -> Length:
    return Length(float(value_m) / _UNIT_TO_M[unit], unit, plane)


def objective_map_from_waists(
    *,
    pre_objective_radius_m: float,
    sample_radius_m: float,
    n_sample: float = 2.44,
    source: str = "target_matched_bessel_gauss_waist_audit",
    mapping_mode: str = "target_matched_inverse_design",
) -> ObjectiveMap:
    """Build the target-matched waist-ratio audit map."""

    return ObjectiveMap(
        demag=float(sample_radius_m) / max(float(pre_objective_radius_m), 1.0e-30),
        n_sample=float(n_sample),
        source=source,
        mapping_mode=mapping_mode,
    )


def objective_map_from_optics(
    *,
    objective_f_eff_m: float,
    effective_relay_f_m: float,
    n_sample: float = 2.44,
    source: str = "objective_f_eff_over_effective_relay_f",
    mapping_mode: str = "fixed_physical_optics",
) -> ObjectiveMap:
    """Build the optics-derived focused-plane demag from objective/relay data."""

    return ObjectiveMap(
        demag=float(objective_f_eff_m) / max(float(effective_relay_f_m), 1.0e-30),
        n_sample=float(n_sample),
        source=source,
        mapping_mode=mapping_mode,
    )


def headline_lengths(config: Any) -> dict[str, Length]:
    """Return the headline configuration lengths with explicit planes."""

    lengths = {
        "target_core_diameter": length_from_m(config.target.target_core_diameter_m, unit="um", plane="surface"),
        "target_bessel_length": length_from_m(config.target.target_bessel_length_m, unit="um", plane="surface"),
        "surface_z": length_from_m(0.0 if getattr(config, "surface_z_m", None) is None else config.surface_z_m, unit="um", plane="surface"),
        "material_write_depth": length_from_m(config.material.write_depth_m, unit="um", plane="sample"),
        "beam_radius_on_slm": length_from_m(config.laser.beam_radius_on_slm_m, unit="mm", plane="slm"),
        "slm_pixel_pitch": length_from_m(config.slm.pixel_pitch_m, unit="um", plane="slm"),
        "objective_pupil_radius": length_from_m(config.objective.pupil_radius_m, unit="mm", plane="pupil"),
    }
    physical = getattr(config, "physical_axicon", None)
    if physical is not None:
        lengths["inter_slm_distance"] = length_from_m(getattr(physical, "inter_slm_z_m", 0.0), unit="mm", plane="free_space")
        aperture = getattr(physical, "axicon_aperture_radius_m", None)
        if aperture is not None:
            lengths["physical_axicon_aperture_radius"] = length_from_m(aperture, unit="mm", plane="free_space")
    return lengths


def headline_lengths_jsonable(config: Any) -> dict[str, Mapping[str, float | str]]:
    return {
        key: {"value": length.value, "unit": length.unit, "plane": length.plane, "value_m": length.to_m()}
        for key, length in headline_lengths(config).items()
    }


__all__ = [
    "Length",
    "LengthUnit",
    "ObjectiveMap",
    "Plane",
    "headline_lengths",
    "headline_lengths_jsonable",
    "length_from_m",
    "objective_map_from_waists",
    "objective_map_from_optics",
]
