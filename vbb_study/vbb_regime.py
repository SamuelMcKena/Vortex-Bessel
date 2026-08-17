"""Regime presets and sampling-validity guards for VBB studies."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any, Literal, Sequence
import warnings

import numpy as np
import pandas as pd

from vbb_study.config import BeamDesign, EPS, TWOPI, TwinConfig, um
from vbb_study.design import axial_scan_values, compute_design_from_config

RegimeName = Literal["general", "limits"]
ViolationAction = Literal["flag", "warn", "raise"]


@dataclass(frozen=True)
class RegimePreset:
    name: RegimeName
    nominal_core_um: float
    nominal_zone_um: float
    core_um_values: tuple[float, ...]
    zone_um_values: tuple[float, ...]
    description: str


GENERAL = RegimePreset(
    name="general",
    nominal_core_um=3.0,
    nominal_zone_um=150.0,
    core_um_values=(3.0, 4.0, 5.0),
    zone_um_values=(150.0, 200.0),
    description="Nominal publication regime: 3-5 um core and 150-200 um beam zone.",
)

LIMITS = RegimePreset(
    name="limits",
    nominal_core_um=2.0,
    nominal_zone_um=300.0,
    core_um_values=(0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 2.8),
    zone_um_values=(25.0, 50.0, 100.0, 200.0, 400.0, 800.0),
    description="Numerical and optical limits push: sub-3 um cores and extreme zones.",
)

REGIMES: dict[str, RegimePreset] = {GENERAL.name: GENERAL, LIMITS.name: LIMITS}


def regime_preset(regime: str | RegimePreset = "general") -> RegimePreset:
    if isinstance(regime, RegimePreset):
        return regime
    key = str(regime).lower().strip()
    if key not in REGIMES:
        raise ValueError(f"Unsupported regime: {regime!r}")
    return REGIMES[key]


def config_for_regime(
    config: TwinConfig,
    regime: str | RegimePreset = "general",
    *,
    core_um: float | None = None,
    zone_um: float | None = None,
) -> TwinConfig:
    preset = regime_preset(regime)
    return replace(
        config,
        regime=preset.name,
        target=replace(
            config.target,
            target_core_diameter_m=float(preset.nominal_core_um if core_um is None else core_um) * um,
            target_bessel_length_m=float(preset.nominal_zone_um if zone_um is None else zone_um) * um,
        ),
    )


def _actual_output_dx_m(config: TwinConfig, result: dict[str, Any] | None) -> float:
    if result is not None and "volume" in result:
        volume = result["volume"]
        peak_index = int(volume.get("peak_index", 0))
        output_dx = np.asarray(volume.get("output_dx_m", []), dtype=float)
        if output_dx.size:
            return float(output_dx[int(np.clip(peak_index, 0, output_dx.size - 1))])
        crop_grid = volume.get("crop_grid")
        if isinstance(crop_grid, dict) and "dx" in crop_grid:
            return float(crop_grid["dx"])
    dx_p = config.slm.pixel_pitch_m * max(1, int(config.grid.device_downsample))
    return float(config.laser.wavelength_m * config.objective.f_eff_m / max(config.grid.N * dx_p, EPS))


def sampling_validity(
    config: TwinConfig,
    design: BeamDesign | None = None,
    result: dict[str, Any] | None = None,
    *,
    min_samples_per_feature: float = 2.0,
    min_samples_per_radial_period: float = 2.0,
    min_axial_samples_per_zone: float = 4.0,
) -> dict[str, Any]:
    """Return a Nyquist-style validity report for the current regime/case."""

    design = design or compute_design_from_config(config)
    dx = _actual_output_dx_m(config, result)
    radial_period_m = TWOPI / max(abs(float(design.kr_sample_m_inv)), EPS)
    target_zone_m = float(design.target_bessel_length_m)
    if result is not None and "volume" in result:
        z = np.asarray(result["volume"].get("z", []), dtype=float)
        axial_dz_m = float(np.nanmedian(np.abs(np.diff(z)))) if z.size > 1 else np.nan
    else:
        z = axial_scan_values(config, design)
        axial_dz_m = float(np.nanmedian(np.diff(z))) if z.size > 1 else np.nan

    # Legacy feature scale: target_core_diameter_m is the equivalent ell=0
    # first-zero diameter. Vortex runs also report the realised ring diameter.
    samples_per_feature = float(design.target_core_diameter_m / max(dx, EPS))
    samples_per_radial_period = float(radial_period_m / max(dx, EPS))
    samples_per_axial_zone = float(target_zone_m / max(axial_dz_m, EPS)) if np.isfinite(axial_dz_m) else float("inf")
    violations: list[str] = []
    if samples_per_feature < float(min_samples_per_feature):
        violations.append("transverse_feature_nyquist")
    if samples_per_radial_period < float(min_samples_per_radial_period):
        violations.append("radial_period_nyquist")
    if samples_per_axial_zone < float(min_axial_samples_per_zone):
        violations.append("axial_zone_sampling")
    valid = not violations
    return {
        "regime": str(getattr(config, "regime", "general")),
        "valid": bool(valid),
        "violations": violations,
        "samples_per_feature": samples_per_feature,
        "samples_per_radial_period": samples_per_radial_period,
        "samples_per_axial_zone": samples_per_axial_zone,
        "output_dx_um": float(dx / um),
        "axial_dz_um": float(axial_dz_m / um) if np.isfinite(axial_dz_m) else np.nan,
        "min_samples_per_feature": float(min_samples_per_feature),
        "min_samples_per_radial_period": float(min_samples_per_radial_period),
        "min_axial_samples_per_zone": float(min_axial_samples_per_zone),
        "on_violation": str(getattr(config, "validity_on_violation", "flag")),
    }


def enforce_validity(report: dict[str, Any], on_violation: ViolationAction | None = None) -> dict[str, Any]:
    action = str(on_violation or report.get("on_violation", "flag")).lower().strip()
    if bool(report.get("valid", False)):
        report["validity_action"] = "pass"
        return report
    report["validity_action"] = action
    message = (
        f"{report.get('validity_name', 'Sampling validity')} failed: "
        f"{', '.join(report.get('violations', []))}"
    )
    if action == "raise":
        raise ValueError(message)
    if action == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return report
    if action != "flag":
        raise ValueError(f"Unsupported validity action: {action!r}")
    return report


def validity_map(
    config: TwinConfig,
    regime: str | RegimePreset = "limits",
    *,
    core_um_values: Sequence[float] | None = None,
    zone_um_values: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build a per-cell validity table for Stage-C QA maps."""

    preset = regime_preset(regime)
    cores = tuple(preset.core_um_values if core_um_values is None else core_um_values)
    zones = tuple(preset.zone_um_values if zone_um_values is None else zone_um_values)
    rows: list[dict[str, Any]] = []
    for core_um in cores:
        for zone_um in zones:
            cfg = config_for_regime(config, preset, core_um=float(core_um), zone_um=float(zone_um))
            design = compute_design_from_config(cfg)
            report = sampling_validity(cfg, design)
            rows.append(
                {
                    "regime": preset.name,
                    "target_core_diameter_um": float(core_um),
                    "target_bessel_length_um": float(zone_um),
                    "valid": bool(report["valid"]),
                    "violations": ",".join(report["violations"]),
                    "samples_per_feature": float(report["samples_per_feature"]),
                    "samples_per_radial_period": float(report["samples_per_radial_period"]),
                    "samples_per_axial_zone": float(report["samples_per_axial_zone"]),
                    "output_dx_um": float(report["output_dx_um"]),
                    "axial_dz_um": float(report["axial_dz_um"]),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "GENERAL",
    "LIMITS",
    "REGIMES",
    "RegimeName",
    "RegimePreset",
    "ViolationAction",
    "config_for_regime",
    "enforce_validity",
    "regime_preset",
    "sampling_validity",
    "validity_map",
]
