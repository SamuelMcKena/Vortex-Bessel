"""Deterministic and Monte Carlo propagation from supplied calibration uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, tan
from typing import Any, Callable, Mapping

import numpy as np

from vbb_study.calibration.schema import CalibrationBundle, uncertainty_at, value_at


@dataclass(frozen=True)
class UncertaintyConfig:
    samples: int = 1000
    random_seed: int = 12345
    distribution: str = "normal"
    confidence_level: float = 0.95


_PARAMETER_PATHS = (
    "laser.wavelength_m",
    "laser.pulse_energy_J",
    "laser.beam_radius_on_slm_m",
    "fourier_filter.focal_length_m",
    "objective.numerical_aperture",
    "axicon.base_angle_deg",
    "axicon.refractive_index",
    "camera.object_plane_scale_m_per_pixel",
    "transmissions.slm1",
    "transmissions.slm2",
    "transmissions.four_f",
    "transmissions.objective",
    "transmissions.interface",
)


def _nominal_parameters(bundle: CalibrationBundle) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for path in _PARAMETER_PATHS:
        value = value_at(bundle, path)
        result[path] = None if value is None else float(value)
    return result


def _metric_functions(reference: Mapping[str, Any]) -> dict[str, tuple[tuple[str, ...], Callable[[Mapping[str, float]], float]]]:
    wavelength_ref = float(reference.get("reference_wavelength_m", 1.029e-6))
    na_ref = float(reference.get("reference_objective_NA", 0.45))
    carrier = float(reference.get("carrier_frequency_cpm", 6250.0))
    feature_ref = reference.get("feature_radius_m")
    first_order = float(reference.get("first_order_efficiency", 0.9493536222723651))
    input_aperture = float(reference.get("input_aperture_fraction", 1.0))
    objective_pupil = float(reference.get("objective_pupil_fraction", 1.0))
    pixel_ref = float(reference.get("reference_output_pixel_m", 0.2058e-6))
    shape_ref = reference.get("peak_shape_factor_m_inv2")

    metrics: dict[str, tuple[tuple[str, ...], Callable[[Mapping[str, float]], float]]] = {
        "physical_pixel_scale_m_per_pixel": (
            ("camera.object_plane_scale_m_per_pixel",),
            lambda p: p["camera.object_plane_scale_m_per_pixel"],
        ),
        "first_order_position_m": (
            ("laser.wavelength_m", "fourier_filter.focal_length_m"),
            lambda p: p["laser.wavelength_m"] * p["fourier_filter.focal_length_m"] * carrier,
        ),
        "bessel_zone_length_m": (
            ("laser.beam_radius_on_slm_m", "axicon.base_angle_deg", "axicon.refractive_index"),
            lambda p: p["laser.beam_radius_on_slm_m"]
            / ((p["axicon.refractive_index"] - 1.0) * tan(np.deg2rad(p["axicon.base_angle_deg"]))),
        ),
    }
    if feature_ref is not None:
        metrics["feature_or_ring_radius_m"] = (
            ("laser.wavelength_m", "objective.numerical_aperture"),
            lambda p: float(feature_ref)
            * (p["laser.wavelength_m"] / wavelength_ref)
            * (na_ref / p["objective.numerical_aperture"]),
        )

    energy_dependencies = (
        "laser.pulse_energy_J", "transmissions.slm1", "transmissions.slm2",
        "transmissions.four_f", "transmissions.objective", "transmissions.interface",
    )
    stages = (
        ("energy_laser_J", ()),
        ("energy_after_input_aperture_J", ("input_aperture",)),
        ("energy_after_slm1_J", ("input_aperture", "transmissions.slm1")),
        ("energy_after_slm2_J", ("input_aperture", "transmissions.slm1", "transmissions.slm2")),
        ("energy_after_first_order_J", ("input_aperture", "transmissions.slm1", "transmissions.slm2", "first_order")),
        ("energy_after_four_f_J", ("input_aperture", "transmissions.slm1", "transmissions.slm2", "first_order", "transmissions.four_f")),
        ("energy_after_objective_pupil_J", ("input_aperture", "transmissions.slm1", "transmissions.slm2", "first_order", "transmissions.four_f", "objective_pupil")),
        ("energy_after_objective_J", ("input_aperture", "transmissions.slm1", "transmissions.slm2", "first_order", "transmissions.four_f", "objective_pupil", "transmissions.objective")),
        ("energy_at_material_J", ("input_aperture", "transmissions.slm1", "transmissions.slm2", "first_order", "transmissions.four_f", "objective_pupil", "transmissions.objective", "transmissions.interface")),
    )
    for name, factors in stages:
        dependencies = tuple(path for path in energy_dependencies if path in factors or path == "laser.pulse_energy_J")

        def energy(p: Mapping[str, float], factors: tuple[str, ...] = factors) -> float:
            result = p["laser.pulse_energy_J"]
            for factor in factors:
                if factor == "first_order":
                    result *= first_order
                elif factor == "input_aperture":
                    result *= input_aperture
                elif factor == "objective_pupil":
                    result *= objective_pupil
                else:
                    result *= p[factor]
            return result

        metrics[name] = (dependencies, energy)

    if shape_ref is not None:
        dependencies = energy_dependencies + ("camera.object_plane_scale_m_per_pixel",)

        def peak_fluence(p: Mapping[str, float]) -> float:
            energy = p["laser.pulse_energy_J"] * input_aperture * first_order * objective_pupil
            for factor in ("transmissions.slm1", "transmissions.slm2", "transmissions.four_f", "transmissions.objective", "transmissions.interface"):
                energy *= p[factor]
            scale_ratio = p["camera.object_plane_scale_m_per_pixel"] / pixel_ref
            return energy * float(shape_ref) / (scale_ratio * scale_ratio) / 1.0e4

        metrics["peak_fluence_J_cm2"] = (dependencies, peak_fluence)
    return metrics


def _unavailable(status: str, missing: tuple[str, ...]) -> dict[str, Any]:
    return {
        "uncertainty_status": status,
        "nominal": None,
        "standard_uncertainty": None,
        "lower_95": None,
        "upper_95": None,
        "samples": 0,
        "failed_samples": 0,
        "missing_calibration": list(missing),
        "parameter_contributions": {},
    }


def propagate_calibration_uncertainty(
    request: Any,
    bundle: CalibrationBundle,
    config: UncertaintyConfig,
    *,
    reference_metrics: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Propagate only supplied standard uncertainties through bounded optical metrics."""

    if config.samples < 0:
        raise ValueError("uncertainty sample count must be non-negative")
    if config.distribution != "normal":
        raise ValueError("only normal supplied-uncertainty propagation is currently supported")
    if not 0.0 < config.confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")

    reference = dict(reference_metrics or {})
    parameters = _nominal_parameters(bundle)
    functions = _metric_functions(reference)
    uncertain = {
        path: float(uncertainty_at(bundle, path))
        for path in _PARAMETER_PATHS
        if uncertainty_at(bundle, path) is not None
    }
    rng = np.random.default_rng(config.random_seed)
    draws: dict[str, np.ndarray] = {}
    for path, nominal in parameters.items():
        if nominal is None:
            continue
        supplied_u = uncertain.get(path)
        if config.samples > 0 and supplied_u is not None:
            draws[path] = rng.normal(nominal, supplied_u, size=config.samples)
        elif config.samples > 0:
            draws[path] = np.full(config.samples, nominal, dtype=float)

    alpha = (1.0 - config.confidence_level) / 2.0
    result: dict[str, dict[str, Any]] = {}
    for metric, (dependencies, function) in functions.items():
        missing = tuple(path for path in dependencies if parameters.get(path) is None)
        if missing:
            result[metric] = _unavailable("unavailable_missing_calibration", missing)
            continue
        nominal_parameters = {path: float(value) for path, value in parameters.items() if value is not None}
        try:
            nominal = float(function(nominal_parameters))
        except (ArithmeticError, ValueError, FloatingPointError):
            result[metric] = _unavailable("unavailable_invalid_nominal", ())
            continue

        values: list[float] = []
        if config.samples > 0:
            for index in range(config.samples):
                sample = dict(nominal_parameters)
                for path in dependencies:
                    sample[path] = float(draws[path][index])
                try:
                    value = float(function(sample))
                except (ArithmeticError, ValueError, FloatingPointError):
                    continue
                if np.isfinite(value) and value >= 0.0:
                    values.append(value)
        array = np.asarray(values, dtype=float)
        failed = config.samples - int(array.size)
        if array.size:
            standard = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
            lower, upper = (float(value) for value in np.quantile(array, [alpha, 1.0 - alpha]))
            lower = min(lower, nominal)
            upper = max(upper, nominal)
        else:
            standard = 0.0
            lower = upper = nominal

        raw_contributions: dict[str, float] = {}
        for path in dependencies:
            supplied_u = uncertain.get(path)
            if supplied_u is None or supplied_u == 0.0:
                continue
            plus = dict(nominal_parameters)
            minus = dict(nominal_parameters)
            plus[path] += supplied_u
            minus[path] -= supplied_u
            try:
                effect = 0.5 * (float(function(plus)) - float(function(minus)))
            except (ArithmeticError, ValueError, FloatingPointError):
                continue
            raw_contributions[path] = effect * effect
        total = sum(raw_contributions.values())
        contributions = {
            path: value / total for path, value in raw_contributions.items()
        } if total > 0.0 else {}
        result[metric] = {
            "uncertainty_status": "available_from_supplied_uncertainty",
            "nominal": nominal,
            "standard_uncertainty": standard,
            "lower_95": lower,
            "upper_95": upper,
            "samples": config.samples,
            "failed_samples": failed,
            "missing_calibration": [],
            "parameter_contributions": contributions,
        }

    if getattr(request, "beam_case", "") == "H1":
        result["H1_dominant_feature_radius_m"] = dict(
            result.get("feature_or_ring_radius_m", _unavailable("unavailable_missing_calibration", ()))
        )
        result["H1_edge_sharpness_mm_inv"] = _unavailable(
            "unavailable_numerical_stability_not_demonstrated",
            (),
        )
    return result
