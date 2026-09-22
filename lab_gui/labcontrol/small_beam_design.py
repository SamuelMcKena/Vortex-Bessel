"""Adapter to the existing sample-plane inverse designer (not a bench solver)."""
from __future__ import annotations

from dataclasses import asdict
import math
import numpy as np
from scipy.special import jnp_zeros, jv
from vbb_study.config import LaserConfig, MaterialConfig, BeamTarget
from vbb_study.design import compute_design_from_targets, J0_FIRST_ZERO


def design_small_beam(*, diameter_um=3.0, length_um=150.0, charge=0,
                      wavelength_nm=1030.0, input_radius_mm=2.0,
                      medium_index=1.0, diameter_definition="actual"):
    values = (diameter_um, length_um, wavelength_nm, input_radius_mm, medium_index)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Size, length, wavelength, input radius and refractive index must be finite and positive.")
    if charge != int(charge) or abs(charge) > 100:
        raise ValueError("Vortex charge must be an integer between -100 and 100.")
    if diameter_definition not in {"actual", "equivalent_q0"}:
        raise ValueError("Unknown diameter definition.")
    q = abs(int(charge))
    ring_factor = float(jnp_zeros(q, 1)[0] / J0_FIRST_ZERO) if q else 1.0
    equivalent_um = diameter_um / ring_factor if q and diameter_definition == "actual" else diameter_um
    laser = LaserConfig(wavelength_m=wavelength_nm * 1e-9,
                        beam_radius_on_slm_m=input_radius_mm * 1e-3)
    design = compute_design_from_targets(
        laser,
        BeamTarget(ell=int(charge), target_core_diameter_m=equivalent_um * 1e-6,
                   target_bessel_length_m=length_um * 1e-6),
        MaterialConfig(refractive_index=medium_index),
        mapping_mode="target_matched_inverse_design",
    )
    sine = design.kr_sample_m_inv / (laser.k0 * medium_index)
    if sine >= 1.0:
        raise ValueError("Requested diameter/charge requires k⊥ ≥ k in the chosen medium: no propagating Bessel cone. Increase diameter or reduce |q|.")
    result = asdict(design)
    result.update({
        "scope": "ideal sample-plane inverse-design reference; not current bench propagation or correction",
        "wavelength_nm": wavelength_nm, "medium_index": medium_index,
        "input_radius_mm": input_radius_mm,
        "requested_diameter_um": diameter_um, "diameter_definition": diameter_definition,
        "required_NA": design.kr_sample_m_inv / laser.k0,
        "cone_half_angle_deg": math.degrees(math.asin(sine)),
        "warning": ("High-angle design: scalar/paraxial length and relay mapping require vector/nonparaxial validation."
                    if sine > 0.3 else "Required mapping is inferred from targets, not a measured objective calibration."),
        "length_definition": "repository geometrical/paraxial reference w0*k/k_perp, NOT intensity FWHM",
    })
    return result


def ideal_radial_reference(design, samples=1200):
    q = abs(int(design["ell"]))
    kp = design["kr_sample_m_inv"]
    ring_r = design["vortex_main_ring_radius_m"] if q else J0_FIRST_ZERO / kp
    r = np.linspace(0.0, max(ring_r * 2.5, ring_r + 6 * math.pi / kp), samples)
    intensity = jv(q, kp * r) ** 2
    return r * 1e6, intensity / intensity.max()
