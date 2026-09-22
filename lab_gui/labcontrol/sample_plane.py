"""Demagnifying objective after the axicon: the route to micron-scale beams.

The bench's 4F relay is 1:1, so the beam reaching the axicon is the beam on the
SLM and the ring size is fixed by the axicon's own k_perp.  A wider input beam
lengthens the Bessel region instead of shrinking the rings, which is why tiny
spots need a *demagnifying objective* after the axicon rather than a different
illumination.

An ideal telescope of transverse magnification M (M = 1/N for "1:N") scales the
whole post-axicon field exactly:

    ring radius     r  ->  M r
    cone angle      th ->  th / M          (k_perp -> k_perp / M)
    axial distance  z  ->  M^2 z           (so the Bessel region shortens by M^2)

The dimensionless field is unchanged: (k_perp * w) and (lambda z / w^2) are both
invariant, so the *pattern* at the sample is the simulated bench pattern with
new axis units.  This module converts between the two, which is why the
simulator needs no extra propagation to show a micron-scale beam.

What this does NOT model: the objective's finite pupil (it may not accept the
full axicon annulus), its aberrations, vector/high-NA effects once the cone
angle is large, and any aberration the imaging path adds when relaying the
sample plane back onto a camera.  Treat a high-NA result as a design target,
not as a prediction of that bench.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any


PARAXIAL_NA_LIMIT = 0.3
"""Above this sample-side NA the scalar paraxial scaling needs vector checking."""


@dataclass(frozen=True)
class SamplePlaneScale:
    """Bench-plane numbers expressed at the sample of a 1:N objective."""

    demagnification: float          # M, 1.0 when no objective is fitted
    ratio_label: str                # "1:1", "1:50"
    pixel_um: float                 # camera pixel projected to the sample
    native_sample_um: float         # model sample spacing projected to the sample
    ring_radius_um: float | None    # bright vortex ring (or q=0 first null)
    bessel_length_um: float | None
    k_perp_m_inv: float | None
    numerical_aperture: float | None
    cone_half_angle_deg: float | None
    z_scale: float                  # sample z = bench z * z_scale
    valid: bool
    message: str

    def sample_z_um(self, bench_z_mm: float) -> float:
        return float(bench_z_mm) * self.z_scale * 1e3

    def bench_z_mm(self, sample_z_um: float) -> float:
        return float(sample_z_um) * 1e-3 / self.z_scale

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_plane_scale(
    *,
    demagnification: float,
    camera_pixel_um: float,
    native_pixel_um: float | None = None,
    ring_radius_um: float | None = None,
    bessel_length_mm: float | None = None,
    k_perp_m_inv: float | None = None,
    wavelength_nm: float = 1030.0,
    medium_index: float = 1.0,
) -> SamplePlaneScale:
    """Project the simulated bench plane through an ideal 1:N objective."""

    m = float(demagnification)
    if not math.isfinite(m) or m <= 0.0 or m > 1.0:
        raise ValueError("Demagnification M must satisfy 0 < M <= 1 (M = 1/N for a 1:N objective).")
    if not math.isfinite(camera_pixel_um) or camera_pixel_um <= 0.0:
        raise ValueError("Camera pixel size must be finite and positive.")
    ratio = "1:1" if abs(m - 1.0) < 1e-12 else f"1:{1.0 / m:g}"
    k_sample = None if k_perp_m_inv is None else float(k_perp_m_inv) / m
    na = None
    cone_deg = None
    valid = True
    notes: list[str] = []
    if k_sample is not None:
        k0 = 2.0 * math.pi / (float(wavelength_nm) * 1e-9)
        na = k_sample / k0
        sine = k_sample / (k0 * float(medium_index))
        if sine >= 1.0:
            valid = False
            notes.append(
                f"1:{1.0 / m:g} needs k_perp >= k in this medium (NA {na:.2f}): no propagating cone. "
                "Use less demagnification, a weaker axicon or a higher-index medium."
            )
        else:
            cone_deg = math.degrees(math.asin(sine))
            if na > PARAXIAL_NA_LIMIT:
                notes.append(
                    f"NA {na:.2f} is beyond the scalar paraxial range: treat the length and ring "
                    "size as a design target needing vector validation."
                )
    if abs(m - 1.0) > 1e-12:
        notes.append("Objective pupil truncation and aberrations are not modelled.")
    return SamplePlaneScale(
        demagnification=m,
        ratio_label=ratio,
        pixel_um=float(camera_pixel_um) * m,
        native_sample_um=float(native_pixel_um if native_pixel_um else camera_pixel_um) * m,
        ring_radius_um=None if ring_radius_um is None else float(ring_radius_um) * m,
        bessel_length_um=None if bessel_length_mm is None else float(bessel_length_mm) * 1e3 * m * m,
        k_perp_m_inv=k_sample,
        numerical_aperture=na,
        cone_half_angle_deg=cone_deg,
        z_scale=m * m,
        valid=valid,
        message=" ".join(notes) if notes else "No objective fitted: the camera is at the axicon output.",
    )


def bench_ring_radius_um(*, k_perp_m_inv: float, charge: int = 0) -> float:
    """Radius of the brightest ring a charge-q Bessel beam makes for this cone.

    For q = 0 this is the first null of J0 (the usual "core radius"); for q != 0
    it is the first maximum of J_q, which is the bright ring you measure.
    """

    from scipy.special import jn_zeros, jnp_zeros

    if not math.isfinite(k_perp_m_inv) or k_perp_m_inv <= 0.0:
        raise ValueError("k_perp must be finite and positive.")
    q = abs(int(charge))
    root = float(jn_zeros(0, 1)[0]) if q == 0 else float(jnp_zeros(q, 1)[0])
    return root / float(k_perp_m_inv) * 1e6


def demagnification_for_ring(*, bench_ring_radius_um: float, target_ring_radius_um: float) -> float:
    """The M that turns this bench's ring into the requested one."""

    if not all(math.isfinite(v) and v > 0 for v in (bench_ring_radius_um, target_ring_radius_um)):
        raise ValueError("Bench and target ring radii must be finite and positive.")
    return min(1.0, float(target_ring_radius_um) / float(bench_ring_radius_um))
