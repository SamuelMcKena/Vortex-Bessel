"""Swappable axicon-generation stage for scalar and vector VBB cases.

The module keeps the legacy holographic axicon as a first-class stage while
adding a physical train:

SLM1 vortex -> free-space propagation -> SLM2 conjugate flattening -> physical
refractive axicon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

import numpy as np

from vbb_study.config import BeamDesign, EPS, TWOPI, um
from vbb_study.design import compute_design_from_config
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import make_bl_asm_propagator
from vbb_study.equations.scalar_bessel import build_conical_axicon_field_ideal

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


ConjugateMode = Literal["full", "lowpass", "zernike", "preserve_vortex"]
GenerationMethod = Literal["holographic", "physical"]


@dataclass
class AxiconResult:
    Ex: np.ndarray
    Ey: np.ndarray | None
    Ez: np.ndarray | None
    grid: object
    k_r: float
    metadata: dict[str, Any] = field(default_factory=dict)


class AxiconStage(Protocol):
    """Produces the post-axicon field on the same grid for either method."""

    def generate(self, input_field: Any, config: Any) -> AxiconResult: ...


def _payload_get(input_field: Any, key: str, default: Any = None) -> Any:
    if isinstance(input_field, Mapping):
        return input_field.get(key, default)
    return default


def _design(input_field: Any, config: Any) -> BeamDesign:
    found = _payload_get(input_field, "design")
    if found is not None:
        return found
    return compute_design_from_config(config)


def _sample_grid(input_field: Any, config: Any) -> dict[str, Any]:
    found = _payload_get(input_field, "grid")
    if found is not None:
        return found
    return make_xy_grid(int(config.grid.ideal_N), float(config.grid.ideal_dx_m))


def _as_scalar_input(input_field: Any, grid: Mapping[str, Any], design: BeamDesign) -> np.ndarray:
    if isinstance(input_field, np.ndarray):
        return np.asarray(input_field, dtype=complex)
    field = _payload_get(input_field, "field")
    if field is not None:
        return np.asarray(field, dtype=complex)
    amp = np.exp(-(np.asarray(grid["R"], dtype=float) ** 2) / max(float(design.w0_sample_m), EPS) ** 2)
    return amp.astype(complex)


def _as_vector_input(
    input_field: Any,
    grid: Mapping[str, Any],
    design: BeamDesign,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, bool]:
    ex = _payload_get(input_field, "Ex")
    ey = _payload_get(input_field, "Ey")
    ez = _payload_get(input_field, "Ez")
    if ex is not None:
        Ex = np.asarray(ex, dtype=complex)
        Ey = None if ey is None else np.asarray(ey, dtype=complex)
        Ez = None if ez is None else np.asarray(ez, dtype=complex)
        return Ex, Ey, Ez, Ey is not None or Ez is not None
    Ex = _as_scalar_input(input_field, grid, design)
    return Ex, None, None, False


def _phase_rms_from_angle(angle: np.ndarray, weights: np.ndarray | None = None) -> float:
    phase = np.asarray(angle, dtype=float)
    if weights is None:
        w = np.ones_like(phase, dtype=float)
    else:
        w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    mask = np.isfinite(phase) & np.isfinite(w) & (w > np.nanmax(w) * 1.0e-8)
    if not np.any(mask):
        return float("nan")
    phase_m = phase[mask]
    w_m = w[mask]
    mean = np.angle(np.sum(w_m * np.exp(1j * phase_m)) / (np.sum(w_m) + EPS))
    residual = np.angle(np.exp(1j * (phase_m - mean)))
    return float(np.sqrt(np.sum(w_m * residual * residual) / (np.sum(w_m) + EPS)))


def residual_phase_rms(field: np.ndarray) -> float:
    """Return weighted RMS phase after removing the circular-mean piston."""

    arr = np.asarray(field, dtype=complex)
    return _phase_rms_from_angle(np.angle(arr), np.abs(arr) ** 2)


def _lowpass(phi: np.ndarray) -> np.ndarray:
    """Smooth a wrapped phase map with a circular Gaussian low-pass."""

    phase = np.asarray(phi, dtype=float)
    if gaussian_filter is None:
        spec = np.fft.fftshift(np.fft.fft2(np.exp(1j * phase)))
        ny, nx = phase.shape
        yy = np.arange(ny) - ny / 2
        xx = np.arange(nx) - nx / 2
        XX, YY = np.meshgrid(xx, yy, indexing="xy")
        radius = 0.12 * min(nx, ny)
        spec *= np.exp(-0.5 * (XX * XX + YY * YY) / max(radius * radius, EPS))
        return np.angle(np.fft.ifft2(np.fft.ifftshift(spec)))
    z = np.exp(1j * phase)
    zr = gaussian_filter(np.real(z), sigma=4.0, mode="nearest")
    zi = gaussian_filter(np.imag(z), sigma=4.0, mode="nearest")
    return np.angle(zr + 1j * zi)


def _zernike_fit_loworder(phi_corr: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Fit piston, tilt, astigmatism, defocus, coma, and spherical terms."""

    target = np.unwrap(np.unwrap(np.asarray(phi_corr, dtype=float), axis=0), axis=1)
    amp = np.abs(np.asarray(field, dtype=complex))
    ny, nx = target.shape
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R2 = X * X + Y * Y
    mask = (R2 <= 1.0) & np.isfinite(target) & (amp > np.nanmax(amp) * 1.0e-6)
    if np.count_nonzero(mask) < 12:
        return np.zeros_like(target)
    basis = [
        np.ones_like(X),
        X,
        Y,
        2.0 * R2 - 1.0,
        X * X - Y * Y,
        2.0 * X * Y,
        (3.0 * R2 - 2.0) * X,
        (3.0 * R2 - 2.0) * Y,
        6.0 * R2 * R2 - 6.0 * R2 + 1.0,
    ]
    A = np.vstack([b[mask].ravel() for b in basis]).T
    w = amp[mask].ravel()
    yv = target[mask].ravel()
    Aw = A * w[:, None]
    yw = yv * w
    coeff, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    fit = np.zeros_like(target)
    for c, b in zip(coeff, basis):
        fit += float(c) * b
    return fit


def _wrapped_residual(phi: np.ndarray, target: np.ndarray | float = 0.0) -> np.ndarray:
    """Wrapped phase residual ``phi - target`` on [-pi, pi)."""

    return np.angle(np.exp(1j * (np.asarray(phi, dtype=float) - target)))


def _slm2_correction(
    field: np.ndarray,
    *,
    mode: ConjugateMode = "full",
    stroke_levels: int | None = None,
    reference_phase: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | None]]:
    """Return the SLM2 correction phase and diagnostics.

    ``mode='full'`` flattens the full incoming phase, including any SLM1 vortex.
    That is useful for making a plane wave before a physical axicon, but it
    deliberately strips topological charge.

    ``mode='preserve_vortex'`` instead flattens only the residual phase relative
    to a supplied helical reference phase, normally ``ell * PHI``.  The outgoing
    field then keeps the intended vortex winding while still removing the
    non-helical propagation/aberration term.
    """

    arr = np.asarray(field, dtype=complex)
    phi = np.angle(arr)
    mode_key = str(mode).lower().strip()
    target_phase = np.zeros_like(phi, dtype=float)

    if mode_key == "preserve_vortex":
        if reference_phase is None:
            raise ValueError("mode='preserve_vortex' requires reference_phase, normally ell * grid['PHI'].")
        target_phase = np.asarray(reference_phase, dtype=float)
        if target_phase.shape != phi.shape:
            raise ValueError("reference_phase must have the same shape as field.")
        phi_corr = -_wrapped_residual(phi, target_phase)
    else:
        phi_corr = -phi
        if mode_key == "zernike":
            phi_corr = _zernike_fit_loworder(phi_corr, arr)
        elif mode_key == "lowpass":
            phi_corr = _lowpass(phi_corr)
        elif mode_key != "full":
            raise ValueError(f"Unsupported SLM2 conjugation mode: {mode!r}")

    if stroke_levels is not None:
        levels = int(stroke_levels)
        if levels < 2:
            raise ValueError("stroke_levels must be at least 2 when provided.")
        phi_corr = np.round(phi_corr / TWOPI * levels) * (TWOPI / levels)
    after_phase = phi + phi_corr
    weights = np.abs(arr) ** 2
    if mode_key == "preserve_vortex":
        before_residual = _wrapped_residual(phi, target_phase)
        after_residual = _wrapped_residual(after_phase, target_phase)
    else:
        # Preserve the exact legacy diagnostics for all pre-existing modes.
        before_residual = phi
        after_residual = after_phase
    diagnostics: dict[str, float | int | str | None] = {
        "mode": mode_key,
        "stroke_levels": None if stroke_levels is None else int(stroke_levels),
        "target_phase": "helical_reference" if mode_key == "preserve_vortex" else "flat",
        "residual_phase_rms_before_rad": _phase_rms_from_angle(before_residual, weights),
        "residual_phase_rms_after_rad": _phase_rms_from_angle(after_residual, weights),
    }
    return phi_corr, diagnostics


def slm2_conjugate(
    field: np.ndarray,
    *,
    mode: ConjugateMode = "full",
    stroke_levels: int | None = None,
    reference_phase: np.ndarray | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str | None]]:
    """Apply the SLM2 conjugate phase.

    The default ``mode='full'`` flattens everything and therefore removes an
    upstream vortex.  Use ``mode='preserve_vortex'`` with ``reference_phase`` to
    leave the intended helical phase in the field.
    """

    arr = np.asarray(field, dtype=complex)
    phi_corr, diagnostics = _slm2_correction(arr, mode=mode, stroke_levels=stroke_levels, reference_phase=reference_phase)
    corrected = arr * np.exp(1j * phi_corr)
    if return_diagnostics:
        return corrected, diagnostics
    return corrected


def validate_slm2_vortex_contract(
    mode: str,
    charge: int,
    *,
    allow_vortex_removal: bool = False,
) -> str:
    """Validate and return the normalised SLM2 mode for a requested charge."""

    mode_key = str(mode).lower().strip()
    if mode_key == "full" and int(charge) != 0 and not bool(allow_vortex_removal):
        raise ValueError(
            "slm2_conjugate_mode='full' removes the intended topological charge "
            f"ell={int(charge)}. Use 'preserve_vortex', or set allow_vortex_removal=True "
            "only for an intentional zero-winding diagnostic."
        )
    return mode_key


class HolographicAxicon:
    """Existing method: axicon phase encoded on the SLM."""

    def generate(self, input_field: Any, config: Any) -> AxiconResult:
        design = _design(input_field, config)
        grid = _sample_grid(input_field, config)
        U = build_conical_axicon_field_ideal(grid, design, config.laser)
        return AxiconResult(
            Ex=U,
            Ey=None,
            Ez=None,
            grid=grid,
            k_r=float(design.kr_sample_m_inv),
            metadata={
                "method": "holographic",
                "stage": "slm_encoded_axicon",
                "k_r_m_inv": float(design.kr_sample_m_inv),
                "field_definition": "conical_axicon_input",
            },
        )


class PhysicalAxicon:
    """SLM1 vortex -> vortex-safe SLM2 correction -> refractive axicon.

    Full conjugation is available only as an explicitly acknowledged
    diagnostic when the requested charge is non-zero because it removes the
    helical phase as well as the unwanted propagation phase.
    """

    def generate(self, input_field: Any, config: Any) -> AxiconResult:
        design = _design(input_field, config)
        grid = _sample_grid(input_field, config)
        physical = getattr(config, "physical_axicon", None)
        inter_slm_z_m = float(getattr(physical, "inter_slm_z_m", 25.0 * um))
        inter_slm_n = float(getattr(physical, "inter_slm_n", 1.0))
        mode = getattr(physical, "slm2_conjugate_mode", "preserve_vortex")
        allow_vortex_removal = bool(getattr(physical, "allow_vortex_removal", False))
        stroke_levels = getattr(physical, "slm2_stroke_levels", 256)
        n_axicon_value = getattr(physical, "n_axicon", None)
        n_axicon = float(design.n_axicon if n_axicon_value is None else n_axicon_value)
        axicon_medium_n = float(getattr(physical, "axicon_medium_n", design.hologram_medium_n))
        aperture_radius_m = getattr(physical, "axicon_aperture_radius_m", None)
        transmission = float(getattr(physical, "axicon_transmission", 1.0))
        slm1_transmission = float(getattr(physical, "slm1_transmission", 1.0))
        slm2_transmission = float(getattr(physical, "slm2_transmission", 1.0))
        charge_value = getattr(physical, "slm1_vortex_charge", None)
        charge = int(design.ell if charge_value is None else charge_value)
        mode_key = validate_slm2_vortex_contract(
            mode,
            charge,
            allow_vortex_removal=allow_vortex_removal,
        )

        Ex0, Ey0, Ez0, vector_input = _as_vector_input(input_field, grid, design)
        vortex = np.exp(1j * charge * np.asarray(grid["PHI"], dtype=float))
        amp_prefix = np.sqrt(max(slm1_transmission, 0.0))
        Ex1 = amp_prefix * Ex0 * vortex
        Ey1 = None if Ey0 is None else amp_prefix * Ey0 * vortex
        Ez1 = None if Ez0 is None else amp_prefix * Ez0 * vortex

        def _propagate(component: np.ndarray | None) -> np.ndarray | None:
            if component is None:
                return None
            if abs(inter_slm_z_m) <= EPS:
                return np.asarray(component, dtype=complex).copy()
            prop = make_bl_asm_propagator(
                np.asarray(component, dtype=complex),
                grid,
                config.laser.wavelength_m,
                n_medium=inter_slm_n,
                bandlimit=True,
            )
            return prop(inter_slm_z_m)

        Ex2 = _propagate(Ex1)
        Ey2 = _propagate(Ey1)
        Ez2 = _propagate(Ez1)
        assert Ex2 is not None
        reference = Ex2
        if Ey2 is not None and np.sum(np.abs(Ey2) ** 2) > np.sum(np.abs(reference) ** 2):
            reference = Ey2
        reference_phase = charge * np.asarray(grid["PHI"], dtype=float) if mode_key == "preserve_vortex" else None
        phi_corr, slm2_diag = _slm2_correction(
            reference,
            mode=mode_key,
            stroke_levels=stroke_levels,
            reference_phase=reference_phase,
        )
        slm2_factor = np.sqrt(max(slm2_transmission, 0.0)) * np.exp(1j * phi_corr)
        Ex_flat = Ex2 * slm2_factor
        Ey_flat = None if Ey2 is None else Ey2 * slm2_factor
        Ez_flat = None if Ez2 is None else Ez2 * slm2_factor

        k_r = float(design.kr_sample_m_inv)
        gamma_override = getattr(physical, "axicon_base_angle_deg", None)
        if gamma_override is None:
            denom = config.laser.k0 * (n_axicon - axicon_medium_n)
            gamma_rad = np.arctan(k_r / max(abs(denom), EPS))
            if denom < 0.0:
                gamma_rad = -gamma_rad
            gamma_deg = float(np.rad2deg(gamma_rad))
        else:
            gamma_deg = float(gamma_override)
            k_r = float(config.laser.k0 * (n_axicon - axicon_medium_n) * np.tan(np.deg2rad(gamma_deg)))

        from . import vbb_polarized_train

        tp, ts = vbb_polarized_train.fresnel_sp_coefficients(n_axicon, axicon_medium_n, gamma_deg)
        scalar_fresnel_amp = np.sqrt(max(0.5 * (abs(tp) ** 2 + abs(ts) ** 2), 0.0))
        axicon_phase = np.exp(-1j * abs(k_r) * np.asarray(grid["R"], dtype=float))
        if aperture_radius_m is None:
            aperture = np.ones_like(np.asarray(grid["R"], dtype=float))
        else:
            aperture = (np.asarray(grid["R"], dtype=float) <= float(aperture_radius_m)).astype(float)
        ax_amp = np.sqrt(max(transmission, 0.0)) * aperture

        if vector_input:
            element = vbb_polarized_train.PhysicalAxiconElement(
                n_axicon=n_axicon,
                base_angle_deg=gamma_deg,
                aperture_radius_m=aperture_radius_m,
                conical_phase_sign=-1.0,
                transmission=transmission,
            )
            state = vbb_polarized_train.OpticalTrainState(
                Ex=Ex_flat,
                Ey=np.zeros_like(Ex_flat) if Ey_flat is None else Ey_flat,
                Ez=np.zeros_like(Ex_flat) if Ez_flat is None else Ez_flat,
                grid=grid,
                metadata={"history": [], "phase_maps": {}, "propagation": []},
            )
            out = vbb_polarized_train.apply_physical_axicon_element(
                state,
                element,
                wavelength_m=config.laser.wavelength_m,
                n_medium=axicon_medium_n,
            )
            Ex_out = out.Ex
            Ey_out = out.Ey
            Ez_out = out.Ez
        else:
            Ex_out = ax_amp * scalar_fresnel_amp * axicon_phase * Ex_flat
            Ey_out = None
            Ez_out = None

        input_power = float(np.sum(np.abs(Ex0) ** 2) * float(grid["dx"]) ** 2)
        output_power = float(
            np.sum(np.abs(Ex_out) ** 2 + (0.0 if Ey_out is None else np.abs(Ey_out) ** 2) + (0.0 if Ez_out is None else np.abs(Ez_out) ** 2))
            * float(grid["dx"]) ** 2
        )
        metadata = {
            "method": "physical",
            "stage": "slm1_slm2_conjugate_physical_axicon",
            "k_r_m_inv": float(abs(k_r)),
            "slm1_phase_components": ("vortex",),
            "slm2_phase_components": ("conjugate_flattening", "optional_amplitude_correction"),
            "slm1_contains_axicon": False,
            "slm2_contains_axicon": False,
            "physical_axicon_phase": "continuous_conical",
            "physical_axicon_pixelated": False,
            "slm1_vortex_charge": int(charge),
            "inter_slm_z_m": inter_slm_z_m,
            "inter_slm_n": inter_slm_n,
            "slm2_conjugate_mode": str(slm2_diag["mode"]),
            "allow_vortex_removal": bool(allow_vortex_removal),
            "vortex_removed_intentionally": bool(mode_key == "full" and charge != 0),
            "vortex_preservation_contract": (
                "preserved" if mode_key == "preserve_vortex"
                else "not_applicable_ell0" if charge == 0
                else "explicitly_acknowledged_removal"
            ),
            "slm2_stroke_levels": slm2_diag["stroke_levels"],
            "slm2_residual_phase_rms_before_rad": slm2_diag["residual_phase_rms_before_rad"],
            "slm2_residual_phase_rms_after_rad": slm2_diag["residual_phase_rms_after_rad"],
            "n_axicon": n_axicon,
            "axicon_medium_n": axicon_medium_n,
            "axicon_base_angle_deg": gamma_deg,
            "tp_abs": float(abs(tp)),
            "ts_abs": float(abs(ts)),
            "scalar_fresnel_amplitude": float(scalar_fresnel_amp),
            "slm1_transmission": slm1_transmission,
            "slm2_transmission": slm2_transmission,
            "axicon_transmission": transmission,
            "input_power_au_m2": input_power,
            "output_power_au_m2": output_power,
            "efficiency": float(output_power / max(input_power, EPS)),
        }
        return AxiconResult(
            Ex=np.asarray(Ex_out, dtype=complex),
            Ey=None if Ey_out is None else np.asarray(Ey_out, dtype=complex),
            Ez=None if Ez_out is None else np.asarray(Ez_out, dtype=complex),
            grid=grid,
            k_r=float(abs(k_r)),
            metadata=metadata,
        )


def axicon_stage(method: GenerationMethod) -> AxiconStage:
    key = str(method).lower().strip()
    if key == "holographic":
        return HolographicAxicon()
    if key == "physical":
        return PhysicalAxicon()
    raise ValueError(f"Unsupported generation method: {method!r}")


__all__ = [
    "AxiconResult",
    "AxiconStage",
    "ConjugateMode",
    "GenerationMethod",
    "HolographicAxicon",
    "PhysicalAxicon",
    "axicon_stage",
    "residual_phase_rms",
    "slm2_conjugate",
    "validate_slm2_vortex_contract",
]
