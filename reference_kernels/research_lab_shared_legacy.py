from __future__ import annotations

"""Shared helpers for the modular PhD research notebooks.

This module keeps the current scalar propagation path intact by loading the
existing shared kernel from the legacy folder, then layers three additions on
it:

1. A small optics-train API that is easy to edit from notebooks.
2. A reusable vector helper surface lifted out of the current vector notebook.
3. An explicitly idealized material-response model for exploratory pulse
   accumulation studies.

The material-response section is intentionally not a validated nonlinear solver.
It is a fast, research-planning model that is useful for hypothesis generation
and for structuring later comparisons against experiment.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "notebook"

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover - fallback for minimal environments
    gaussian_filter = None


_THIS_DIR = Path(__file__).resolve().parent
_SEARCH_ROOTS = [_THIS_DIR, *_THIS_DIR.parents]


def _find_existing_file(*relative_parts: str) -> Path:
    for root in _SEARCH_ROOTS:
        candidate = root.joinpath(*relative_parts)
        if candidate.exists():
            return candidate
    joined = "/".join(relative_parts)
    raise FileNotFoundError(f"Could not locate {joined} from {__file__}")


_SCALAR_KERNEL = _find_existing_file("Full Code", "balyian_shared_kernel_v4.py")
exec(compile(_SCALAR_KERNEL.read_text(encoding="utf-8"), str(_SCALAR_KERNEL), "exec"), globals(), globals())


VECTOR_QUALITY_MAP: Dict[str, Dict[str, int]] = {
    "fast": {"N": 768, "ns": 50},
    "balanced": {"N": 1024, "ns": 70},
    "paper": {"N": 2048, "ns": 120},
}


def make_scalar_lab_context(
    quality: str = "balanced",
    metric_quality: str = "balanced",
    showcase_quality: str = "paper",
    wavelength: float = 1030 * nm,
    w0: float = 2.0 * mm,
    ell: int = 3,
    gamma_deg: float = 1.0,
    attach_device_realism: bool = True,
    device_grid: bool = False,
    device_dx_um: float = 4.0,
    device_N: int = 1536,
) -> Dict[str, object]:
    """Build one notebook-friendly scalar context.

    `device_grid=False` keeps the flexible research grid while optionally
    attaching actual-device defaults. `device_grid=True` switches to the fixed
    integer-pitch grid that mirrors the real SLM baseline more closely.
    """
    if device_grid:
        ctx = make_integer_pitch_device_context(
            dx_um=device_dx_um,
            N=device_N,
            wavelength=wavelength,
            w0=w0,
            ell=ell,
            gamma_deg=gamma_deg,
            quality_label=quality,
        )
    else:
        ctx = make_research_context(
            quality=quality,
            metric_quality=metric_quality,
            showcase_quality=showcase_quality,
            wavelength=wavelength,
            w0=w0,
            ell=ell,
            gamma_deg=gamma_deg,
        )
        if attach_device_realism:
            attach_actual_slm_defaults(ctx)
    return ctx


def make_vector_lab_context(
    quality: str = "balanced",
    wavelength: float = 532 * nm,
    n_medium: float = 1.0,
    w0: float = 2.0 * mm,
    ell: int = 3,
    n_axicon: float = 1.5,
    gamma_deg: float = 1.0,
    use_eq16_qwp2: bool = True,
) -> Dict[str, object]:
    """Build the base vector notebook context.

    The defaults mirror the current vector notebook, but this helper keeps the
    tunable knobs in one place and makes later notebook edits smaller.
    """
    if quality not in VECTOR_QUALITY_MAP:
        raise ValueError("quality must be 'fast', 'balanced', or 'paper'")

    N_use = int(VECTOR_QUALITY_MAP[quality]["N"])
    ns_use = int(VECTOR_QUALITY_MAP[quality]["ns"])

    gs = GridSpec(N=N_use, L=5.0 * mm)
    beam = BeamSpec(wavelength=wavelength, n_medium=n_medium, w0=w0, ell=int(ell))
    axicon = AxiconSpec(n_axicon=n_axicon, gamma_deg=gamma_deg, kr_mode="tan")
    mask = MaskSpec(signum_pi_flip=False)
    realism = SLMRealism(
        quantize_bits=None,
        pixel_pitch=None,
        carrier_lpmm=0.0,
        tilt_mrad=(0.0, 0.0),
        active_size=None,
        fill_factor=None,
    )
    sim = SimSpec(include_evanescent=True, alpha_np_per_m=0.0)

    k0 = 2 * np.pi / beam.wavelength
    gamma = np.deg2rad(axicon.gamma_deg)
    kr_est = compute_kr(k0, axicon.n_axicon, beam.n_medium, gamma, mode=axicon.kr_mode)
    zmax_est = zmax_baliyan(beam.w0, k0, kr_est)
    scan = PathScanSpec(s_i=0.0, s_f=max(320 * mm, 1.25 * zmax_est), ns=ns_use, x_crop_mm=1.0)

    return {
        "quality": quality,
        "gs": gs,
        "beam": beam,
        "axicon": axicon,
        "mask": mask,
        "realism": realism,
        "sim": sim,
        "scan": scan,
        "kr_est": kr_est,
        "zmax_est": zmax_est,
        "use_eq16_qwp2": bool(use_eq16_qwp2),
    }


class NDFilter(OpticalElement):
    """Neutral-density filter modeled as a uniform amplitude attenuation."""

    def __init__(
        self,
        optical_density: float = 0.0,
        transmission: Optional[float] = None,
        name: str = "NDFilter",
    ):
        super().__init__(name=name)
        self.optical_density = float(optical_density)
        self.transmission = float(10.0 ** (-self.optical_density) if transmission is None else transmission)

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        amp = math.sqrt(max(self.transmission, 0.0))
        out = ScalarField(U=field.U * amp, wavelength=field.wavelength, n_medium=field.n_medium, grid=field.grid)
        return out, {
            "transmission": np.array([self.transmission], dtype=float),
            "optical_density": np.array([self.optical_density], dtype=float),
        }


class PhasePlate(OpticalElement):
    """Apply a user-supplied phase map or phase-map callback."""

    def __init__(
        self,
        phase: Union[np.ndarray, Callable[[Grid], np.ndarray]],
        name: str = "PhasePlate",
    ):
        super().__init__(name=name)
        self.phase = phase

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        if callable(self.phase):
            phi = np.asarray(self.phase(field.grid), dtype=float)
        else:
            phi = np.asarray(self.phase, dtype=float)
        if phi.shape != field.U.shape:
            raise ValueError("PhasePlate phase array must match the field shape.")
        out = ScalarField(
            U=field.U * np.exp(1j * phi),
            wavelength=field.wavelength,
            n_medium=field.n_medium,
            grid=field.grid,
        )
        return out, {"phase": phi}


class ThinSample(OpticalElement):
    """Simple thin-slab sample model with phase delay and linear attenuation."""

    def __init__(
        self,
        thickness: float,
        n_sample: float,
        alpha_np_per_m: float = 0.0,
        phase_offset: float = 0.0,
        name: str = "ThinSample",
    ):
        super().__init__(name=name)
        self.thickness = float(thickness)
        self.n_sample = float(n_sample)
        self.alpha_np_per_m = float(alpha_np_per_m)
        self.phase_offset = float(phase_offset)

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        k = 2 * np.pi / field.wavelength
        phase = k * (self.n_sample - field.n_medium) * self.thickness + self.phase_offset
        amp = math.exp(-0.5 * self.alpha_np_per_m * self.thickness)
        out = ScalarField(
            U=field.U * amp * np.exp(1j * phase),
            wavelength=field.wavelength,
            n_medium=field.n_medium,
            grid=field.grid,
        )
        return out, {
            "thickness": np.array([self.thickness], dtype=float),
            "n_sample": np.array([self.n_sample], dtype=float),
            "alpha_np_per_m": np.array([self.alpha_np_per_m], dtype=float),
            "phase_shift_rad": np.array([phase], dtype=float),
        }


def list_supported_optical_elements() -> List[str]:
    return [
        "slm",
        "free_space",
        "lens",
        "aperture",
        "nd_filter",
        "thin_sample",
        "phase_plate",
        "camera",
    ]


BenchElementSpec = Union[OpticalElement, Dict[str, object]]


def _sequence_kind(spec: BenchElementSpec) -> Optional[str]:
    if isinstance(spec, dict):
        return str(spec.get("type", "")).strip().lower()
    if isinstance(spec, SLMPhase):
        return "slm"
    return None


def make_optical_element(
    spec: BenchElementSpec,
    *,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
) -> OpticalElement:
    if isinstance(spec, OpticalElement):
        return spec

    if not isinstance(spec, dict):
        raise TypeError("Each optical-train item must be an OpticalElement or a dict specification.")

    kind = str(spec.get("type", "")).strip().lower()
    name = str(spec.get("name", kind or "Element"))

    if kind == "slm":
        return SLMPhase(
            beam=spec.get("beam", beam),
            axicon=spec.get("axicon", axicon),
            mask=spec.get("mask", mask),
            realism=spec.get("realism", realism),
            name=name,
        )
    if kind == "free_space":
        return FreeSpace(z=float(spec["z"]), sim=spec.get("sim", sim), name=name)
    if kind == "lens":
        return Lens(f=float(spec["f"]), name=name)
    if kind == "aperture":
        rect = spec.get("rect")
        rect_tuple = tuple(rect) if rect is not None else None
        return Aperture(radius=spec.get("radius"), rect=rect_tuple, name=name)
    if kind == "nd_filter":
        return NDFilter(
            optical_density=float(spec.get("optical_density", spec.get("od", 0.0))),
            transmission=spec.get("transmission"),
            name=name,
        )
    if kind == "thin_sample":
        return ThinSample(
            thickness=float(spec["thickness"]),
            n_sample=float(spec.get("n_sample", beam.n_medium)),
            alpha_np_per_m=float(spec.get("alpha_np_per_m", 0.0)),
            phase_offset=float(spec.get("phase_offset", 0.0)),
            name=name,
        )
    if kind == "phase_plate":
        if "phase" not in spec:
            raise KeyError("phase_plate specifications require a 'phase' entry.")
        return PhasePlate(phase=spec["phase"], name=name)
    if kind == "camera":
        return CameraElement(name=name)

    raise ValueError(f"Unsupported optical element type: {kind}")


def run_optical_train(
    ctx: Dict[str, object],
    sequence: Sequence[BenchElementSpec],
    *,
    beam: Optional[BeamSpec] = None,
    axicon: Optional[AxiconSpec] = None,
    mask: Optional[MaskSpec] = None,
    realism: Optional[SLMRealism] = None,
    sim: Optional[SimSpec] = None,
    gs: Optional[GridSpec] = None,
    prepend_slm: bool = True,
) -> Dict[str, object]:
    """Run a simple editable optics train.

    The notebook-facing convention is that `sequence` is a list of dicts like:

        {"type": "free_space", "z": 50 * mm}
        {"type": "lens", "f": 100 * mm}
        {"type": "nd_filter", "optical_density": 0.6}
        {"type": "camera", "name": "After objective"}
    """
    beam = ctx.get("beam_base", ctx.get("beam")) if beam is None else beam
    axicon = ctx.get("axicon_base", ctx.get("axicon")) if axicon is None else axicon
    mask = ctx.get("mask") if mask is None else mask
    realism = ctx.get("realism") if realism is None else realism
    sim = ctx.get("sim") if sim is None else sim
    gs = ctx.get("gs", ctx.get("gs_metric", ctx.get("gs_showcase"))) if gs is None else gs

    if beam is None or axicon is None or mask is None or realism is None or sim is None or gs is None:
        raise ValueError("Context is missing one of beam/axicon/mask/realism/sim/gs.")

    input_field = build_input_field(gs, beam)
    bench = OpticalBench()

    has_slm = any(_sequence_kind(spec) == "slm" for spec in sequence)
    if prepend_slm and not has_slm:
        bench.add(SLMPhase(beam=beam, axicon=axicon, mask=mask, realism=realism, name="SLM"))

    for spec in sequence:
        bench.add(make_optical_element(spec, beam=beam, axicon=axicon, mask=mask, realism=realism, sim=sim))

    final_field, captures, meta = bench.run(input_field)
    capture_intensity = {name: np.abs(field.U) ** 2 for name, field in captures.items()}

    return {
        "input_field": input_field,
        "final_field": final_field,
        "captures": captures,
        "capture_intensity": capture_intensity,
        "meta": meta,
        "grid": final_field.grid,
    }


def capture_order(result: Dict[str, object]) -> List[str]:
    return list(result.get("captures", {}).keys())


@dataclass
class VectorField:
    Ex: np.ndarray
    Ey: np.ndarray


def _stage5_vector_module():
    from vbb_study import vbb_vector

    return vbb_vector


def _to_stage5_vector_field(vf: VectorField):
    vbb_vector = _stage5_vector_module()
    return vbb_vector.VectorField(Ex=np.asarray(vf.Ex, dtype=complex), Ey=np.asarray(vf.Ey, dtype=complex))


def _from_stage5_vector_field(vf) -> VectorField:
    return VectorField(Ex=np.asarray(vf.Ex, dtype=complex), Ey=np.asarray(vf.Ey, dtype=complex))


def _grid_to_stage5_mapping(grid: Grid) -> Dict[str, object]:
    return {
        "N": int(grid.N),
        "dx": float(grid.dx),
        "x": np.asarray(grid.x, dtype=float),
        "X": np.asarray(grid.X, dtype=float),
        "Y": np.asarray(grid.Y, dtype=float),
        "R": np.asarray(grid.R, dtype=float),
        "Phi": np.asarray(grid.PHI, dtype=float),
        "FX": np.asarray(grid.FX, dtype=float),
        "FY": np.asarray(grid.FY, dtype=float),
    }


J_QWP1_AZ = np.array([[-1j, 0], [0, 1]], dtype=complex)
J_QWP1_RAD = np.array([[1, 0], [0, -1j]], dtype=complex)
J_QWP2_EQ16 = np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex)
J_QWP2_PHYS45 = 0.5 * np.array([[1 + 1j, -1 + 1j], [-1 + 1j, 1 + 1j]], dtype=complex)


def apply_signed_phase_pattern(
    field_in: ScalarField,
    phi: np.ndarray,
    sign: float,
    active_size: Optional[Tuple[float, float]] = None,
) -> ScalarField:
    U = field_in.U * np.exp(1j * float(sign) * phi)
    if active_size is not None:
        sx, sy = active_size
        aperture = ((np.abs(field_in.grid.X) <= 0.5 * sx) & (np.abs(field_in.grid.Y) <= 0.5 * sy)).astype(float)
        U = U * aperture
    return ScalarField(U=U, wavelength=field_in.wavelength, n_medium=field_in.n_medium, grid=field_in.grid)


def build_dualpass_slm_fields(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
) -> Dict[str, object]:
    field_in = build_input_field(gs, beam)
    slm = SLMPhase(beam=beam, axicon=axicon, mask=mask, realism=realism, name="SLM")

    field_plus, meta = slm.apply(field_in)
    phi = meta["phase"]
    field_plus_manual = apply_signed_phase_pattern(field_in, phi, sign=+1.0, active_size=realism.active_size)
    field_minus = apply_signed_phase_pattern(field_in, phi, sign=-1.0, active_size=realism.active_size)

    plus_err = np.linalg.norm(field_plus.U - field_plus_manual.U) / (np.linalg.norm(field_plus.U) + 1e-15)
    return {
        "field_in": field_in,
        "phi": phi,
        "field_plus": field_plus,
        "field_minus": field_minus,
        "plus_err": float(plus_err),
    }


def build_component_propagators(
    fields: Dict[str, object],
    sim: SimSpec,
) -> Tuple[Callable[[float], ScalarField], Callable[[float], ScalarField]]:
    prop_x = build_propagator(fields["field_plus"], sim)
    prop_y = build_propagator(fields["field_minus"], sim)
    return prop_x, prop_y


def apply_jones_to_vector(vf: VectorField, J: np.ndarray) -> VectorField:
    Ex2 = J[0, 0] * vf.Ex + J[0, 1] * vf.Ey
    Ey2 = J[1, 0] * vf.Ex + J[1, 1] * vf.Ey
    return VectorField(Ex=Ex2, Ey=Ey2)


def apply_paper_optics(vf: VectorField, mode: str, use_eq16_qwp2: bool = True) -> VectorField:
    """Apply the paper-replica waveplate chain, not the actual Case-1 lab path."""

    if not bool(use_eq16_qwp2):
        raise ValueError("The Stage 5 implementation uses the paper eq. 16 QWP2 definition only.")
    vbb_vector = _stage5_vector_module()
    return _from_stage5_vector_field(vbb_vector.apply_paper_optics(_to_stage5_vector_field(vf), mode))


def total_intensity(vf: VectorField) -> np.ndarray:
    vbb_vector = _stage5_vector_module()
    return vbb_vector.total_intensity(_to_stage5_vector_field(vf))


def stokes_parameters(vf: VectorField) -> Dict[str, np.ndarray]:
    vbb_vector = _stage5_vector_module()
    return vbb_vector.stokes_parameters(_to_stage5_vector_field(vf))


def linear_analyzer_intensity(vf: VectorField, theta_deg: float) -> np.ndarray:
    vbb_vector = _stage5_vector_module()
    return vbb_vector.linear_analyzer_intensity(_to_stage5_vector_field(vf), theta_deg)


def analyzer_maps(vf: VectorField, angles_deg: Tuple[int, ...] = (0, 45, 90, 135)) -> Dict[int, np.ndarray]:
    vbb_vector = _stage5_vector_module()
    return vbb_vector.analyzer_maps(_to_stage5_vector_field(vf), angles_deg)


def predicted_ring_radius(ell: int, kr: float) -> float:
    l_abs = int(abs(ell))
    if l_abs == 0:
        return 2.405 / kr
    return sp.jnp_zeros(l_abs, 1)[0] / kr


def ring_roi(grid: Grid, r_center: float, rel_width: float = 0.18) -> np.ndarray:
    r_lo = max(0.0, (1.0 - rel_width) * r_center)
    r_hi = (1.0 + rel_width) * r_center
    roi = (grid.R >= r_lo) & (grid.R <= r_hi)
    if np.count_nonzero(roi) < 32:
        roi = (grid.R >= max(0.0, 0.5 * r_center)) & (grid.R <= max(3.0 * grid.dx, 1.5 * r_center))
    return roi


def image_correlation(A: np.ndarray, B: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        a = np.asarray(A, float).ravel()
        b = np.asarray(B, float).ravel()
    else:
        a = np.asarray(A[mask], float).ravel()
        b = np.asarray(B[mask], float).ravel()

    if a.size < 8:
        return 1.0

    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt(np.sum(a * a))
    db = np.sqrt(np.sum(b * b))
    if da <= 1e-20 or db <= 1e-20:
        return 1.0
    return float(np.sum(a * b) / (da * db))


def petal_modulation_metric(I: np.ndarray, roi: np.ndarray) -> float:
    vbb_vector = _stage5_vector_module()
    return vbb_vector.petal_modulation_metric(I, roi)


def phase_winding_estimate(U: np.ndarray, grid: Grid, radius: float, n_samples: int = 720) -> float:
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=True)
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    ix = np.clip(np.round((xs - grid.x[0]) / grid.dx).astype(int), 0, grid.N - 1)
    iy = np.clip(np.round((ys - grid.x[0]) / grid.dx).astype(int), 0, grid.N - 1)
    phase = np.unwrap(np.angle(U[iy, ix]))
    return float((phase[-1] - phase[0]) / (2.0 * np.pi))


def conj_match_report(vf: VectorField, mask: np.ndarray) -> Tuple[complex, float]:
    ref = np.conj(vf.Ex[mask])
    tgt = vf.Ey[mask]
    c = np.vdot(ref, tgt) / (np.vdot(ref, ref) + 1e-20)
    err = np.linalg.norm(tgt - c * ref) / (np.linalg.norm(tgt) + 1e-20)
    return c, float(err)


def print_vector_diagnostics(
    mode: str,
    ell: int,
    vf: VectorField,
    analyzer: Dict[int, np.ndarray],
    roi: np.ndarray,
    kr: float,
    grid: Grid,
) -> None:
    Px = float(np.sum(np.abs(vf.Ex) ** 2))
    Py = float(np.sum(np.abs(vf.Ey) ** 2))
    ratio = Py / (Px + 1e-20)
    print(f"[{mode} l={ell}] Py/Px = {ratio:.6f}")
    if ratio < 0.1 or ratio > 10.0:
        print(f"WARNING: [{mode} l={ell}] strong component imbalance (Py/Px not O(1)).")

    mods = {ang: petal_modulation_metric(analyzer[ang], roi) for ang in (0, 45, 90, 135)}
    print(
        f"[{mode} l={ell}] petal modulation std/mean: "
        f"0={mods[0]:.4f}, 45={mods[45]:.4f}, 90={mods[90]:.4f}, 135={mods[135]:.4f}"
    )

    c01 = image_correlation(analyzer[0], analyzer[45], roi)
    c13 = image_correlation(analyzer[45], analyzer[135], roi)
    c02 = image_correlation(analyzer[0], analyzer[90], roi)
    c23 = image_correlation(analyzer[90], analyzer[135], roi)
    print(
        f"[{mode} l={ell}] analyzer corr: "
        f"corr(I0,I45)={c01:.4f}, corr(I45,I135)={c13:.4f}, "
        f"corr(I0,I90)={c02:.4f}, corr(I90,I135)={c23:.4f}"
    )

    r_ring = predicted_ring_radius(ell, kr)
    w_ex = phase_winding_estimate(vf.Ex, grid, r_ring)
    w_ey = phase_winding_estimate(vf.Ey, grid, r_ring)
    coeff, err = conj_match_report(vf, roi)
    print(
        f"[{mode} l={ell}] winding Ex~{w_ex:.3f}, Ey~{w_ey:.3f}; "
        f"Ey ~= c*conj(Ex): |c|={abs(coeff):.3e}, arg(c)={np.angle(coeff):.3f} rad, rel_err={err:.3e}"
    )


def select_z_cam_by_overlap(
    prop_x: Callable[[float], ScalarField],
    prop_y: Callable[[float], ScalarField],
    grid: Grid,
    ell: int,
    kr: float,
    z_values: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    roi = ring_roi(grid, predicted_ring_radius(ell, kr), rel_width=0.18)
    scores = np.zeros(len(z_values), dtype=float)
    for i, z in enumerate(z_values):
        Ix = np.abs(prop_x(float(z)).U) ** 2
        Iy = np.abs(prop_y(float(z)).U) ** 2
        scores[i] = image_correlation(Ix, Iy, roi)
    j = int(np.argmax(scores))
    return float(z_values[j]), float(scores[j]), z_values, scores, roi


def build_vector_at_camera(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
    z_cam: float,
    mode: str,
    use_eq16_qwp2: bool = True,
    method: str = "A",
    future_element: str = "none",
    paper_replica: bool = False,
) -> Dict[str, object]:
    """Build the vector camera prediction for the current lab unless requested otherwise.

    By default this is the confirmed Case-1 hardware: H is shaped, V is a fixed
    reference, and no waveplate Jones matrix is applied. Set
    ``paper_replica=True`` only when reproducing the Baliyan-Nishchal waveplate
    chain as a simulation cross-check.
    """

    fields = build_dualpass_slm_fields(gs, beam, axicon, mask, realism)
    if paper_replica:
        prop_x, prop_y = build_component_propagators(fields, sim)
        field_x_cam = prop_x(z_cam)
        field_y_cam = prop_y(z_cam)
        vf_pre = VectorField(Ex=field_x_cam.U, Ey=field_y_cam.U)
        vf_post = apply_paper_optics(vf_pre, mode=mode, use_eq16_qwp2=use_eq16_qwp2)
        analyzer = analyzer_maps(vf_post)
        return {
            "fields": fields,
            "prop_x": prop_x,
            "prop_y": prop_y,
            "vf_pre": vf_pre,
            "vf_post": vf_post,
            "analyzer": analyzer,
            "total": total_intensity(vf_post),
            "stokes": stokes_parameters(vf_post),
            "vector_model": "paper-replica (with-waveplates)",
            "lab_realizable": False,
            "simulation_only": True,
            "uses_waveplates": True,
        }

    vbb_vector = _stage5_vector_module()
    k0 = 2.0 * np.pi / float(beam.wavelength)
    gamma = np.deg2rad(float(axicon.gamma_deg))
    kr = compute_kr(k0, axicon.n_axicon, beam.n_medium, gamma, mode=axicon.kr_mode)
    lab_case = vbb_vector.build_actual_lab_vector_case(
        _grid_to_stage5_mapping(fields["field_in"].grid),
        ell=int(beam.ell),
        kr_m_inv=float(kr),
        waist_m=float(beam.w0),
        target=mode,
        method=method,
        future_element=future_element,
        carrier_lpmm=float(getattr(realism, "carrier_lpmm", 0.0)),
        z_m=float(z_cam),
        wavelength_m=float(beam.wavelength),
        n_medium=float(beam.n_medium),
    )
    vf_post = None if lab_case.get("field") is None else _from_stage5_vector_field(lab_case["field"])

    return {
        "fields": fields,
        "prop_x": None,
        "prop_y": None,
        "vf_pre": None,
        "vf_post": vf_post,
        "analyzer": lab_case.get("analyzer", {}),
        "total": lab_case.get("total_intensity"),
        "stokes": lab_case.get("stokes", {}),
        "lab_observable": lab_case.get("lab_observable", {}),
        "lab_case": lab_case,
        "vector_model": "actual Case-1 lab" if lab_case.get("lab_realizable") else "simulation-only/unreachable",
        "lab_realizable": bool(lab_case.get("lab_realizable", False)),
        "simulation_only": bool(lab_case.get("simulation_only", False)),
        "uses_waveplates": False,
    }


def sweep_vector_orders(
    ctx: Dict[str, object],
    ells: Sequence[int],
    z_cam: float,
    mode: str,
    *,
    use_eq16_qwp2: Optional[bool] = None,
    method: str = "A",
    future_element: str = "none",
    paper_replica: bool = False,
) -> List[Dict[str, object]]:
    use_eq16_qwp2 = ctx.get("use_eq16_qwp2", True) if use_eq16_qwp2 is None else use_eq16_qwp2
    rows: List[Dict[str, object]] = []
    for ell in ells:
        beam = BeamSpec(
            wavelength=ctx["beam"].wavelength,
            n_medium=ctx["beam"].n_medium,
            w0=ctx["beam"].w0,
            ell=int(ell),
        )
        row = build_vector_at_camera(
            gs=ctx["gs"],
            beam=beam,
            axicon=ctx["axicon"],
            mask=ctx["mask"],
            realism=ctx["realism"],
            sim=ctx["sim"],
            z_cam=z_cam,
            mode=mode,
            use_eq16_qwp2=bool(use_eq16_qwp2),
            method=method,
            future_element=future_element,
            paper_replica=paper_replica,
        )
        row["ell"] = int(ell)
        rows.append(row)
    return rows


def plot_overlap_scan(z_values: np.ndarray, scores: np.ndarray, z_cam: float) -> None:
    plt.figure(figsize=(8.8, 4.4))
    plt.plot(z_values / mm, scores, color="#0b4f6c", linewidth=2.0)
    plt.axvline(z_cam / mm, color="k", linestyle="--", linewidth=1.1, label="z_cam")
    plt.xlabel("z [mm]")
    plt.ylabel("Ring-ROI corr(|Ex|^2, |Ey|^2)")
    plt.title("Camera-plane overlap scan")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_vector_grid(
    rows: Sequence[Dict[str, object]],
    extent_mm: List[float],
    mode: str,
    z_cam: float,
    lim_mm: float = 1.0,
) -> None:
    from vbb_study import vbb_style

    vbb_style.apply_style()
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, 5, figsize=(18.5, 2.9 * max(nrows, 1)), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])

    im = None
    labels = ["Total", "0 deg", "45 deg", "90 deg", "135 deg"]
    for i, row in enumerate(rows):
        total_I = row["total"]
        analyzer = row["analyzer"]
        if total_I is None or not analyzer:
            for j, label in enumerate(labels):
                ax = axes[i, j]
                ax.text(0.5, 0.5, "unreachable on\nCase-1 bench", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(label if i == 0 else "")
                ax.set_axis_off()
            continue
        analyzer_vmax = max(float(analyzer[a].max()) for a in (0, 45, 90, 135)) + 1e-15
        total_vmax = float(total_I.max()) + 1e-15
        data = [
            vbb_style.display_scale(total_I / total_vmax, gamma=0.45),
            vbb_style.display_scale(analyzer[0] / analyzer_vmax, gamma=0.45),
            vbb_style.display_scale(analyzer[45] / analyzer_vmax, gamma=0.45),
            vbb_style.display_scale(analyzer[90] / analyzer_vmax, gamma=0.45),
            vbb_style.display_scale(analyzer[135] / analyzer_vmax, gamma=0.45),
        ]
        for j, (img, label) in enumerate(zip(data, labels)):
            ax = axes[i, j]
            im = ax.imshow(
                img,
                interpolation="spline36",
                aspect=1.0,
                extent=extent_mm,
                cmap=vbb_style.INTENSITY_CMAP,
                origin="lower",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xlim(-lim_mm, lim_mm)
            ax.set_ylim(-lim_mm, lim_mm)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(f"l={row['ell']}\ny [mm]")
            else:
                ax.set_ylabel("")
            if i == nrows - 1:
                ax.set_xlabel("x [mm]")
            else:
                ax.set_xlabel("")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Display intensity [a.u.]")
    fig.suptitle(f"{mode.capitalize()} vector BG, z = {z_cam / mm:.1f} mm")
    plt.show()


@dataclass
class VoxelGridSpec:
    nx: int = 96
    ny: int = 96
    nz: int = 128
    x_lim: float = 0.30 * mm
    y_lim: float = 0.30 * mm
    z_min: float = 0.0 * mm
    z_max: float = 1.20 * mm


@dataclass
class MaterialSpec:
    name: str = "Fused silica (idealized)"
    linear_index: float = 1.45
    threshold_dose: float = 0.22
    incubation_exponent: float = 0.84
    multiphoton_order: int = 5
    multiphoton_strength: float = 1.0
    nonlinear_weight: float = 0.20
    response_order: float = 1.35
    saturation_dose: float = 5.0
    linear_absorption_np_per_m: float = 0.0
    thermal_blur_um: float = 0.0
    relaxation_per_pulse: float = 0.0
    report_level: float = 0.35


@dataclass
class PulseTrainSpec:
    n_pulses: int = 100
    pulse_energy_scale: float = 1.0
    snapshot_pulses: Tuple[int, ...] = (1, 10, 50, 100)
    include_nonlinear: bool = True


def make_fused_silica_material() -> MaterialSpec:
    return MaterialSpec()


def make_voxel_grid(spec: VoxelGridSpec) -> Dict[str, np.ndarray]:
    x = np.linspace(-spec.x_lim, spec.x_lim, int(spec.nx), dtype=float)
    y = np.linspace(-spec.y_lim, spec.y_lim, int(spec.ny), dtype=float)
    z = np.linspace(spec.z_min, spec.z_max, int(spec.nz), dtype=float)
    dx = float(x[1] - x[0]) if len(x) > 1 else 2.0 * spec.x_lim
    dy = float(y[1] - y[0]) if len(y) > 1 else 2.0 * spec.y_lim
    dz = float(z[1] - z[0]) if len(z) > 1 else max(spec.z_max - spec.z_min, 1.0)
    return {"x": x, "y": y, "z": z, "dx": dx, "dy": dy, "dz": dz}


def axisymmetric_volume_from_row(
    row: Dict[str, object],
    grid_spec: VoxelGridSpec,
    *,
    normalize: bool = True,
) -> Dict[str, object]:
    """Convert a scalar x-z slice into a manageable cylindrical 3D volume.

    The reconstruction assumes approximate cylindrical symmetry around the beam
    axis. That assumption is acceptable for the exploratory material notebook,
    but it should not be treated as a replacement for a full 3D vector solver.
    """
    grid = make_voxel_grid(grid_spec)
    x_src = np.asarray(row["x"], dtype=float)
    z_src = np.asarray(row.get("z_values", row.get("s")), dtype=float)
    xz_src = np.asarray(row["xs"], dtype=float)

    if xz_src.ndim != 2:
        raise ValueError("row['xs'] must be a 2D x-z intensity slice.")

    mid = int(np.argmin(np.abs(x_src)))
    r_src = np.abs(x_src[mid:])
    if np.any(np.diff(r_src) <= 0.0):
        raise ValueError("Source x grid is not strictly increasing away from the axis.")

    X, Y = np.meshgrid(grid["x"], grid["y"], indexing="xy")
    R = np.sqrt(X ** 2 + Y ** 2)
    intensity = np.zeros((len(grid["z"]), len(grid["y"]), len(grid["x"])), dtype=np.float32)

    for iz, z_t in enumerate(grid["z"]):
        if z_t <= z_src[0]:
            profile_full = xz_src[:, 0]
        elif z_t >= z_src[-1]:
            profile_full = xz_src[:, -1]
        else:
            j = int(np.searchsorted(z_src, z_t, side="right"))
            z0 = float(z_src[j - 1])
            z1 = float(z_src[j])
            w = 0.0 if z1 == z0 else (float(z_t) - z0) / (z1 - z0)
            profile_full = (1.0 - w) * xz_src[:, j - 1] + w * xz_src[:, j]

        profile = np.asarray(profile_full[mid:], dtype=float)
        plane = np.interp(R, r_src, profile, left=float(profile[0]), right=0.0)
        intensity[iz] = plane.astype(np.float32)

    if normalize:
        intensity /= float(intensity.max()) + 1e-15

    return {
        "grid": grid,
        "intensity": intensity,
        "source_x": x_src,
        "source_z": z_src,
    }


def accumulate_material_response(
    volume_case: Dict[str, object],
    material: MaterialSpec,
    pulse_train: PulseTrainSpec,
) -> Dict[str, object]:
    """Run an idealized pulse-accumulation model.

    The state variable is a normalized modification fraction in [0, 1].
    Threshold reduction with pulse count is included through an incubation law,
    while the nonlinear toggle boosts the local drive in already active regions.
    This is useful for screening trends, not for claiming predictive process
    windows without calibration.
    """
    grid = volume_case["grid"]
    intensity = np.asarray(volume_case["intensity"], dtype=float)
    z = np.asarray(grid["z"], dtype=float)
    attenuation = np.exp(-material.linear_absorption_np_per_m * (z - float(z[0])))[:, None, None]

    base_drive = material.multiphoton_strength * pulse_train.pulse_energy_scale * attenuation * intensity
    if pulse_train.include_nonlinear:
        base_drive = np.power(base_drive, max(int(material.multiphoton_order), 1))

    cumulative_dose = np.zeros_like(base_drive, dtype=float)
    modification = np.zeros_like(base_drive, dtype=float)
    final_view = np.zeros_like(base_drive, dtype=float)

    voxel_volume_mm3 = float(grid["dx"] * grid["dy"] * grid["dz"] / (mm ** 3))
    pulse_axis: List[int] = []
    volume_history_mm3: List[float] = []
    peak_history: List[float] = []
    snapshots: Dict[int, np.ndarray] = {}

    requested_snapshots = sorted({int(p) for p in pulse_train.snapshot_pulses if int(p) >= 1})
    if pulse_train.n_pulses not in requested_snapshots:
        requested_snapshots.append(int(pulse_train.n_pulses))

    for pulse_idx in range(1, int(pulse_train.n_pulses) + 1):
        threshold = material.threshold_dose * (pulse_idx ** (material.incubation_exponent - 1.0))
        local_gain = 1.0 + material.nonlinear_weight * modification if pulse_train.include_nonlinear else 1.0
        cumulative_dose += base_drive * local_gain

        if material.relaxation_per_pulse > 0.0:
            cumulative_dose *= max(0.0, 1.0 - material.relaxation_per_pulse)

        drive = np.maximum(cumulative_dose / max(threshold, 1e-20) - 1.0, 0.0)
        modification = 1.0 - np.exp(-np.power(drive, material.response_order))
        if material.saturation_dose > 0.0:
            modification = np.minimum(modification, 1.0 - np.exp(-cumulative_dose / material.saturation_dose))

        view = modification
        if material.thermal_blur_um > 0.0 and gaussian_filter is not None:
            sigma_xy = (material.thermal_blur_um * um) / max(grid["dx"], 1e-20)
            sigma_z = (material.thermal_blur_um * um) / max(grid["dz"], 1e-20)
            view = gaussian_filter(modification, sigma=(sigma_z, sigma_xy, sigma_xy), mode="nearest")

        pulse_axis.append(pulse_idx)
        volume_history_mm3.append(voxel_volume_mm3 * float(np.count_nonzero(view >= material.report_level)))
        peak_history.append(float(view.max()))
        final_view = view

        if pulse_idx in requested_snapshots:
            snapshots[pulse_idx] = view.astype(np.float32).copy()

    mask = final_view >= material.report_level
    if np.any(mask):
        z_mm = grid["z"][np.any(mask, axis=(1, 2))] / mm
        depth_mm = float(z_mm[-1] - z_mm[0]) if len(z_mm) > 1 else 0.0
    else:
        depth_mm = 0.0

    return {
        "grid": grid,
        "intensity": intensity.astype(np.float32),
        "modification": final_view.astype(np.float32),
        "snapshots": snapshots,
        "pulse_axis": np.asarray(pulse_axis, dtype=int),
        "volume_history_mm3": np.asarray(volume_history_mm3, dtype=float),
        "peak_history": np.asarray(peak_history, dtype=float),
        "material": material,
        "pulse_train": pulse_train,
        "modified_volume_mm3": float(volume_history_mm3[-1]) if volume_history_mm3 else 0.0,
        "peak_modification": float(peak_history[-1]) if peak_history else 0.0,
        "depth_extent_mm": depth_mm,
    }


def _snapshot_or_final(result: Dict[str, object], pulse: Optional[int]) -> np.ndarray:
    if pulse is None:
        return np.asarray(result["modification"], dtype=float)
    pulse_i = int(pulse)
    if pulse_i not in result["snapshots"]:
        raise KeyError(f"No stored snapshot for pulse {pulse_i}")
    return np.asarray(result["snapshots"][pulse_i], dtype=float)


def plot_material_history(result: Dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    axes[0].plot(result["pulse_axis"], result["volume_history_mm3"], color="#0b4f6c", linewidth=2.0)
    axes[0].set_title("Modified volume history")
    axes[0].set_xlabel("Pulse number")
    axes[0].set_ylabel("Volume above report level [mm^3]")
    axes[0].grid(alpha=0.25)

    axes[1].plot(result["pulse_axis"], result["peak_history"], color="#bc5090", linewidth=2.0)
    axes[1].set_title("Peak modification history")
    axes[1].set_xlabel("Pulse number")
    axes[1].set_ylabel("Peak modification fraction")
    axes[1].grid(alpha=0.25)
    plt.show()


def plot_material_slices(result: Dict[str, object], pulse: Optional[int] = None) -> None:
    volume = _snapshot_or_final(result, pulse)
    grid = result["grid"]
    x_mm = np.asarray(grid["x"], dtype=float) / mm
    y_mm = np.asarray(grid["y"], dtype=float) / mm
    z_mm = np.asarray(grid["z"], dtype=float) / mm

    iz = len(z_mm) // 2
    iy = len(y_mm) // 2
    ix = len(x_mm) // 2
    title_suffix = "final" if pulse is None else f"pulse {int(pulse)}"

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)

    im0 = axes[0].imshow(
        volume[iz],
        origin="lower",
        aspect="equal",
        cmap="inferno",
        extent=[x_mm[0], x_mm[-1], y_mm[0], y_mm[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title(f"XY slice ({title_suffix})")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")

    axes[1].imshow(
        volume[:, iy, :],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        extent=[x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("XZ slice")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("z [mm]")

    axes[2].imshow(
        volume[:, :, ix],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        extent=[y_mm[0], y_mm[-1], z_mm[0], z_mm[-1]],
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title("YZ slice")
    axes[2].set_xlabel("y [mm]")
    axes[2].set_ylabel("z [mm]")

    cbar = fig.colorbar(im0, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Modification fraction")
    plt.show()


def plot_material_point_cloud(
    result: Dict[str, object],
    pulse: Optional[int] = None,
    *,
    level: Optional[float] = None,
    max_points: int = 6000,
) -> None:
    volume = _snapshot_or_final(result, pulse)
    grid = result["grid"]
    level = result["material"].report_level if level is None else float(level)

    mask = volume >= level
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("No voxels are above the requested level.")
        return

    if len(coords) > int(max_points):
        stride = int(np.ceil(len(coords) / float(max_points)))
        coords = coords[::stride]

    z = grid["z"][coords[:, 0]] / mm
    y = grid["y"][coords[:, 1]] / mm
    x = grid["x"][coords[:, 2]] / mm
    val = volume[coords[:, 0], coords[:, 1], coords[:, 2]]

    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=val,
            colorscale='Inferno',
            opacity=0.65,
            colorbar=dict(title='Modification fraction')
        )
    )])
    
    fig.update_layout(
        title="Modification cloud preview",
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title="z [mm]"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.show()
