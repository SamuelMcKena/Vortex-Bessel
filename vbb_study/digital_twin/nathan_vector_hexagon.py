"""Nathan six-sector vector-hexagon routes inside the existing digital twin.

This module is intentionally digital-twin-facing glue.  It reuses the existing
Stage 7 vector-arm, vector ASM, SLM realism, hexagon metrics, and vector axicon
code, but defines one canonical Nathan target field shared by all routes.  The
main study geometry is inherited from :class:`vbb_study.config.TwinConfig`;
Nathan's original free-space millimetre-scale script is only a convention
reference for the radial/azimuthal sector mechanism.

Model boundary
--------------
The output is an air-side optical prediction.  It is not a camera model, not a
material-response model.  The current shared downstream propagation uses the
repository's vector ASM plus existing ObjectiveMap scalar focusing per
transverse component; this module also provides a scoped Nathan-only
vectorial pupil-spectrum reference for validating that focal bridge.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from vbb_study.config import EPS, TwinConfig
from vbb_study.design import J0_FIRST_ZERO, compute_design_from_targets, default_config
from vbb_study.digital_twin.nathan_literal_source_port import (
    NathanLiteralSourceConfig,
    apply_source_axicon,
    make_segmented_ra_input as literal_make_segmented_ra_input,
    make_source_grid as literal_make_source_grid,
    propagate_source_vector_asm,
    run_literal_source_port,
)
from vbb_study.equations.fields import fft2c, ifft2c, make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl, focus_to_focal_plane
from vbb_study.equations.vector_jones import circular_to_linear, stokes_from_linear_components
from vbb_study.slm_model import field_power
from vbb_study.vbb_hexagon_metrics import hexagon_acceptance, sixfold_from_intensity
from vbb_study.vbb_polarized_train import retarder_jones
from vbb_study.vector_arm_chain import (
    default_vector_arm_grid,
    gaussian_envelope,
    headless_angle_delta,
    local_polarization_angle,
    run_vector_arm,
)
from vbb_study.vector_arm_config import SLMPanelConfig, VectorArmConfig
from vbb_study.vector_arm_metrics import h6_z_curve, sample_ring_profile
from vbb_study.vector_axicon import (
    VectorAxiconResult,
    apply_vector_axicon,
    assert_locked_kr_fingerprint,
    fresnel_sp_amplitudes,
    resolve_vector_axicon_parameters,
    run_vector_axicon_to_surface,
)
from vbb_study.vector_field import VectorField, propagate_vector_asm, spectral_transversality_residual
from vbb_study.vector_fourier import apply_fourier_iris, carrier_collinearity_report

STAGE = "nathan_vector_hexagon_digital_twin"
MODEL_STATUS = "optical_prediction"
FINAL_EXPORT_ALLOWED = False
TWOPI = 2.0 * np.pi
DEFAULT_FAST_N = 128
DEFAULT_FAST_Z_PLANES = 9
DEFAULT_WALL_CONTINUITY_LEVEL = 0.25
DEFAULT_WALL_CONTINUITY_MIN_FRACTION = 0.55
VISUAL_LADDER_STATUS = "visual_gate_only_no_route_ranking"
DEFAULT_NATHAN_FIGURE4_REFERENCE = (
    Path(__file__).resolve().parents[3] / "outputs" / "reference" / "nathan_marco_report_figure4_page7_crop.png"
)
V0_ALLOWED_VISUAL_VERDICTS = ("PASS", "PARTIAL", "FAIL", "UNRESOLVED")
FOCUS_VALIDATION_UNRESOLVED_STATEMENT = (
    "The current model preserves vector components through the axicon and later vector ASM, "
    "but its scalar per-component focal transform has not been validated for the NA = 0.45 "
    "vector-field problem. The micro-scale outcome is therefore unresolved pending a vectorial "
    "pupil-to-focus reference and a multiscale sampling check."
)


@dataclass(frozen=True)
class NathanSourceParityConfig:
    """Frozen source-convention controls for the isolated V0 visual gate."""

    wavelength_m: float = 1030e-9
    beam_radius_m: float = 2.0e-3
    axicon_n: float = 1.458
    medium_n: float = 1.0
    axicon_apex_angle_deg: float = 176.0
    grid_n: int = 1024
    window_m: float = 10.0e-3
    n_pairs: int = 3
    sector_theta_rad: float = np.pi / 3.0
    sector_rotation_rad: float = 0.0
    z_reference_m: float = 60.0e-3
    z_start_m: float = 0.1e-3
    z_end_m: float = 290.0e-3
    z_planes: int = 61
    z_span_m: float | None = None

    @property
    def axicon_base_angle_rad(self) -> float:
        return 0.5 * np.deg2rad(max(0.0, 180.0 - float(self.axicon_apex_angle_deg)))

    @property
    def k_r_m_inv(self) -> float:
        return float(TWOPI / self.wavelength_m * (self.axicon_n - self.medium_n) * np.tan(self.axicon_base_angle_rad))


@dataclass(frozen=True)
class VisualLadderStageResult:
    """One fixed-plane visual ladder stage, without metric-led plane selection."""

    stage_id: str
    title: str
    z_values_m: np.ndarray
    reference_index: int
    grid: Mapping[str, Any]
    intensity_stack: np.ndarray
    comparison_stack: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    verdict: str = "VISUAL_REVIEW_REQUIRED"


@dataclass(frozen=True)
class VisualLadderReport:
    """Compact visual reproduction ladder report."""

    source_parity: Mapping[str, Any]
    stages: tuple[VisualLadderStageResult, ...]
    stopping_result: str
    output_paths: Mapping[str, Path] = field(default_factory=dict)
    status_report: Mapping[str, Any] = field(default_factory=dict)
    convergence_report: Mapping[str, Any] | None = None


def _panel_from_existing_slm(template: SLMPanelConfig, twin: TwinConfig) -> SLMPanelConfig:
    """Return vector-arm panel settings inherited from the study SLM config."""

    phase_levels = max(2, int(2 ** int(twin.slm.phase_bits)))
    return replace(
        template,
        n_x=int(twin.slm.resolution_x),
        n_y=int(twin.slm.resolution_y),
        pitch_m=float(twin.slm.pixel_pitch_m),
        phase_levels=phase_levels,
        fill_factor=float(twin.slm.fill_factor),
        carrier_lp_per_mm=abs(float(twin.slm.carrier_lpmm)),
    )


def vector_config_from_existing_twin(
    twin: TwinConfig,
    template: VectorArmConfig | None = None,
    *,
    ideal_components: bool | None = None,
) -> VectorArmConfig:
    """Resolve vector-generator settings without replacing the study geometry."""

    base = template or VectorArmConfig(ideal_components=True if ideal_components is None else bool(ideal_components))
    if ideal_components is not None:
        base = replace(base, ideal_components=bool(ideal_components))
    return replace(
        base,
        wavelength_m=float(twin.laser.wavelength_m),
        pulse_duration_s=float(twin.laser.pulse_duration_s),
        waist_m=float(twin.laser.beam_radius_on_slm_m),
        slm1=_panel_from_existing_slm(base.slm1, twin),
        slm2=_panel_from_existing_slm(base.slm2, twin),
    )


def _twin_with_axial_points(twin: TwinConfig, axial_points: int) -> TwinConfig:
    """Return the existing twin with only the z-stack sampling count changed."""

    return replace(twin, grid=replace(twin.grid, axial_points=int(axial_points)))


@dataclass(frozen=True)
class NathanHexagonConfig:
    """Nathan sector controls inserted into an existing Digital Twin baseline."""

    twin: TwinConfig = field(default_factory=lambda: default_config("fast"))
    baseline_preset: str = "fast"
    vector: VectorArmConfig = field(default_factory=lambda: VectorArmConfig(ideal_components=True))
    grid_n: int = DEFAULT_FAST_N
    z_planes: int = DEFAULT_FAST_Z_PLANES
    z_span_factor: float = 1.0
    angular_samples: int = 1440
    route_grid_side_factor: float = 3.6

    @classmethod
    def fast(cls, **overrides: Any) -> "NathanHexagonConfig":
        return cls.from_existing_digital_twin_baseline("fast", **{
            "grid_n": 128,
            "z_planes": 9,
            "angular_samples": 1440,
            "vector": VectorArmConfig(ideal_components=True),
            **overrides,
        })

    @classmethod
    def paper(cls, **overrides: Any) -> "NathanHexagonConfig":
        return cls.from_existing_digital_twin_baseline("publication", **{
            "grid_n": 512,
            "z_planes": 41,
            "angular_samples": 4096,
            "vector": VectorArmConfig(ideal_components=True),
            **overrides,
        })

    @classmethod
    def from_existing_digital_twin_baseline(
        cls,
        baseline: str | TwinConfig = "fast",
        **overrides: Any,
    ) -> "NathanHexagonConfig":
        """Build the Nathan add-on config from the pre-existing twin baseline."""

        if isinstance(baseline, TwinConfig):
            twin = baseline
            preset = str(overrides.pop("baseline_preset", getattr(twin.grid, "label", "custom")))
        else:
            preset = str(baseline)
            twin = default_config(preset)
        twin = overrides.pop("twin", twin)
        vector_template = overrides.pop("vector", VectorArmConfig(ideal_components=True))
        vector = vector_config_from_existing_twin(twin, vector_template)
        return cls(twin=twin, baseline_preset=preset, vector=vector, **overrides)


NathanMicroHexagonConfig = NathanHexagonConfig


@dataclass(frozen=True)
class PatternedHWPConfig:
    """Controls for the patterned half-wave-plate route."""

    case: str = "continuous"
    tiles_per_sector: int = 1
    seam_width_rad: float = 0.0
    central_defect_radius_m: float = 0.0
    fast_axis_error_rad: float = 0.0
    retardance_error_rad: float = 0.0
    aperture_radius_m: float | None = None
    transmission: float = 1.0


@dataclass(frozen=True)
class FieldComparison:
    """Route-to-reference comparison after removing one constant piston only."""

    piston_rad: float
    complex_overlap: float
    normalized_rms_error: float
    normalized_max_error: float
    stokes_rms_error: float
    angle_rms_rad: float
    power_ratio: float


@dataclass(frozen=True)
class RouteFieldResult:
    """One field at the common vector-generator handoff plane."""

    route_id: str
    field: VectorField
    target: VectorField
    comparison: FieldComparison
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HexagonPlaneMetrics:
    """Raw hollow-hexagon metrics for one propagated z plane."""

    z_m: float
    ring_radius_m: float
    h6: float
    order6_over_order0: float
    order6_over_non_dc: float
    wall_continuity: float
    core_darkness: float
    wall_power_fraction: float
    sidelobe_fraction: float
    hexagon_acceptance_pass: bool


@dataclass(frozen=True)
class RoutePropagationResult:
    """Shared axicon propagation result plus per-plane metrics."""

    route: RouteFieldResult
    axicon: VectorAxiconResult
    metrics: tuple[HexagonPlaneMetrics, ...]
    output_overlap_to_canonical: tuple[float, ...] = ()


@dataclass(frozen=True)
class RouteComparisonReport:
    """Route comparison at vector handoff and through the inherited twin optics."""

    config: NathanHexagonConfig
    input_rows: tuple[dict[str, Any], ...]
    output_rows: tuple[dict[str, Any], ...]
    controls: Mapping[str, Any]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class DownstreamRouteResult:
    """One S0/S1/V1 downstream-model result for one handoff field."""

    route_id: str
    control_id: str
    z_values_m: np.ndarray
    intensity_stack: np.ndarray
    metrics: tuple[HexagonPlaneMetrics, ...]
    ex_energy_fraction: tuple[float, ...]
    ey_energy_fraction: tuple[float, ...]
    ez_energy_fraction: tuple[float, ...]
    transversality_residual: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    stokes_maps: Mapping[str, np.ndarray] = field(default_factory=dict)


def default_nathan_grid(config: NathanHexagonConfig | None = None) -> dict[str, Any]:
    """Return the inherited vector-generator grid for Nathan route comparisons."""

    cfg = config or NathanHexagonConfig.fast()
    return default_vector_arm_grid(cfg.vector, int(cfg.grid_n))


def nathan_sector_mask(grid: Mapping[str, Any], config: NathanHexagonConfig | None = None) -> np.ndarray:
    """Return integer sector labels: 0 radial, 1 azimuthal."""

    cfg = config or NathanHexagonConfig.fast()
    theta = np.asarray(grid["PHI"], dtype=float)
    cell = TWOPI / float(cfg.vector.n_pairs)
    local = np.mod(theta - float(cfg.vector.sector_rotation_rad), cell)
    return (local < float(cfg.vector.sector_duty) * cell).astype(np.int8)


def nathan_alpha(
    grid: Mapping[str, Any],
    config: NathanHexagonConfig | None = None,
    *,
    control: str = "segmented",
) -> np.ndarray:
    """Return the local linear-polarisation angle alpha(theta)."""

    theta = np.asarray(grid["PHI"], dtype=float)
    key = str(control).lower().strip()
    if key in {"segmented", "nathan"}:
        return theta + np.where(nathan_sector_mask(grid, config) == 1, 0.5 * np.pi, 0.0)
    if key in {"radial", "all_radial"}:
        return theta
    if key in {"azimuthal", "all_azimuthal"}:
        return theta + 0.5 * np.pi
    if key in {"linear_x", "scalar_x"}:
        return np.zeros_like(theta)
    raise ValueError(f"Unsupported target/control kind: {control!r}")


def canonical_target_field(
    config: NathanHexagonConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
    control: str = "segmented",
) -> VectorField:
    """Build the canonical Nathan field at the vector-generator handoff plane."""

    cfg = config or NathanHexagonConfig.fast()
    grid_dict = dict(default_nathan_grid(cfg) if grid is None else grid)
    alpha = nathan_alpha(grid_dict, cfg, control=control)
    amp = gaussian_envelope(grid_dict, cfg.vector)
    return VectorField(
        ex=amp * np.cos(alpha),
        ey=amp * np.sin(alpha),
        ez=np.zeros_like(amp, dtype=complex),
        grid=grid_dict,
        wavelength_m=cfg.vector.wavelength_m,
        medium_index=1.0,
        metadata={
            "stage": STAGE,
            "field": f"canonical_{control}",
            "geometry_source": "existing_digital_twin_baseline",
            "envelope": "TwinConfig.laser Gaussian exp(-(r/w0)^2)",
            "w0_m": float(cfg.vector.waist_m),
            "wavelength_m": float(cfg.vector.wavelength_m),
        },
    )


def nathan_literal_segmented_ra_input(
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    beam_radius_m: float,
    n_pairs: int = 3,
    sector_theta_rad: float = np.pi / 3.0,
    sector_rotation_rad: float = 0.0,
) -> tuple[VectorField, np.ndarray]:
    """Literal Nathan segmented RA input formula for V0 source parity."""

    theta = np.asarray(grid["PHI"], dtype=float)
    R = np.asarray(grid["R"], dtype=float)
    cell_angle = TWOPI / float(n_pairs)
    phi = np.mod(theta - float(sector_rotation_rad), TWOPI)
    phi_cell = np.mod(phi, cell_angle)
    radial_mask = phi_cell >= (cell_angle - float(sector_theta_rad))
    phi0_map = np.where(radial_mask, 0.0, 0.5 * np.pi)
    u0 = np.exp(-(R**2) / max(float(beam_radius_m), EPS) ** 2)
    field = VectorField(
        ex=u0 * np.cos(theta + phi0_map),
        ey=u0 * np.sin(theta + phi0_map),
        ez=np.zeros_like(u0, dtype=complex),
        grid=grid,
        wavelength_m=float(wavelength_m),
        medium_index=1.0,
        metadata={
            "stage": "V0_literal_nathan_segmented_ra_input",
            "n_pairs": int(n_pairs),
            "sector_theta_rad": float(sector_theta_rad),
            "sector_rotation_rad": float(sector_rotation_rad),
            "beam_radius_m": float(beam_radius_m),
        },
    )
    return field, radial_mask


def source_parity_grid(config: NathanSourceParityConfig | None = None) -> dict[str, Any]:
    """Return the frozen V0 source-parity grid (Nathan's axis-sampled convention).

    V0 must use Nathan's ``x = -L + arange*dx`` centring, which samples the optical
    axis exactly at index ``n//2``.  The project's default ``make_xy_grid`` straddles
    zero (``arange - n/2 + 0.5``); on that grid the r=0 radial/azimuthal polarisation
    singularity of the segmented field is carried by four uncancelled near-axis
    pixels that inject a spurious bright on-axis core after propagation, so V0B
    would not reproduce Nathan's dark-core hexagonal Bessel beam.  This helper is
    built inline (not via ``make_xy_grid`` and not via the V0A literal port) so the
    V0B numerical path stays independent while adopting the correct source centring.
    See docs/53 for the grid-centring sensitivity evidence.
    """

    cfg = config or NathanSourceParityConfig()
    n = int(cfg.grid_n)
    dx = float(cfg.window_m) / float(n)
    x = (np.arange(n, dtype=float) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    return {
        "N": n,
        "dx": dx,
        "x": x,
        "y": x,
        "X": X,
        "Y": Y,
        "R": np.hypot(X, Y),
        "PHI": np.arctan2(Y, X),
        "FX": FX,
        "FY": FY,
    }


def _v0_z_values(config: NathanSourceParityConfig) -> np.ndarray:
    """Return V0 z planes, inserting the declared reference plane exactly."""

    planes = max(2, int(config.z_planes))
    if config.z_span_m is not None:
        z_values = np.linspace(0.0, float(config.z_span_m), planes)
    else:
        z_values = np.linspace(float(config.z_start_m), float(config.z_end_m), planes)
    ref = float(config.z_reference_m)
    if float(np.min(z_values)) - EPS <= ref <= float(np.max(z_values)) + EPS and not np.any(np.isclose(z_values, ref, rtol=0.0, atol=1e-15)):
        z_values = np.sort(np.append(z_values, ref))
    return z_values.astype(float)


def _parameter_close(actual: float, expected: float, *, rtol: float = 1e-12, atol: float = 1e-15) -> bool:
    return bool(np.isclose(float(actual), float(expected), rtol=float(rtol), atol=float(atol)))


def v0_source_parameter_parity(config: NathanSourceParityConfig | None = None) -> dict[str, Any]:
    """Report whether V0 is using Nathan's source-style reproduction parameters."""

    cfg = config or NathanSourceParityConfig()
    expected: dict[str, float | int] = {
        "window_m": 10.0e-3,
        "grid_n_minimum": 1024,
        "beam_radius_m": 2.0e-3,
        "wavelength_m": 1030e-9,
        "axicon_n": 1.458,
        "medium_n": 1.0,
        "axicon_apex_angle_deg": 176.0,
        "axicon_base_angle_deg": 2.0,
        "z_start_m": 0.1e-3,
        "z_end_m": 290.0e-3,
        "z_reference_m": 60.0e-3,
        "z_planes_minimum": 60,
        "n_pairs": 3,
        "sector_theta_rad": np.pi / 3.0,
    }
    checks = {
        "window_m": _parameter_close(cfg.window_m, float(expected["window_m"])),
        "grid_n_minimum": int(cfg.grid_n) >= int(expected["grid_n_minimum"]),
        "beam_radius_m": _parameter_close(cfg.beam_radius_m, float(expected["beam_radius_m"])),
        "wavelength_m": _parameter_close(cfg.wavelength_m, float(expected["wavelength_m"])),
        "axicon_n": _parameter_close(cfg.axicon_n, float(expected["axicon_n"])),
        "medium_n": _parameter_close(cfg.medium_n, float(expected["medium_n"])),
        "axicon_apex_angle_deg": _parameter_close(cfg.axicon_apex_angle_deg, float(expected["axicon_apex_angle_deg"])),
        "axicon_base_angle_deg": _parameter_close(np.rad2deg(cfg.axicon_base_angle_rad), float(expected["axicon_base_angle_deg"])),
        "z_start_m": cfg.z_span_m is None and _parameter_close(cfg.z_start_m, float(expected["z_start_m"])),
        "z_end_m": cfg.z_span_m is None and _parameter_close(cfg.z_end_m, float(expected["z_end_m"])),
        "z_reference_m": _parameter_close(cfg.z_reference_m, float(expected["z_reference_m"])),
        "z_planes_minimum": int(_v0_z_values(cfg).size) >= int(expected["z_planes_minimum"]),
        "n_pairs": int(cfg.n_pairs) == int(expected["n_pairs"]),
        "sector_theta_rad": _parameter_close(cfg.sector_theta_rad, float(expected["sector_theta_rad"])),
    }
    dx_m = float(cfg.window_m) / max(int(cfg.grid_n), 1)
    return {
        "status": "source_parameters_match_nathan" if all(checks.values()) else "source_parameters_not_nathan_style",
        "checks": checks,
        "actual": {
            "window_m": float(cfg.window_m),
            "grid_n": int(cfg.grid_n),
            "dx_m": dx_m,
            "dx_um": dx_m / 1e-6,
            "beam_radius_m": float(cfg.beam_radius_m),
            "wavelength_m": float(cfg.wavelength_m),
            "axicon_n": float(cfg.axicon_n),
            "medium_n": float(cfg.medium_n),
            "axicon_apex_angle_deg": float(cfg.axicon_apex_angle_deg),
            "axicon_base_angle_deg": float(np.rad2deg(cfg.axicon_base_angle_rad)),
            "z_start_m": float(_v0_z_values(cfg)[0]),
            "z_end_m": float(_v0_z_values(cfg)[-1]),
            "z_reference_m": float(cfg.z_reference_m),
            "z_plane_count": int(_v0_z_values(cfg).size),
            "n_pairs": int(cfg.n_pairs),
            "sector_theta_rad": float(cfg.sector_theta_rad),
        },
        "expected": expected,
    }


def v0_numerical_resolution_status(config: NathanSourceParityConfig | None = None) -> dict[str, Any]:
    """Return the V0 sampling status without making a visual pass/fail claim."""

    cfg = config or NathanSourceParityConfig()
    dx_m = float(cfg.window_m) / max(int(cfg.grid_n), 1)
    n = int(cfg.grid_n)
    parameter_parity = v0_source_parameter_parity(cfg)
    if n >= 1024 and parameter_parity["status"] == "source_parameters_match_nathan":
        status = "primary_v0_resolution"
    elif n >= 512 and _parameter_close(cfg.window_m, 10.0e-3):
        status = "convergence_support_resolution_not_primary"
    else:
        status = "diagnostic_only_too_coarse_for_v0_decision"
    return {
        "status": status,
        "grid_n": n,
        "window_m": float(cfg.window_m),
        "dx_m": dx_m,
        "dx_um": dx_m / 1e-6,
        "primary_required_grid_n": 1024,
        "primary_required_window_m": 10.0e-3,
        "n192_warning": bool(n <= 192),
    }


def source_parity_comparison(config: NathanSourceParityConfig | None = None) -> dict[str, Any]:
    """Compare canonical target arrays against Nathan's literal source formula."""

    cfg = config or NathanSourceParityConfig()
    grid = source_parity_grid(cfg)
    literal, radial_mask = nathan_literal_segmented_ra_input(
        grid,
        wavelength_m=cfg.wavelength_m,
        beam_radius_m=cfg.beam_radius_m,
        n_pairs=cfg.n_pairs,
        sector_theta_rad=cfg.sector_theta_rad,
        sector_rotation_rad=cfg.sector_rotation_rad,
    )
    vector = VectorArmConfig(
        wavelength_m=cfg.wavelength_m,
        waist_m=cfg.beam_radius_m,
        n_pairs=cfg.n_pairs,
        sector_duty=float(cfg.sector_theta_rad) / (TWOPI / float(cfg.n_pairs)),
        sector_rotation_rad=cfg.sector_rotation_rad,
        ideal_components=True,
    )
    base_twin = default_config("fast")
    source_twin = replace(
        base_twin,
        laser=replace(base_twin.laser, wavelength_m=float(cfg.wavelength_m), beam_radius_on_slm_m=float(cfg.beam_radius_m)),
    )
    canonical_cfg = NathanHexagonConfig.from_existing_digital_twin_baseline(source_twin, grid_n=cfg.grid_n, vector=vector)
    canonical = canonical_target_field(canonical_cfg, grid=grid)
    canonical_radial = nathan_sector_mask(grid, canonical_cfg) == 0
    centre_mask = np.asarray(grid["R"], dtype=float) > 2.0 * float(grid["dx"])
    ex_diff = np.asarray(canonical.ex - literal.ex)
    ey_diff = np.asarray(canonical.ey - literal.ey)
    return {
        "status": "source_parity_exact" if bool(np.array_equal(canonical_radial, radial_mask)) and float(np.max(np.abs(ex_diff[centre_mask]))) < 1e-12 and float(np.max(np.abs(ey_diff[centre_mask]))) < 1e-12 else "source_parity_mismatch",
        "radial_mask_equal": bool(np.array_equal(canonical_radial, radial_mask)),
        "max_ex_abs_diff_away_from_centre": float(np.max(np.abs(ex_diff[centre_mask]))) if np.any(centre_mask) else 0.0,
        "max_ey_abs_diff_away_from_centre": float(np.max(np.abs(ey_diff[centre_mask]))) if np.any(centre_mask) else 0.0,
        "literal_power": float(literal.power),
        "canonical_power": float(canonical.power),
        "relative_power_error": float(abs(canonical.power - literal.power) / max(literal.power, EPS)),
        "sector_rotation_rad": float(cfg.sector_rotation_rad),
        "centre_pixel_radius_m": float(np.min(np.asarray(grid["R"], dtype=float))),
        "grid_n": int(cfg.grid_n),
        "window_m": float(cfg.window_m),
    }


def _apply_free_space_vector_axicon(
    field: VectorField,
    *,
    n_axicon: float,
    n_medium: float,
    base_angle_rad: float,
    aperture_radius_m: float | None = None,
) -> tuple[VectorField, dict[str, Any]]:
    """Apply a thin vector axicon for free-space V0/V1 visual gates."""

    grid = field.grid
    R = np.asarray(grid["R"], dtype=float)
    phi = np.asarray(grid["PHI"], dtype=float)
    er_x = np.cos(phi)
    er_y = np.sin(phi)
    ep_x = -np.sin(phi)
    ep_y = np.cos(phi)
    er = field.ex * er_x + field.ey * er_y
    ephi = field.ex * ep_x + field.ey * ep_y
    t_entry, t_p, t_s = fresnel_sp_amplitudes(float(n_axicon), float(n_medium), float(base_angle_rad))
    k_r = float(TWOPI / field.wavelength_m * (float(n_axicon) - float(n_medium)) * np.tan(float(base_angle_rad)))
    phase = np.exp(-1j * abs(k_r) * R)
    if aperture_radius_m is None:
        aperture = np.ones_like(R, dtype=float)
    else:
        aperture = (R <= float(aperture_radius_m)).astype(float)
    er_out = t_entry * t_p * er * phase * aperture
    ephi_out = t_entry * t_s * ephi * phase * aperture
    out = VectorField(
        ex=er_out * er_x + ephi_out * ep_x,
        ey=er_out * er_y + ephi_out * ep_y,
        ez=field.ez * phase * aperture * t_entry * 0.5 * (t_p + t_s),
        grid=grid,
        wavelength_m=field.wavelength_m,
        medium_index=float(n_medium),
        metadata={**dict(field.metadata), "stage": "visual_ladder_after_free_space_vector_axicon"},
    )
    return out, {
        "n_axicon": float(n_axicon),
        "n_medium": float(n_medium),
        "base_angle_rad": float(base_angle_rad),
        "base_angle_deg": float(np.rad2deg(base_angle_rad)),
        "k_r_m_inv": k_r,
        "t_entry_abs": float(abs(t_entry)),
        "t_p_abs": float(abs(t_p)),
        "t_s_abs": float(abs(t_s)),
        "aperture_radius_m": None if aperture_radius_m is None else float(aperture_radius_m),
    }


def _free_space_intensity_stack(
    field: VectorField,
    z_values_m: Sequence[float],
    *,
    return_fields: bool = False,
) -> tuple[np.ndarray, tuple[VectorField, ...]]:
    """Propagate a V0/V1 field over many z planes using one spectral projection."""

    grid = field.grid
    z_values = np.asarray(z_values_m, dtype=float)
    k = TWOPI * float(field.medium_index) / float(field.wavelength_m)
    kx = TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = TWOPI * np.asarray(grid["FY"], dtype=float)
    kz = np.sqrt((k * k - kx * kx - ky * ky) + 0j)
    kz = np.where(np.imag(kz) < 0.0, -kz, kz)
    ax = fft2c(field.ex)
    ay = fft2c(field.ey)
    az = fft2c(field.ez)
    sx = kx / max(float(k), EPS)
    sy = ky / max(float(k), EPS)
    sz = kz / max(float(k), EPS)
    dot = sx * ax + sy * ay + sz * az
    ax_p = ax - sx * dot
    ay_p = ay - sy * dot
    az_p = az - sz * dot
    stack = np.empty((z_values.size,) + field.ex.shape, dtype=np.float32)
    planes: list[VectorField] = []
    for idx, z_m in enumerate(z_values):
        transfer = np.exp(1j * kz * float(z_m))
        ex = ifft2c(ax_p * transfer)
        ey = ifft2c(ay_p * transfer)
        ez = ifft2c(az_p * transfer)
        stack[idx] = (np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2).astype(np.float32)
        if return_fields:
            planes.append(
                VectorField(
                    ex=ex,
                    ey=ey,
                    ez=ez,
                    grid=field.grid,
                    wavelength_m=field.wavelength_m,
                    medium_index=field.medium_index,
                    metadata={**dict(field.metadata), "vector_asm_z_m": float(z_m)},
                )
            )
    return stack, tuple(planes)


def _nearest_z_index(z_values_m: Sequence[float], reference_z_m: float) -> int:
    z = np.asarray(z_values_m, dtype=float)
    return int(np.argmin(np.abs(z - float(reference_z_m)))) if z.size else 0


def _v0_plane_diagnostics(intensity: np.ndarray, grid: Mapping[str, Any]) -> dict[str, Any]:
    """Return morphology diagnostics for the declared V0 reference plane."""

    plane = np.asarray(intensity, dtype=float)
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    ring_radius = _peak_annulus_radius(plane, grid)
    ring_mask = (R >= 0.75 * ring_radius) & (R <= 1.25 * ring_radius)
    profile = sample_ring_profile(plane, grid, ring_radius, angular_samples=1440)
    sixfold = sixfold_from_intensity(plane, R, PHI, ring_radius, angular_bins=720)
    centre = tuple(s // 2 for s in plane.shape)
    ring_peak = float(np.nanmax(plane[ring_mask])) if np.any(ring_mask) else float(np.nanmax(plane))
    total = float(np.sum(plane))
    wall_power = float(np.sum(plane[ring_mask])) if np.any(ring_mask) else 0.0
    return {
        "ring_radius_m": float(ring_radius),
        "ring_radius_um": float(ring_radius / 1e-6),
        "central_core_darkness": float(plane[centre] / max(ring_peak, EPS)),
        "sixfold_order6_over_order0": float(sixfold["order6_over_order0"]),
        "sixfold_order6_over_non_dc": float(sixfold["order6_over_non_dc"]),
        "sixfold_dominant_order": int(sixfold["dominant_order"]),
        "wall_continuity": _wall_continuity_from_profile(profile),
        "wall_power_fraction": float(wall_power / max(total, EPS)),
    }


def _automated_v0_visual_verdict(diagnostics: Mapping[str, Any]) -> str:
    """Suggest an allowed V0 visual verdict from fixed diagnostics only."""

    sixfold_non_dc = float(diagnostics.get("sixfold_order6_over_non_dc", 0.0))
    sixfold_dc = float(diagnostics.get("sixfold_order6_over_order0", 0.0))
    dominant_order = int(diagnostics.get("sixfold_dominant_order", 0))
    core = float(diagnostics.get("central_core_darkness", 1.0))
    continuity = float(diagnostics.get("wall_continuity", 0.0))
    wall_power = float(diagnostics.get("wall_power_fraction", 0.0))
    has_sixfold = dominant_order in {6, 12, 18} or sixfold_non_dc >= 0.18 or sixfold_dc >= 0.04
    hollow = core <= 0.55
    if has_sixfold and hollow and continuity >= 0.45 and wall_power >= 0.15:
        return "PASS"
    if has_sixfold or (hollow and continuity >= 0.25):
        return "PARTIAL"
    return "FAIL"


def v0_source_output_parity_report(
    stage: VisualLadderStageResult,
    *,
    operator_visual_assessment: str | None = None,
    convergence_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report V0 parity gates without letting input parity stand in for output parity."""

    assessment = None if operator_visual_assessment is None else str(operator_visual_assessment).strip().upper()
    if assessment is not None and assessment not in V0_ALLOWED_VISUAL_VERDICTS:
        raise ValueError(f"operator_visual_assessment must be one of {V0_ALLOWED_VISUAL_VERDICTS}.")
    metadata = dict(stage.metadata)
    suggested = str(metadata.get("suggested_verdict", stage.verdict)).strip().upper()
    visual_verdict = assessment or suggested
    numerical = metadata.get("numerical_resolution_status", {})
    if convergence_report is not None:
        numerical = {
            **dict(numerical),
            "convergence_status": convergence_report.get("status"),
            "convergence_reference_grid_n": convergence_report.get("reference_grid_n"),
        }
    source_parity = dict(metadata.get("source_parity", {}))
    input_status = "exact" if source_parity.get("status") == "source_parity_exact" else "fail"
    if source_parity.get("status") not in {"source_parity_exact", "source_parity_mismatch"}:
        input_status = "partial"
    return {
        "input_array_parity": {**source_parity, "input_array_parity": input_status},
        "source_parameter_parity": metadata.get("source_parameter_parity", {}),
        "propagated_output_parity": visual_verdict.lower(),
        "propagated_output_visual_verdict": visual_verdict,
        "suggested_verdict": suggested,
        "operator_visual_assessment": assessment,
        "automated_diagnostic_status": metadata.get("automated_diagnostic_status", {}),
        "numerical_resolution_status": numerical,
        "propagated_output_verdict_is_separate_from_input_parity": True,
    }


def _literal_config_from_source_config(config: NathanSourceParityConfig) -> NathanLiteralSourceConfig:
    z_start = 0.0 if config.z_span_m is not None else float(config.z_start_m)
    z_end = float(config.z_span_m) if config.z_span_m is not None else float(config.z_end_m)
    return NathanLiteralSourceConfig(
        wavelength_m=float(config.wavelength_m),
        beam_radius_m=float(config.beam_radius_m),
        axicon_n=float(config.axicon_n),
        medium_n=float(config.medium_n),
        axicon_apex_angle_deg=float(config.axicon_apex_angle_deg),
        grid_n=int(config.grid_n),
        window_m=float(config.window_m),
        n_pairs=int(config.n_pairs),
        sector_theta_rad=float(config.sector_theta_rad),
        sector_rotation_rad=float(config.sector_rotation_rad),
        z_start_m=z_start,
        z_end_m=z_end,
        z_reference_m=float(config.z_reference_m),
        z_planes=int(config.z_planes),
    )


def run_v0_source_parity_visual_control(config: NathanSourceParityConfig | None = None) -> VisualLadderStageResult:
    """Run V0: Nathan source-convention parity without ObjectiveMap/focus."""

    cfg = config or NathanSourceParityConfig()
    grid = source_parity_grid(cfg)
    vector = VectorArmConfig(
        wavelength_m=cfg.wavelength_m,
        waist_m=cfg.beam_radius_m,
        n_pairs=cfg.n_pairs,
        sector_duty=float(cfg.sector_theta_rad) / (TWOPI / float(cfg.n_pairs)),
        sector_rotation_rad=cfg.sector_rotation_rad,
        ideal_components=True,
    )
    base_twin = default_config("fast")
    source_twin = replace(
        base_twin,
        laser=replace(base_twin.laser, wavelength_m=float(cfg.wavelength_m), beam_radius_on_slm_m=float(cfg.beam_radius_m)),
    )
    canonical_cfg = NathanHexagonConfig.from_existing_digital_twin_baseline(source_twin, grid_n=cfg.grid_n, vector=vector)
    current = canonical_target_field(canonical_cfg, grid=grid)
    current_axicon, ax_meta = _apply_free_space_vector_axicon(
        current,
        n_axicon=cfg.axicon_n,
        n_medium=cfg.medium_n,
        base_angle_rad=cfg.axicon_base_angle_rad,
    )
    z_values = _v0_z_values(cfg)
    parity = source_parity_comparison(cfg)
    literal_result = run_literal_source_port(_literal_config_from_source_config(cfg))
    literal_stack = np.asarray(literal_result["intensity_stack"], dtype=np.float32)
    current_stack, _ = _free_space_intensity_stack(current_axicon, z_values)
    stack_rms = _normalised_stack_rms(current_stack, literal_stack)
    reference_index = _nearest_z_index(z_values, cfg.z_reference_m)
    parameter_parity = v0_source_parameter_parity(cfg)
    resolution_status = v0_numerical_resolution_status(cfg)
    ref_diagnostics = _v0_plane_diagnostics(literal_stack[reference_index], grid)
    suggested_verdict = _automated_v0_visual_verdict(ref_diagnostics)
    verdict = suggested_verdict
    if parity["status"] != "source_parity_exact":
        verdict = "UNRESOLVED"
    return VisualLadderStageResult(
        stage_id="V0",
        title="V0A literal versus V0B project propagated-output parity control",
        z_values_m=z_values,
        reference_index=reference_index,
        grid=grid,
        intensity_stack=literal_stack,
        comparison_stack=current_stack,
        metadata={
            "status": VISUAL_LADDER_STATUS,
            "source_parameters": asdict(cfg),
            "axicon": ax_meta,
            "v0a_literal_source_port": {
                "module": "vbb_study.digital_twin.nathan_literal_source_port",
                "role": "literal Nathan-source port",
            },
            "v0b_project_path": {
                "module": "vbb_study.digital_twin.nathan_vector_hexagon",
                "role": "current project vector implementation with identical source parameters",
            },
            "source_parity": parity,
            "source_parameter_parity": parameter_parity,
            "numerical_resolution_status": resolution_status,
            "reference_plane_diagnostics": ref_diagnostics,
            "automated_diagnostic_status": {
                "status": "automated_suggested_verdict_not_operator_visual_assessment",
                "allowed_visual_verdicts": V0_ALLOWED_VISUAL_VERDICTS,
            },
            "suggested_verdict": verdict,
            "operator_visual_assessment": None,
            "propagated_output_visual_verdict": verdict,
            "propagated_output_verdict_is_separate_from_input_parity": True,
            "v0b_project_minus_v0a_literal_stack_rms": stack_rms,
            "current_minus_literal_stack_rms": stack_rms,
            "reference_z_m": float(z_values[reference_index]),
            "plane_selection": "declared nearest z_reference_m, not metric-led",
        },
        verdict=verdict,
    )


def run_v1_inherited_preobjective_visual_gate(
    config: NathanHexagonConfig | None = None,
    *,
    twin_config: TwinConfig | None = None,
    z_span_m: float | None = None,
    z_planes: int | None = None,
    reference_z_m: float = 0.0,
) -> VisualLadderStageResult:
    """Run V1: inherited laser/axicon, free-space vector ASM only before focus."""

    cfg = config or NathanHexagonConfig.fast()
    twin = twin_config or cfg.twin
    grid = default_nathan_grid(cfg)
    field = canonical_target_field(cfg, grid=grid)
    after_axicon, params, ax_meta = apply_vector_axicon(field, twin)
    if z_span_m is None:
        design = compute_design_from_targets(twin.laser, twin.target, twin.material)
        z_span_m = max(float(design.target_bessel_length_m), 10.0e-3)
    planes = int(z_planes or max(7, cfg.z_planes))
    z_values = np.linspace(0.0, float(z_span_m), planes)
    stack, _ = _free_space_intensity_stack(after_axicon, z_values)
    reference_index = _nearest_z_index(z_values, reference_z_m)
    return VisualLadderStageResult(
        stage_id="V1",
        title="V1 inherited laser/axicon pre-objective free-space gate",
        z_values_m=z_values,
        reference_index=reference_index,
        grid=field.grid,
        intensity_stack=stack,
        metadata={
            "status": VISUAL_LADDER_STATUS,
            "geometry": "inherited laser/vector axicon; no ObjectiveMap, scalar focus bridge, or sample mapping",
            "axicon": {**ax_meta, "k_r_surface_m_inv": float(params.k_r_surface_m_inv)},
            "reference_z_m": float(z_values[reference_index]),
            "plane_selection": "declared reference_z_m, not metric-led",
            "visual_classes": ("survives visually", "weak / parameter-sensitive", "absent"),
        },
    )


def run_v2_full_sample_visual_gate(
    config: NathanHexagonConfig | None = None,
    *,
    twin_config: TwinConfig | None = None,
    z_planes: int = 7,
    controls: Sequence[str] = ("nathan_six_sector", "all_radial", "all_azimuthal", "scalar_bessel_gaussian_baseline"),
) -> VisualLadderStageResult:
    """Run V2 fixed-parameter sample-geometry visual gate with F0/F2 diagnostics."""

    cfg = config or NathanHexagonConfig.fast(z_planes=z_planes)
    cfg = replace(cfg, z_planes=int(z_planes))
    gate = build_downstream_focus_validation_gate(
        cfg,
        twin_config=twin_config,
        control_ids=controls,
        f2_solver="fft",
    )
    results = {(result.route_id, result.control_id): result for result in gate["route_results"]}
    f0 = results[("F0_current_scalar_focus_bridge", "nathan_six_sector")]
    f2 = results[("F2_vectorial_pupil_spectrum_reference", "nathan_six_sector")]
    return VisualLadderStageResult(
        stage_id="V2",
        title="V2 inherited full sample-geometry visual gate",
        z_values_m=f0.z_values_m,
        reference_index=0,
        grid={"N": f0.metadata["output_grid_N"], "dx": f0.metadata["output_grid_dx_m"], **make_xy_grid(int(f0.metadata["output_grid_N"]), float(f0.metadata["output_grid_dx_m"]))},
        intensity_stack=f0.intensity_stack,
        comparison_stack=f2.intensity_stack,
        metadata={
            "status": VISUAL_LADDER_STATUS,
            "geometry": "inherited vector axicon -> current F0 bridge with F2 pupil-spectrum diagnostic",
            "focus_gate_status": gate["status"],
            "controls": tuple(controls),
            "reference_z_m": float(f0.z_values_m[0]),
            "plane_selection": "declared z=0 and adjacent planes, not metric-led",
            "visual_classes": ("survives the objective", "becomes circular/faceted", "becomes a six-lobe lattice", "disappears entirely"),
            "focus_comparisons": gate["comparisons"],
        },
    )


def _normalise_image(image: np.ndarray, *, local: bool, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    scale = float(np.nanmax(arr)) if local or vmax is None else float(vmax)
    return arr / max(scale, EPS)


def _central_crop_2d(image: np.ndarray, crop_fraction: float = 0.4) -> np.ndarray:
    arr = np.asarray(image)
    frac = min(max(float(crop_fraction), EPS), 1.0)
    ny, nx = arr.shape[-2], arr.shape[-1]
    cy, cx = max(1, int(round(frac * ny))), max(1, int(round(frac * nx)))
    y0, x0 = (ny - cy) // 2, (nx - cx) // 2
    return arr[y0 : y0 + cy, x0 : x0 + cx]


def visual_ladder_stage_arrays(stage: VisualLadderStageResult) -> dict[str, Any]:
    """Return fixed-plane xy/crop/xz/profile arrays for visual review."""

    stack = np.asarray(stage.intensity_stack, dtype=float)
    ref = int(stage.reference_index)
    grid = stage.grid
    x_m = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float)
    z_m = np.asarray(stage.z_values_m, dtype=float)
    mid = int(stack.shape[-2] // 2)
    xy = stack[ref]
    return {
        "stage_id": stage.stage_id,
        "reference_index": ref,
        "reference_z_m": float(z_m[ref]),
        "xy_full": xy,
        "xy_crop": _central_crop_2d(xy),
        "xz": stack[:, mid, :],
        "x_profile": xy[mid, :],
        "on_axis_z": stack[:, mid, int(stack.shape[-1] // 2)],
        "x_um": x_m / 1e-6,
        "z_um": z_m / 1e-6,
    }


def plot_visual_ladder_stage(
    stage: VisualLadderStageResult,
    *,
    output_path: str | Path | None = None,
    crop_fraction: float = 0.4,
) -> tuple[Any, Any]:
    """Plot one visual ladder stage with fixed planes and no best-metric selection."""

    import matplotlib.pyplot as plt

    stacks = [np.asarray(stage.intensity_stack, dtype=float)]
    if stage.stage_id == "V0" and "v0a_literal_source_port" in stage.metadata:
        labels = ["V0A literal source port"]
    else:
        labels = [stage.stage_id]
    if stage.comparison_stack is not None:
        stacks.append(np.asarray(stage.comparison_stack, dtype=float))
        if stage.stage_id == "V0" and "v0b_project_path" in stage.metadata:
            labels.append("V0B project implementation")
        else:
            labels.append(f"{stage.stage_id} comparison")
    vmax = max(float(np.nanmax(stack)) for stack in stacks)
    rows = len(stacks)
    fig, axes = plt.subplots(rows, 4, figsize=(14.0, max(3.2, 3.0 * rows)), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(rows, 4)
    grid = stage.grid
    x_um = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float) / 1e-6
    z_um = np.asarray(stage.z_values_m, dtype=float) / 1e-6
    ref = int(stage.reference_index)
    mid = int(stacks[0].shape[-2] // 2)
    for row, (stack, label) in enumerate(zip(stacks, labels, strict=True)):
        xy = stack[ref]
        frac = min(max(float(crop_fraction), EPS), 1.0)
        cy = max(1, int(round(frac * xy.shape[-2])))
        cx = max(1, int(round(frac * xy.shape[-1])))
        y0 = (xy.shape[-2] - cy) // 2
        x0 = (xy.shape[-1] - cx) // 2
        crop = xy[y0 : y0 + cy, x0 : x0 + cx]
        x_crop_um = x_um[x0 : x0 + cx]
        xz = stack[:, mid, :]
        ext_xy = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
        ext_crop = [float(x_crop_um[0]), float(x_crop_um[-1]), float(x_crop_um[0]), float(x_crop_um[-1])]
        ext_xz = [float(x_um[0]), float(x_um[-1]), float(z_um[0]), float(z_um[-1])]
        axes_arr[row, 0].imshow(_normalise_image(xy, local=False, vmax=vmax), origin="lower", extent=ext_xy, cmap="inferno", vmin=0.0, vmax=1.0)
        axes_arr[row, 0].set_title(f"{label} full xy\ncommon norm")
        axes_arr[row, 1].imshow(_normalise_image(crop, local=True), origin="lower", extent=ext_crop, cmap="inferno", vmin=0.0, vmax=1.0)
        axes_arr[row, 1].set_title("central crop\nlocal norm")
        axes_arr[row, 2].imshow(_normalise_image(xz, local=False, vmax=vmax), origin="lower", aspect="auto", extent=ext_xz, cmap="inferno", vmin=0.0, vmax=1.0)
        axes_arr[row, 2].set_title("x-z path\ncommon norm")
        axes_arr[row, 3].plot(x_um, xy[mid, :] / max(float(np.max(xy[mid, :])), EPS), label="x profile")
        axes_arr[row, 3].plot(z_um, stack[:, mid, int(stack.shape[-1] // 2)] / max(float(np.max(stack[:, mid, int(stack.shape[-1] // 2)])), EPS), label="on-axis z")
        axes_arr[row, 3].set_title("profiles")
        axes_arr[row, 3].legend(fontsize=7)
        for ax in axes_arr[row, :3]:
            ax.set_xlabel("um")
            ax.set_ylabel("um")
    fig.suptitle(f"{stage.title}; verdict={stage.verdict}; fixed z={stage.z_values_m[ref] / 1e-3:.3f} mm", fontsize=11)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return fig, axes


def _central_crop_2d_with_extent(
    image: np.ndarray,
    axis_m: np.ndarray,
    crop_fraction: float,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    arr = np.asarray(image)
    x = np.asarray(axis_m, dtype=float)
    frac = min(max(float(crop_fraction), EPS), 1.0)
    ny, nx = arr.shape[-2], arr.shape[-1]
    cy, cx = max(1, int(round(frac * ny))), max(1, int(round(frac * nx)))
    y0, x0 = (ny - cy) // 2, (nx - cx) // 2
    crop = arr[y0 : y0 + cy, x0 : x0 + cx]
    x_crop = x[x0 : x0 + cx]
    extent = [float(x_crop[0] / 1e-3), float(x_crop[-1] / 1e-3), float(x_crop[0] / 1e-3), float(x_crop[-1] / 1e-3)]
    return crop, extent, x_crop


def plot_v0_direct_visual_comparison(
    stage: VisualLadderStageResult,
    *,
    reference_image_path: str | Path | None = DEFAULT_NATHAN_FIGURE4_REFERENCE,
    output_path: str | Path | None = None,
    crop_fraction: float = 0.10,
) -> tuple[Any, Any]:
    """Save a direct Nathan-reference versus current V0 propagated-output figure."""

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    stack = np.asarray(stage.intensity_stack, dtype=float)
    grid = stage.grid
    x_m = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float)
    z_m = np.asarray(stage.z_values_m, dtype=float)
    ref = int(stage.reference_index)
    xy = stack[ref]
    xy_crop, xy_extent, x_crop = _central_crop_2d_with_extent(xy, x_m, crop_fraction)
    x0 = int((xy.shape[-1] - x_crop.size) // 2)
    xz_crop = stack[:, xy.shape[-2] // 2, x0 : x0 + x_crop.size]
    mid = xy.shape[-2] // 2
    vmax = float(np.nanmax(xy_crop))

    fig = plt.figure(figsize=(14.0, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=(1.15, 1.0, 1.0))
    ax_ref = fig.add_subplot(gs[:, 0])
    ref_path = None if reference_image_path is None else Path(reference_image_path)
    if ref_path is not None and ref_path.is_file():
        ax_ref.imshow(mpimg.imread(ref_path))
        ax_ref.set_title("Reference: Laser_Manufacturing.pdf page 7, Figure 4 crop")
    else:
        ax_ref.text(0.5, 0.5, "Correct Figure 4\nreference image not found", ha="center", va="center")
        ax_ref.set_title("Reference: missing")
    ax_ref.set_axis_off()

    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.imshow(_normalise_image(xy_crop, local=False, vmax=vmax), origin="lower", extent=xy_extent, cmap="inferno", vmin=0.0, vmax=1.0)
    ax_xy.set_title(f"Current V0 xy at z = {z_m[ref] / 1e-3:.3f} mm")
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")

    ax_xz = fig.add_subplot(gs[0, 2])
    ax_xz.imshow(
        _normalise_image(xz_crop, local=True),
        origin="lower",
        aspect="auto",
        extent=[float(x_crop[0] / 1e-3), float(x_crop[-1] / 1e-3), float(z_m[0] / 1e-3), float(z_m[-1] / 1e-3)],
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    ax_xz.axhline(float(z_m[ref] / 1e-3), color="white", linewidth=0.8, alpha=0.75)
    ax_xz.set_title("Current V0 xz propagation map")
    ax_xz.set_xlabel("x (mm)")
    ax_xz.set_ylabel("z (mm)")

    ax_profile = fig.add_subplot(gs[1, 1])
    line = xy[mid, :]
    ax_profile.plot(x_m / 1e-3, line / max(float(np.nanmax(line)), EPS), color="black", linewidth=1.2)
    ax_profile.set_xlim(float(x_crop[0] / 1e-3), float(x_crop[-1] / 1e-3))
    ax_profile.set_ylim(0.0, 1.05)
    ax_profile.set_title("Current V0 x-axis profile at z = 60 mm")
    ax_profile.set_xlabel("x (mm)")
    ax_profile.set_ylabel("normalised intensity")

    ax_axis = fig.add_subplot(gs[1, 2])
    on_axis = stack[:, mid, xy.shape[-1] // 2]
    ax_axis.plot(z_m / 1e-3, on_axis / max(float(np.nanmax(on_axis)), EPS), color="black", linewidth=1.2)
    ax_axis.axvline(float(z_m[ref] / 1e-3), color="0.35", linewidth=0.8)
    ax_axis.set_title("Current V0 on-axis intensity versus z")
    ax_axis.set_xlabel("z (mm)")
    ax_axis.set_ylabel("normalised intensity")
    ax_axis.set_ylim(0.0, 1.05)
    fig.suptitle(
        f"V0 propagated-output parity: suggested={stage.metadata.get('suggested_verdict', stage.verdict)}; input parity={stage.metadata.get('source_parity', {}).get('status')}",
        fontsize=11,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return fig, (ax_ref, ax_xy, ax_xz, ax_profile, ax_axis)


def plot_v0_field_views(
    stage: VisualLadderStageResult,
    *,
    output_path: str | Path | None = None,
    crop_fraction: float = 0.10,
) -> tuple[Any, Any]:
    """Save full-field and Nathan-style central-crop V0 views."""

    import matplotlib.pyplot as plt

    stack = np.asarray(stage.intensity_stack, dtype=float)
    grid = stage.grid
    x_m = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float)
    ref = int(stage.reference_index)
    xy = stack[ref]
    crop, crop_extent, _ = _central_crop_2d_with_extent(xy, x_m, crop_fraction)
    full_extent = [float(x_m[0] / 1e-3), float(x_m[-1] / 1e-3), float(x_m[0] / 1e-3), float(x_m[-1] / 1e-3)]
    vmax = float(np.nanmax(xy))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), constrained_layout=True)
    panels = (
        (_normalise_image(xy, local=False, vmax=vmax), full_extent, "full transverse field\ncommon normalisation"),
        (_normalise_image(xy, local=True), full_extent, "full transverse field\nlocal normalisation"),
        (_normalise_image(crop, local=False, vmax=vmax), crop_extent, "central 10% zoom\ncommon normalisation"),
        (_normalise_image(crop, local=True), crop_extent, "central 10% zoom\nlocal normalisation"),
    )
    for ax, (image, extent, title) in zip(np.ravel(axes), panels, strict=True):
        ax.imshow(image, origin="lower", extent=extent, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.suptitle(f"V0 field views at z = {stage.z_values_m[ref] / 1e-3:.3f} mm", fontsize=11)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return fig, axes


def plot_v0_reference_literal_project_comparison(
    stage: VisualLadderStageResult,
    *,
    reference_image_path: str | Path | None = DEFAULT_NATHAN_FIGURE4_REFERENCE,
    output_path: str | Path | None = None,
    crop_fraction: float = 0.10,
) -> tuple[Any, Any]:
    """Save the required reference / V0A / V0B / diagnostics comparison."""

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    v0a = np.asarray(stage.intensity_stack, dtype=float)
    if stage.comparison_stack is None:
        raise ValueError("V0 project comparison stack is required for the four-row V0 figure.")
    v0b = np.asarray(stage.comparison_stack, dtype=float)
    grid = stage.grid
    x_m = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float)
    z_m = np.asarray(stage.z_values_m, dtype=float)
    ref = int(stage.reference_index)
    mid = int(v0a.shape[-2] // 2)
    x_mm = x_m / 1e-3
    z_mm = z_m / 1e-3
    frac = min(max(float(crop_fraction), EPS), 1.0)
    cx = max(1, int(round(frac * x_m.size)))
    x0 = (x_m.size - cx) // 2
    x_crop_m = x_m[x0 : x0 + cx]
    xy_extent = [float(x_crop_m[0] / 1e-3), float(x_crop_m[-1] / 1e-3), float(x_crop_m[0] / 1e-3), float(x_crop_m[-1] / 1e-3)]
    xz_extent = [float(x_crop_m[0] / 1e-3), float(x_crop_m[-1] / 1e-3), float(z_mm[0]), float(z_mm[-1])]
    vmax = max(float(np.nanmax(v0a[ref])), float(np.nanmax(v0b[ref])))

    def _draw_path(row_axes: Sequence[Any], stack: np.ndarray, label: str) -> None:
        xy = stack[ref]
        xy_crop = xy[x0 : x0 + cx, x0 : x0 + cx]
        row_axes[0].imshow(_normalise_image(xy_crop, local=False, vmax=vmax), origin="lower", extent=xy_extent, cmap="inferno", vmin=0.0, vmax=1.0)
        row_axes[0].set_title(f"{label}: xy at z = {z_m[ref] / 1e-3:.3f} mm")
        row_axes[0].set_xlabel("x (mm)")
        row_axes[0].set_ylabel("y (mm)")
        line = xy[mid, :]
        row_axes[1].plot(x_mm, line / max(float(np.nanmax(line)), EPS), color="black", linewidth=1.0)
        row_axes[1].set_xlim(float(x_crop_m[0] / 1e-3), float(x_crop_m[-1] / 1e-3))
        row_axes[1].set_title(f"{label}: x profile")
        row_axes[1].set_xlabel("x (mm)")
        row_axes[1].set_ylabel("norm. I")
        row_axes[1].set_ylim(0.0, 1.05)
        xz = stack[:, mid, x0 : x0 + cx]
        row_axes[2].imshow(_normalise_image(xz, local=True), origin="lower", aspect="auto", extent=xz_extent, cmap="inferno", vmin=0.0, vmax=1.0)
        row_axes[2].axhline(float(z_m[ref] / 1e-3), color="white", linewidth=0.8, alpha=0.75)
        row_axes[2].set_title(f"{label}: xz propagation")
        row_axes[2].set_xlabel("x (mm)")
        row_axes[2].set_ylabel("z (mm)")
        on_axis = stack[:, mid, stack.shape[-1] // 2]
        row_axes[3].plot(z_mm, on_axis / max(float(np.nanmax(on_axis)), EPS), color="black", linewidth=1.0)
        row_axes[3].axvline(float(z_m[ref] / 1e-3), color="0.5", linewidth=0.8)
        row_axes[3].set_title(f"{label}: on-axis z")
        row_axes[3].set_xlabel("z (mm)")
        row_axes[3].set_ylabel("norm. I")
        row_axes[3].set_ylim(0.0, 1.05)

    fig = plt.figure(figsize=(17.0, 15.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, height_ratios=(1.0, 1.0, 1.0, 0.95))
    axes: list[Any] = []
    ax_ref = fig.add_subplot(gs[0, :])
    ref_path = None if reference_image_path is None else Path(reference_image_path)
    if ref_path is not None and ref_path.is_file():
        ax_ref.imshow(mpimg.imread(ref_path))
        ax_ref.set_title("Reference: Laser_Manufacturing.pdf page 7, Figure 4 crop")
    else:
        ax_ref.text(0.5, 0.5, "Correct Nathan Figure 4 reference crop not found", ha="center", va="center")
        ax_ref.set_title("Reference: missing")
    ax_ref.set_axis_off()
    axes.append(ax_ref)

    row2 = [fig.add_subplot(gs[1, col]) for col in range(4)]
    row3 = [fig.add_subplot(gs[2, col]) for col in range(4)]
    _draw_path(row2, v0a, "V0A literal source port")
    _draw_path(row3, v0b, "V0B project implementation")
    axes.extend(row2)
    axes.extend(row3)

    diff_xy_full = _equal_power_stack(v0b[ref][None, ...])[0] - _equal_power_stack(v0a[ref][None, ...])[0]
    diff_xy = diff_xy_full[x0 : x0 + cx, x0 : x0 + cx]
    diff_crop = _central_crop_2d(diff_xy_full, crop_fraction)
    diff_xz_full = _equal_power_stack(v0b[:, mid, :]) - _equal_power_stack(v0a[:, mid, :])
    diff_xz = diff_xz_full[:, x0 : x0 + cx]
    row4 = [fig.add_subplot(gs[3, col]) for col in range(4)]
    diff_v = max(float(np.nanmax(np.abs(diff_xy))), EPS)
    row4[0].imshow(diff_xy, origin="lower", extent=xy_extent, cmap="coolwarm", vmin=-diff_v, vmax=diff_v)
    row4[0].set_title("V0B - V0A xy equal-power diff")
    row4[0].set_xlabel("x (mm)")
    row4[0].set_ylabel("y (mm)")
    row4[1].imshow(diff_crop, origin="lower", cmap="coolwarm", vmin=-diff_v, vmax=diff_v)
    row4[1].set_title("central-crop difference")
    row4[1].set_axis_off()
    xz_v = max(float(np.nanmax(np.abs(diff_xz))), EPS)
    row4[2].imshow(diff_xz, origin="lower", aspect="auto", extent=xz_extent, cmap="coolwarm", vmin=-xz_v, vmax=xz_v)
    row4[2].set_title("xz equal-power difference")
    row4[2].set_xlabel("x (mm)")
    row4[2].set_ylabel("z (mm)")
    metrics = _equal_power_shape_metrics(v0b[ref][None, ...], v0a[ref][None, ...])
    crop_metrics = _equal_power_shape_metrics(v0b[ref][None, ...], v0a[ref][None, ...], crop_fraction=crop_fraction)
    row4[3].plot(x_mm, _normalise_image(v0a[ref, mid, :], local=True), label="V0A")
    row4[3].plot(x_mm, _normalise_image(v0b[ref, mid, :], local=True), label="V0B", linestyle="--")
    row4[3].set_title(
        "diagnostics\n"
        f"xy RMS={metrics['equal_power_shape_rms']:.3g}, corr={metrics['equal_power_intensity_correlation']:.4f}\n"
        f"crop RMS={crop_metrics['equal_power_shape_rms']:.3g}"
    )
    row4[3].set_xlabel("x (mm)")
    row4[3].legend(fontsize=7)
    axes.extend(row4)
    fig.suptitle(
        f"V0 propagated-output parity: {stage.metadata.get('propagated_output_visual_verdict', stage.verdict)}; "
        f"input_array_parity={stage.metadata.get('source_parity', {}).get('status')}",
        fontsize=12,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180)
    return fig, axes


def _resample_image_to_grid(
    image: np.ndarray,
    source_grid: Mapping[str, Any],
    target_grid: Mapping[str, Any],
) -> np.ndarray:
    src_x = np.asarray(source_grid["x"], dtype=float)
    src_y = np.asarray(source_grid.get("y", src_x), dtype=float)
    target_x = np.asarray(target_grid["X"], dtype=float)
    target_y = np.asarray(target_grid["Y"], dtype=float)
    dx = float(np.median(np.diff(src_x))) if src_x.size > 1 else float(source_grid["dx"])
    dy = float(np.median(np.diff(src_y))) if src_y.size > 1 else float(source_grid.get("dy", dx))
    col = (target_x - float(src_x[0])) / max(dx, EPS)
    row = (target_y - float(src_y[0])) / max(dy, EPS)
    return map_coordinates(np.asarray(image, dtype=float), [row, col], order=1, mode="nearest")


def _normalised_image_rms(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalise_image(np.asarray(a, dtype=float), local=True)
    bb = _normalise_image(np.asarray(b, dtype=float), local=True)
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _run_v0_reference_plane_for_convergence(config: NathanSourceParityConfig) -> dict[str, Any]:
    """Return only the V0 z-reference plane for numerical convergence checks."""

    cfg = config
    literal_cfg = _literal_config_from_source_config(cfg)
    literal_grid = literal_make_source_grid(literal_cfg.grid_n, literal_cfg.window_m)
    literal_source = literal_make_segmented_ra_input(literal_grid, literal_cfg)
    literal_axicon = apply_source_axicon(
        literal_source["ex"],
        literal_source["ey"],
        literal_source["ez"],
        literal_grid,
        literal_cfg,
    )
    v0a_stack = propagate_source_vector_asm(
        literal_axicon["ex"],
        literal_axicon["ey"],
        literal_axicon["ez"],
        literal_grid,
        literal_cfg,
        [float(cfg.z_reference_m)],
    )
    v0a_xy = np.asarray(v0a_stack[0], dtype=np.float32)

    grid = source_parity_grid(cfg)
    vector = VectorArmConfig(
        wavelength_m=cfg.wavelength_m,
        waist_m=cfg.beam_radius_m,
        n_pairs=cfg.n_pairs,
        sector_duty=float(cfg.sector_theta_rad) / (TWOPI / float(cfg.n_pairs)),
        sector_rotation_rad=cfg.sector_rotation_rad,
        ideal_components=True,
    )
    base_twin = default_config("fast")
    source_twin = replace(
        base_twin,
        laser=replace(base_twin.laser, wavelength_m=float(cfg.wavelength_m), beam_radius_on_slm_m=float(cfg.beam_radius_m)),
    )
    canonical_cfg = NathanHexagonConfig.from_existing_digital_twin_baseline(source_twin, grid_n=cfg.grid_n, vector=vector)
    current = canonical_target_field(canonical_cfg, grid=grid)
    current_axicon, _ = _apply_free_space_vector_axicon(
        current,
        n_axicon=cfg.axicon_n,
        n_medium=cfg.medium_n,
        base_angle_rad=cfg.axicon_base_angle_rad,
    )
    stack, _ = _free_space_intensity_stack(current_axicon, [float(cfg.z_reference_m)])
    v0b_xy = np.asarray(stack[0], dtype=np.float32)
    diagnostics = _v0_plane_diagnostics(v0a_xy, literal_grid)
    return {
        "grid": literal_grid,
        "xy": v0a_xy,
        "v0a_xy": v0a_xy,
        "v0b_xy": v0b_xy,
        "z_reference_m": float(cfg.z_reference_m),
        "diagnostics": diagnostics,
        "suggested_verdict": _automated_v0_visual_verdict(diagnostics),
        "numerical_resolution_status": v0_numerical_resolution_status(cfg),
        "v0b_minus_v0a_full_field_rms": _normalised_image_rms(v0b_xy, v0a_xy),
        "v0b_minus_v0a_central_10pct_rms": _normalised_image_rms(_central_crop_2d(v0b_xy, 0.10), _central_crop_2d(v0a_xy, 0.10)),
    }


def run_v0_numerical_convergence(
    config: NathanSourceParityConfig | None = None,
    *,
    grid_ns: Sequence[int] = (512, 1024),
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run V0 at multiple grids and compare the z=60 mm output plane."""

    base = config or NathanSourceParityConfig()
    records: dict[int, dict[str, Any]] = {}
    for n in sorted({int(value) for value in grid_ns if int(value) > 0}):
        cfg = replace(base, grid_n=n, window_m=10.0e-3, z_span_m=None)
        records[n] = _run_v0_reference_plane_for_convergence(cfg)

    if not records:
        return {"status": "no_convergence_grids_requested", "rows": [], "reference_grid_n": None}
    reference_n = max(records)
    reference = records[reference_n]
    rows: list[dict[str, Any]] = []
    for n, record in sorted(records.items()):
        ref_on_grid = _resample_image_to_grid(reference["xy"], reference["grid"], record["grid"])
        full_rms = _normalised_image_rms(record["xy"], ref_on_grid)
        crop_rms = _normalised_image_rms(_central_crop_2d(record["xy"], 0.10), _central_crop_2d(ref_on_grid, 0.10))
        diag = dict(record["diagnostics"])
        ref_diag = dict(reference["diagnostics"])
        ring_rel = abs(float(diag["ring_radius_m"]) - float(ref_diag["ring_radius_m"])) / max(float(ref_diag["ring_radius_m"]), EPS)
        sixfold_delta = abs(float(diag["sixfold_order6_over_non_dc"]) - float(ref_diag["sixfold_order6_over_non_dc"]))
        core_delta = abs(float(diag["central_core_darkness"]) - float(ref_diag["central_core_darkness"]))
        rows.append(
            {
                "grid_n": int(n),
                "reference_grid_n": int(reference_n),
                "full_field_normalised_intensity_rms": float(full_rms),
                "central_10pct_normalised_intensity_rms": float(crop_rms),
                "sixfold_order6_over_non_dc": float(diag["sixfold_order6_over_non_dc"]),
                "sixfold_delta_to_reference": float(sixfold_delta),
                "ring_radius_m": float(diag["ring_radius_m"]),
                "ring_radius_relative_delta_to_reference": float(ring_rel),
                "central_core_darkness": float(diag["central_core_darkness"]),
                "central_core_darkness_delta_to_reference": float(core_delta),
                "v0b_project_minus_v0a_literal_full_field_rms": float(record.get("v0b_minus_v0a_full_field_rms", np.nan)),
                "v0b_project_minus_v0a_literal_central_10pct_rms": float(record.get("v0b_minus_v0a_central_10pct_rms", np.nan)),
                "suggested_verdict": record["suggested_verdict"],
            }
        )

    materially_consistent = True
    comparison_rows = [row for row in rows if int(row["grid_n"]) != int(reference_n) and int(row["grid_n"]) >= 512]
    if comparison_rows:
        materially_consistent = all(
            row["full_field_normalised_intensity_rms"] <= 0.08
            and row["central_10pct_normalised_intensity_rms"] <= 0.18
            and row["ring_radius_relative_delta_to_reference"] <= 0.12
            for row in comparison_rows
        )
    status = "v0_convergence_materially_consistent" if materially_consistent else "v0_unresolved_n512_n1024_disagree_materially"

    paths: dict[str, Path] = {}
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(records), figsize=(4.2 * len(records), 4.0), constrained_layout=True)
        axes_arr = np.asarray(axes).reshape(-1)
        vmax = max(float(np.nanmax(record["xy"])) for record in records.values())
        for ax, (n, record) in zip(axes_arr, sorted(records.items()), strict=True):
            crop = _central_crop_2d(record["xy"], 0.10)
            ax.imshow(_normalise_image(crop, local=False, vmax=vmax), origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(f"N={n}; {record['suggested_verdict']}")
            ax.set_axis_off()
        path = out / "nathan_visual_ladder_v0_convergence_xy.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["xy_visual_comparison"] = path

    result = {
        "status": status,
        "reference_grid_n": int(reference_n),
        "rows": rows,
        "output_paths": paths,
        "rule": "If any reported convergence grid disagrees materially with the highest-N reference, V0 is unresolved rather than a mechanism failure.",
    }
    if output_dir is not None:
        report_path = Path(output_dir) / "nathan_visual_ladder_v0_convergence_report.json"
        report_path.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        paths["convergence_report_json"] = report_path
        result["output_paths"] = paths
    return result


def build_visual_reproduction_ladder_report(
    *,
    source_config: NathanSourceParityConfig | None = None,
    config: NathanHexagonConfig | None = None,
    output_dir: str | Path | None = None,
    reference_image_path: str | Path | None = DEFAULT_NATHAN_FIGURE4_REFERENCE,
    run_convergence: bool = False,
    convergence_grid_ns: Sequence[int] = (512, 1024),
    run_v1: bool = False,
    run_v2: bool = False,
) -> VisualLadderReport:
    """Build the strict visual ladder without turning V0 input parity into V1/V2 claims."""

    v0 = run_v0_source_parity_visual_control(source_config)
    stages: list[VisualLadderStageResult] = [v0]
    convergence_report = None
    paths: dict[str, Path] = {}
    if run_convergence:
        convergence_report = run_v0_numerical_convergence(
            source_config or NathanSourceParityConfig(),
            grid_ns=convergence_grid_ns,
            output_dir=output_dir,
        )
        paths.update(dict(convergence_report.get("output_paths", {})))
    v0_report = v0_source_output_parity_report(v0, convergence_report=convergence_report)
    v0_visual = str(v0_report["propagated_output_visual_verdict"])
    convergence_status = None if convergence_report is None else str(convergence_report.get("status"))
    convergence_documented = convergence_status == "v0_convergence_materially_consistent"
    v2_allowed = bool(v0_visual == "PASS" and convergence_documented)
    if v0_visual == "PARTIAL" and convergence_documented:
        v2_allowed = True

    v1_status = "not_run"
    if run_v1:
        v1 = run_v1_inherited_preobjective_visual_gate(config)
        stages.append(v1)
        v1_status = "current inherited pre-objective result under its present parameterisation"
        if run_v2 and v2_allowed:
            stages.append(run_v2_full_sample_visual_gate(config))
            v1_status = "current inherited pre-objective result generated; V2 diagnostic also generated"
    v2_status = "allowed" if v2_allowed else "blocked_until_v0_pass_or_converged_partial"
    if run_v2 and not v2_allowed:
        v2_status = "requested_but_blocked_until_v0_pass_or_converged_partial"
    stopping_result = (
        f"V0 source-output parity status: {v0_visual}; "
        f"V1 inherited pre-objective visual status: {v1_status}; "
        f"V2 allowed: {v2_status}."
    )
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            path = out / f"nathan_visual_ladder_{stage.stage_id.lower()}.png"
            fig, _ = plot_visual_ladder_stage(stage, output_path=path)
            import matplotlib.pyplot as plt

            plt.close(fig)
            paths[stage.stage_id] = path
            if stage.stage_id == "V0":
                direct_path = out / "nathan_visual_ladder_v0_reference_vs_reproduction.png"
                fig, _ = plot_v0_reference_literal_project_comparison(stage, reference_image_path=reference_image_path, output_path=direct_path)
                plt.close(fig)
                paths["V0_direct_visual_comparison"] = direct_path
                fields_path = out / "nathan_visual_ladder_v0_field_views.png"
                fig, _ = plot_v0_field_views(stage, output_path=fields_path)
                plt.close(fig)
                paths["V0_field_views"] = fields_path
    status_report = {
        **v0_report,
        "v1_inherited_preobjective_visual_status": v1_status,
        "v2_allowed": v2_allowed,
        "v2_status": v2_status,
    }
    if output_dir is not None:
        status_path = Path(output_dir) / "nathan_visual_ladder_v0_status_report.json"
        status_path.write_text(
            json.dumps(
                _json_ready(
                    {
                        "stopping_result": stopping_result,
                        "status_report": status_report,
                        "convergence_report": convergence_report,
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["V0_status_report_json"] = status_path
    return VisualLadderReport(
        source_parity=v0.metadata["source_parity"],
        stages=tuple(stages),
        stopping_result=stopping_result,
        output_paths=paths,
        status_report=status_report,
        convergence_report=convergence_report,
    )


def canonical_target_diagnostics(
    config: NathanHexagonConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return arrays required for target-field visualisation/export."""

    cfg = config or NathanHexagonConfig.fast()
    grid_dict = dict(default_nathan_grid(cfg) if grid is None else grid)
    target = canonical_target_field(cfg, grid=grid_dict)
    plus, minus = target.circular_components()
    stokes = target.stokes()
    return {
        "field": target,
        "sector_mask": nathan_sector_mask(grid_dict, cfg),
        "alpha_rad": nathan_alpha(grid_dict, cfg),
        "Ex_amplitude": np.abs(target.ex),
        "Ex_phase_rad": np.angle(target.ex),
        "Ey_amplitude": np.abs(target.ey),
        "Ey_phase_rad": np.angle(target.ey),
        "intensity": target.intensity,
        "polarization_angle_rad": local_polarization_angle(target),
        "stokes": stokes,
        "circular_plus": plus,
        "circular_minus": minus,
        "radial_control": canonical_target_field(cfg, grid=grid_dict, control="radial"),
        "azimuthal_control": canonical_target_field(cfg, grid=grid_dict, control="azimuthal"),
    }


def inherited_parameter_rows(config: NathanHexagonConfig | None = None) -> list[dict[str, Any]]:
    """Return the existing Digital Twin parameters inherited by this branch."""

    cfg = config or NathanHexagonConfig.fast()
    twin = cfg.twin
    design = compute_design_from_targets(twin.laser, twin.target, twin.material)
    axicon = twin.physical_axicon
    return [
        {"parameter": "laser.name", "value": twin.laser.name, "source": "TwinConfig.laser", "role": "inherited fixed source"},
        {"parameter": "laser.wavelength_m", "value": twin.laser.wavelength_m, "source": "TwinConfig.laser.wavelength_m", "role": "inherited fixed source"},
        {"parameter": "laser.pulse_duration_s", "value": twin.laser.pulse_duration_s, "source": "TwinConfig.laser.pulse_duration_s", "role": "inherited fixed source"},
        {"parameter": "laser.beam_radius_on_slm_m", "value": twin.laser.beam_radius_on_slm_m, "source": "TwinConfig.laser.beam_radius_on_slm_m", "role": "inherited Gaussian envelope"},
        {"parameter": "slm.resolution_x/y", "value": f"{twin.slm.resolution_x} x {twin.slm.resolution_y}", "source": "TwinConfig.slm", "role": "inherited panel aperture"},
        {"parameter": "slm.pixel_pitch_m", "value": twin.slm.pixel_pitch_m, "source": "TwinConfig.slm.pixel_pitch_m", "role": "inherited panel sampling"},
        {"parameter": "slm.phase_bits", "value": twin.slm.phase_bits, "source": "TwinConfig.slm.phase_bits", "role": "inherited quantisation baseline"},
        {"parameter": "slm.fill_factor", "value": twin.slm.fill_factor, "source": "TwinConfig.slm.fill_factor", "role": "inherited panel-realistic baseline"},
        {"parameter": "slm.blaze_period_px", "value": twin.slm.blaze_period_px, "source": "TwinConfig.slm.blaze_period_px", "role": "inherited carrier baseline"},
        {"parameter": "objective.NA", "value": twin.objective.NA, "source": "TwinConfig.objective.NA", "role": "inherited focusing"},
        {"parameter": "objective.f_eff_m", "value": twin.objective.f_eff_m, "source": "TwinConfig.objective.f_eff_m", "role": "inherited focusing"},
        {"parameter": "objective.pupil_fill", "value": twin.objective.pupil_fill, "source": "TwinConfig.objective.pupil_fill", "role": "inherited pupil convention"},
        {"parameter": "relay.effective_relay_f_m", "value": twin.relay.effective_relay_f_m, "source": "TwinConfig.relay.effective_relay_f_m", "role": "inherited ObjectiveMap"},
        {"parameter": "material.refractive_index", "value": twin.material.refractive_index, "source": "TwinConfig.material.refractive_index", "role": "inherited design medium for kr"},
        {"parameter": "target.ell", "value": twin.target.ell, "source": "TwinConfig.target.ell", "role": "inherited scalar baseline/control"},
        {"parameter": "target.target_core_diameter_m", "value": twin.target.target_core_diameter_m, "source": "TwinConfig.target.target_core_diameter_m", "role": "inherited transverse scale target"},
        {"parameter": "target.target_bessel_length_m", "value": twin.target.target_bessel_length_m, "source": "TwinConfig.target.target_bessel_length_m", "role": "inherited axial scale target"},
        {"parameter": "target.n_axicon", "value": twin.target.n_axicon, "source": "TwinConfig.target.n_axicon", "role": "inherited axicon material"},
        {"parameter": "design.kr_sample_m_inv", "value": design.kr_sample_m_inv, "source": "compute_design_from_targets", "role": "inherited micro-scale radial wavevector"},
        {"parameter": "design.gamma_slm_deg", "value": design.gamma_slm_deg, "source": "compute_design_from_targets", "role": "inherited pre-objective axicon angle"},
        {"parameter": "physical_axicon.axicon_base_angle_deg", "value": axicon.axicon_base_angle_deg, "source": "TwinConfig.physical_axicon", "role": "inherited if explicitly configured, otherwise BeamDesign.gamma_slm_rad"},
        {"parameter": "grid.N", "value": twin.grid.N, "source": "TwinConfig.grid.N", "role": "inherited baseline grid; cfg.grid_n may downsample for developer checks"},
        {"parameter": "grid.axial_range_m", "value": twin.grid.axial_range_m, "source": "TwinConfig.grid.axial_range_m", "role": "inherited axial window convention"},
        {"parameter": "grid.axial_points", "value": twin.grid.axial_points, "source": "TwinConfig.grid.axial_points", "role": "inherited baseline z sampling; cfg.z_planes may downsample for developer checks"},
        {"parameter": "propagation.method", "value": twin.propagation.method, "source": "TwinConfig.propagation.method", "role": "inherited propagation convention"},
    ]


def nathan_specific_parameter_rows(config: NathanHexagonConfig | None = None) -> list[dict[str, Any]]:
    """Return the new Nathan-sector parameters layered on top of the twin."""

    cfg = config or NathanHexagonConfig.fast()
    sectors = 2 * int(cfg.vector.n_pairs)
    return [
        {"parameter": "baseline_preset", "value": cfg.baseline_preset, "meaning": "Existing Digital Twin preset used as the root geometry.", "status": "fixed per run"},
        {"parameter": "grid_n", "value": cfg.grid_n, "meaning": "Route-generator computational sampling; developer fast runs may downsample.", "status": "fixed or convergence-swept"},
        {"parameter": "z_planes", "value": cfg.z_planes, "meaning": "Number of micro-scale z planes sampled for this run.", "status": "fixed or convergence-swept"},
        {"parameter": "vector.n_pairs", "value": cfg.vector.n_pairs, "meaning": "Three radial/azimuthal pairs produce six sectors.", "status": "fixed for Nathan six-sector study"},
        {"parameter": "sector_count", "value": sectors, "meaning": "Total alternating radial/azimuthal sectors.", "status": "derived fixed"},
        {"parameter": "vector.sector_duty", "value": cfg.vector.sector_duty, "meaning": "Fraction of each pair assigned to one sector state.", "status": "fixed or swept"},
        {"parameter": "vector.sector_rotation_rad", "value": cfg.vector.sector_rotation_rad, "meaning": "Angular rotation of the sector boundary pattern.", "status": "swept"},
        {"parameter": "PatternedHWPConfig.case", "value": "continuous/six_wedges/mosaic", "meaning": "Patterned-retarder approximation family.", "status": "route selection"},
        {"parameter": "PatternedHWPConfig.tiles_per_sector", "value": "integer", "meaning": "Mosaic angular resolution inside each 60 degree sector.", "status": "swept"},
        {"parameter": "PatternedHWPConfig.seam_width_rad", "value": "radians", "meaning": "Dead or uncertain retarder boundary width.", "status": "assumed or swept"},
        {"parameter": "PatternedHWPConfig.central_defect_radius_m", "value": "metres", "meaning": "Optional central manufacturing defect mask.", "status": "assumed or swept"},
        {"parameter": "serial_slm.case", "value": "ideal/panel_realistic", "meaning": "Dual-SLM vector-generator realism level before common handoff.", "status": "route selection"},
        {"parameter": "serial_slm.naive_psi2", "value": False, "meaning": "Wrong-phase-sign control hook.", "status": "control only"},
        {"parameter": "serial_slm.wrong_carrier_sign", "value": False, "meaning": "Wrong carrier-sign control hook.", "status": "control only"},
    ]


def digital_twin_plane_map_rows(config: NathanHexagonConfig | None = None) -> list[dict[str, Any]]:
    """Return the declared plane map for inserting the vector generator."""

    cfg = config or NathanHexagonConfig.fast()
    return [
        {"plane": "P0", "meaning": "Laser Gaussian input inherited from PHAROS-like baseline.", "coordinates_units": "SLM/pre-objective x,y in metres; reported in micrometres for plots", "field_representation": "scalar Gaussian amplitude metadata", "scalar_vector": "scalar source envelope", "module_function": "TwinConfig.laser; vector_arm_chain.gaussian_envelope", "plane_type": "laser/source plane"},
        {"plane": "P1", "meaning": "Vector-generation input plane using the inherited Gaussian envelope.", "coordinates_units": "same centred handoff grid as default_nathan_grid", "field_representation": "Jones Ex/Ey input, horizontal for HWP or 45 degree for serial SLM", "scalar_vector": "vector/Jones", "module_function": "default_nathan_grid; run_patterned_hwp_route; run_vector_arm", "plane_type": "SLM/vector-generator input image plane"},
        {"plane": "P2", "meaning": "Common vector-generator output handoff from patterned-HWP route or serial-SLM chain.", "coordinates_units": "same handoff grid, metres", "field_representation": "VectorField Ex/Ey/Ez with Stokes/circular diagnostics", "scalar_vector": "vector/Jones", "module_function": "canonical_target_field; run_patterned_hwp_route; run_serial_slm_route", "plane_type": "vector-generator output handoff plane"},
        {"plane": "P3", "meaning": "Existing downstream shaping/filter handoff where enabled by the inherited route architecture.", "coordinates_units": "same handoff grid plus Fourier coordinates for iris diagnostics", "field_representation": "VectorField or circular components after common physical iris when enabled", "scalar_vector": "vector/Jones", "module_function": "vector_fourier.apply_fourier_iris; carrier_collinearity_report", "plane_type": "Fourier/filter handoff only for supported serial-SLM cases"},
        {"plane": "P4", "meaning": "Existing physical axicon and objective/pupil input convention.", "coordinates_units": "pre-objective/axicon grid in metres", "field_representation": "VectorField after thin vector axicon phase and Fresnel s/p split", "scalar_vector": "vector propagated component-wise", "module_function": "vector_axicon.apply_vector_axicon; resolve_vector_axicon_parameters", "plane_type": "axicon/pupil/objective input"},
        {"plane": "P5", "meaning": "Existing focused surface/sample reference plane; z=0 for the air-side stack.", "coordinates_units": "focused x,y in metres; plot labels in micrometres", "field_representation": "VectorField and SurfaceField handoff for F0; vectorial pupil-spectrum fields for F2 validation", "scalar_vector": "F0 uses scalar per-component focus; F2 is a scoped vectorial pupil-spectrum reference", "module_function": "vector_axicon.focus_vector_to_surface; run_vector_axicon_to_surface; build_downstream_focus_validation_gate", "plane_type": "sample/focal reference plane"},
        {"plane": "P6+", "meaning": "Micro-scale propagation region around and beyond the surface reference.", "coordinates_units": f"z in metres from P5; this run samples {cfg.z_planes} planes", "field_representation": "intensity z-stack plus equal-power difference stacks in the focus gate", "scalar_vector": "F0 uses vector ASM; F2 uses vectorial pupil-spectrum propagation at the same z values", "module_function": "air_z_values; vector_field.propagate_vector_asm; vectorial_pupil_spectrum_reference; hexagon_metrics_for_stack", "plane_type": "sample-region axial scan"},
    ]


def source_convention_validation_control(config: NathanHexagonConfig | None = None) -> dict[str, Any]:
    """Validate Nathan's sector convention without treating it as study geometry."""

    cfg = config or NathanHexagonConfig.fast()
    sectors = 2 * int(cfg.vector.n_pairs)
    centres = (np.arange(sectors, dtype=float) + 0.5) * TWOPI / float(sectors) + float(cfg.vector.sector_rotation_rad)
    grid = {
        "PHI": centres.reshape(1, -1),
        "R": np.ones((1, sectors), dtype=float),
        "X": np.cos(centres).reshape(1, -1),
        "Y": np.sin(centres).reshape(1, -1),
        "x": np.arange(sectors, dtype=float),
        "y": np.asarray([0.0]),
        "dx": 1.0,
        "dy": 1.0,
    }
    mask = nathan_sector_mask(grid, cfg).reshape(-1)
    alpha = nathan_alpha(grid, cfg).reshape(-1)
    expected = centres + np.where(mask == 1, 0.5 * np.pi, 0.0)
    radial_ok = bool(np.allclose(alpha[mask == 0], centres[mask == 0]))
    azimuthal_ok = bool(np.allclose(alpha[mask == 1], centres[mask == 1] + 0.5 * np.pi))
    alternating = bool(np.all(mask != np.roll(mask, 1))) if mask.size > 1 else False
    return {
        "control_id": "nathan_source_convention_only",
        "scope": "Source-convention validation only; not the Digital Twin geometry.",
        "sector_count": int(sectors),
        "sector_width_deg": float(360.0 / float(sectors)),
        "sector_mask_sequence": tuple(int(v) for v in mask),
        "state_labels": {0: "radial", 1: "azimuthal"},
        "radial_alpha_equals_theta": radial_ok,
        "azimuthal_alpha_equals_theta_plus_pi_over_2": azimuthal_ok,
        "alternating_sixty_degree_states": alternating and sectors == 6,
        "max_alpha_error_rad": float(np.max(np.abs(np.angle(np.exp(1j * (alpha - expected)))))),
    }


def _sector_tile_centres(
    theta: np.ndarray,
    *,
    sectors: int,
    tiles_per_sector: int,
    rotation_rad: float = 0.0,
) -> np.ndarray:
    count = int(sectors) * max(1, int(tiles_per_sector))
    width = TWOPI / float(count)
    t = np.mod(theta - float(rotation_rad), TWOPI)
    idx = np.floor(t / width).astype(int)
    return (idx + 0.5) * width + float(rotation_rad)


def _nearest_boundary_distance(theta: np.ndarray, *, sectors: int, rotation_rad: float = 0.0) -> np.ndarray:
    width = TWOPI / float(sectors)
    frac = np.mod(theta - float(rotation_rad), width) / width
    return np.minimum(frac, 1.0 - frac) * width


def patterned_hwp_axis_map(
    grid: Mapping[str, Any],
    config: NathanHexagonConfig | None = None,
    hwp: PatternedHWPConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(fast_axis_rad, transmission_mask)` for the patterned-HWP route."""

    cfg = config or NathanHexagonConfig.fast()
    hp = hwp or PatternedHWPConfig()
    theta = np.asarray(grid["PHI"], dtype=float)
    key = hp.case.lower().strip()
    sectors = int(2 * cfg.vector.n_pairs)
    rotation = float(cfg.vector.sector_rotation_rad)
    if key in {"continuous", "ideal_continuous"}:
        alpha = nathan_alpha(grid, cfg)
    elif key in {"six_wedges", "wedges", "six_large_wedges"}:
        theta_c = _sector_tile_centres(theta, sectors=sectors, tiles_per_sector=1, rotation_rad=rotation)
        temp_grid = {**dict(grid), "PHI": theta_c}
        alpha = nathan_alpha(temp_grid, cfg)
    elif key in {"mosaic", "tiled", "fine_tiles"}:
        theta_c = _sector_tile_centres(theta, sectors=sectors, tiles_per_sector=max(1, hp.tiles_per_sector), rotation_rad=rotation)
        temp_grid = {**dict(grid), "PHI": theta_c}
        alpha = nathan_alpha(temp_grid, cfg)
    else:
        raise ValueError(f"Unsupported patterned-HWP case: {hp.case!r}")

    axis = 0.5 * alpha + float(hp.fast_axis_error_rad)
    mask = np.ones_like(axis, dtype=float) * np.sqrt(max(float(hp.transmission), 0.0))
    if float(hp.seam_width_rad) > 0.0:
        mask = np.where(
            _nearest_boundary_distance(theta, sectors=sectors, rotation_rad=rotation) <= 0.5 * float(hp.seam_width_rad),
            0.0,
            mask,
        )
    if float(hp.central_defect_radius_m) > 0.0:
        mask = np.where(np.asarray(grid["R"], dtype=float) <= float(hp.central_defect_radius_m), 0.0, mask)
    if hp.aperture_radius_m is not None:
        mask = np.where(np.asarray(grid["R"], dtype=float) <= float(hp.aperture_radius_m), mask, 0.0)
    return axis, mask


def run_patterned_hwp_route(
    config: NathanHexagonConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
    hwp: PatternedHWPConfig | None = None,
    route_id: str | None = None,
) -> RouteFieldResult:
    """Run one patterned-HWP route up to the common handoff plane."""

    cfg = config or NathanHexagonConfig.fast()
    hp = hwp or PatternedHWPConfig()
    grid_dict = dict(default_nathan_grid(cfg) if grid is None else grid)
    amp = gaussian_envelope(grid_dict, cfg.vector)
    axis, mask = patterned_hwp_axis_map(grid_dict, cfg, hp)
    ex, ey = retarder_jones(
        amp.astype(complex),
        np.zeros_like(amp, dtype=complex),
        np.pi + float(hp.retardance_error_rad),
        axis,
    )
    ex = ex * mask
    ey = ey * mask
    field_out = VectorField(
        ex=ex,
        ey=ey,
        ez=np.zeros_like(ex, dtype=complex),
        grid=grid_dict,
        wavelength_m=cfg.vector.wavelength_m,
        medium_index=1.0,
        metadata={"route": "patterned_hwp", "case": hp.case, "stage": STAGE},
    )
    target = canonical_target_field(cfg, grid=grid_dict)
    comparison = compare_vector_fields(field_out, target)
    p_in = field_power(amp, grid_dict)
    return RouteFieldResult(
        route_id=route_id or f"patterned_hwp_{hp.case}",
        field=field_out,
        target=target,
        comparison=comparison,
        metadata={
            "route_family": "patterned_hwp",
            "hwp": asdict(hp),
            "power_transmission": field_out.power / max(p_in, EPS),
            "model_status": MODEL_STATUS,
            "geometry_source": "existing_digital_twin_baseline",
        },
    )


def run_serial_slm_route(
    config: NathanHexagonConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
    case: str = "ideal",
    apply_order_filter: bool = True,
    wrong_carrier_sign: bool = False,
    naive_psi2: bool = False,
    piston_delta_rad: float | None = None,
    route_id: str | None = None,
) -> RouteFieldResult:
    """Run the serial dual-SLM route up to the common handoff plane."""

    cfg = config or NathanHexagonConfig.fast()
    key = str(case).lower().strip()
    vcfg = cfg.vector
    if piston_delta_rad is not None:
        vcfg = replace(vcfg, piston_delta_rad=float(piston_delta_rad))
    if key in {"ideal", "ideal_serial_slm"}:
        vcfg = replace(vcfg, ideal_components=True, apply_carrier=False, quantise=False, apply_fill_factor=False)
    elif key in {"panel_realistic", "realistic", "panel"}:
        vcfg = replace(vcfg, ideal_components=False, apply_carrier=True, quantise=True, apply_fill_factor=True)
    elif key in {"degraded", "waveplate_registration_errors"}:
        vcfg = replace(
            vcfg,
            ideal_components=False,
            apply_carrier=True,
            quantise=True,
            apply_fill_factor=True,
            hwp_retardance_error_rad=vcfg.hwp_retardance_error_rad + 2.0 * np.pi / 100.0,
            qwp_retardance_error_rad=vcfg.qwp_retardance_error_rad + 2.0 * np.pi / 100.0,
        )
    else:
        raise ValueError(f"Unsupported serial-SLM case: {case!r}")
    if wrong_carrier_sign:
        vcfg = replace(vcfg, slm2=replace(vcfg.slm2, carrier_sign=+1))

    grid_dict = dict(default_vector_arm_grid(vcfg, cfg.grid_n) if grid is None else grid)
    run = run_vector_arm(vcfg, grid=grid_dict, naive_psi2=bool(naive_psi2), return_debug=True)
    field_out = run.field
    extra: dict[str, Any] = {"ledgers": dict(run.ledgers), "order_filter_applied": False}

    if apply_order_filter and not vcfg.ideal_components and vcfg.apply_carrier:
        plus, minus = field_out.circular_components()
        carrier = float(vcfg.effective_slm2.carrier_lp_per_m)
        report = carrier_collinearity_report(plus, minus, grid_dict)
        signal_fx = 0.5 * (float(report.plus_peak.fx_cpm) + float(report.minus_peak.fx_cpm))
        signal_fy = 0.5 * (float(report.plus_peak.fy_cpm) + float(report.minus_peak.fy_cpm))
        extra["carrier_collinearity"] = {
            "separation_pixels": float(report.separation_pixels),
            "plus_fx_cpm": float(report.plus_peak.fx_cpm),
            "minus_fx_cpm": float(report.minus_peak.fx_cpm),
            "configured_signal_fx_cpm": carrier,
            "measured_signal_fx_cpm": signal_fx,
            "measured_signal_fy_cpm": signal_fy,
        }
        iris = apply_fourier_iris(
            (plus, minus),
            grid_dict,
            signal_fx_cpm=signal_fx,
            signal_fy_cpm=signal_fy,
            iris_radius_frac=float(vcfg.iris_radius_frac),
            wavelength_m=vcfg.wavelength_m,
            tilt_tolerance_rad=2e-3,
        )
        plus_f, minus_f = iris.signal
        ex, ey = circular_to_linear(plus_f, minus_f)
        field_out = VectorField(
            ex=ex,
            ey=ey,
            ez=np.zeros_like(ex, dtype=complex),
            grid=grid_dict,
            wavelength_m=vcfg.wavelength_m,
            medium_index=1.0,
            metadata={**dict(field_out.metadata), "post_slm2_order_filter": "applied"},
        )
        extra["order_filter_applied"] = True
        extra["iris_ledger"] = iris.ledger.as_dict()
        extra["residual_tilt_rad"] = tuple(float(v) for v in iris.residual_tilt_rad)

    target = canonical_target_field(replace(cfg, vector=vcfg), grid=grid_dict)
    comparison = compare_vector_fields(field_out, target)
    return RouteFieldResult(
        route_id=route_id or f"serial_slm_{key}",
        field=field_out,
        target=target,
        comparison=comparison,
        metadata={
            "route_family": "serial_dual_slm",
            "case": key,
            "naive_psi2": bool(naive_psi2),
            "wrong_carrier_sign": bool(wrong_carrier_sign),
            "vector_config": {
                "ideal_components": bool(vcfg.ideal_components),
                "quantise": bool(vcfg.quantise),
                "apply_carrier": bool(vcfg.apply_carrier),
                "apply_fill_factor": bool(vcfg.apply_fill_factor),
            },
            "model_status": MODEL_STATUS,
            "geometry_source": "existing_digital_twin_baseline",
            **extra,
        },
    )


def compare_vector_fields(field: VectorField, reference: VectorField) -> FieldComparison:
    """Compare vector fields after fitting one constant piston only."""

    inner = np.sum(np.conj(reference.ex) * field.ex + np.conj(reference.ey) * field.ey + np.conj(reference.ez) * field.ez)
    ref_norm = float(np.sqrt(np.sum(reference.intensity)))
    fld_norm = float(np.sqrt(np.sum(field.intensity)))
    piston = float(np.angle(inner))
    aligned = field.replace(
        ex=field.ex * np.exp(-1j * piston),
        ey=field.ey * np.exp(-1j * piston),
        ez=field.ez * np.exp(-1j * piston),
    )
    diff_int = (
        np.abs(aligned.ex - reference.ex) ** 2
        + np.abs(aligned.ey - reference.ey) ** 2
        + np.abs(aligned.ez - reference.ez) ** 2
    )
    norm = max(ref_norm, EPS)
    max_ref = max(float(np.max(np.sqrt(reference.intensity))), EPS)
    st_f = aligned.stokes()
    st_r = reference.stokes()
    st_scale = max(float(np.sqrt(np.mean(st_r["S0"] ** 2))), EPS)
    stokes_rms = np.sqrt(
        np.mean(
            (st_f["S0"] - st_r["S0"]) ** 2
            + (st_f["S1"] - st_r["S1"]) ** 2
            + (st_f["S2"] - st_r["S2"]) ** 2
            + (st_f["S3"] - st_r["S3"]) ** 2
        )
    ) / st_scale
    angle_f = local_polarization_angle(aligned)
    angle_r = local_polarization_angle(reference)
    mask = reference.intensity > 0.05 * float(np.max(reference.intensity))
    angle_err = headless_angle_delta(angle_f[mask], angle_r[mask]) if np.any(mask) else np.asarray([0.0])
    return FieldComparison(
        piston_rad=piston,
        complex_overlap=float(abs(inner) / max(ref_norm * fld_norm, EPS)),
        normalized_rms_error=float(np.sqrt(np.sum(diff_int)) / norm),
        normalized_max_error=float(np.max(np.sqrt(diff_int)) / max_ref),
        stokes_rms_error=float(stokes_rms),
        angle_rms_rad=float(np.sqrt(np.mean(angle_err**2))),
        power_ratio=float(field.power / max(reference.power, EPS)),
    )


def air_z_values(config: TwinConfig | None = None, *, planes: int = DEFAULT_FAST_Z_PLANES, span_factor: float = 1.0) -> np.ndarray:
    """Return a forward air-side z range through the predicted Bessel-like zone.

    ``propagate_vector_asm`` uses the repository's ``exp(+i kz z)`` convention;
    positive z decays evanescent content.  This digital-twin report therefore
    uses the focused surface plane as z=0 and scans forward through the air-side
    zone instead of doing long negative-z back-propagation.
    """

    twin = config or default_config("fast")
    design = compute_design_from_targets(twin.laser, twin.target, twin.material)
    span = float(span_factor) * float(design.target_bessel_length_m)
    return np.linspace(0.0, span, int(planes))


def propagate_route_through_shared_axicon(
    route: RouteFieldResult,
    twin_config: TwinConfig | None = None,
    *,
    z_values_m: Sequence[float] | None = None,
    canonical_axicon: VectorAxiconResult | None = None,
) -> RoutePropagationResult:
    """Feed one route into the shared vector axicon and compute raw metrics."""

    twin = twin_config or default_config("fast")
    z_values = np.asarray(z_values_m if z_values_m is not None else air_z_values(twin, planes=DEFAULT_FAST_Z_PLANES), dtype=float)
    result = run_vector_axicon_to_surface(route.field, twin, z_values_m=z_values)
    assert_locked_kr_fingerprint(result.parameters.k_r_surface_m_inv)
    metrics = tuple(hexagon_metrics_for_stack(result))
    overlaps: list[float] = []
    if canonical_axicon is not None:
        for idx, plane_intensity in enumerate(result.intensity_stack):
            ref = canonical_axicon.intensity_stack[idx]
            overlaps.append(float(np.sum(np.sqrt(np.maximum(plane_intensity, 0.0) * np.maximum(ref, 0.0))) / max(np.sqrt(np.sum(plane_intensity) * np.sum(ref)), EPS)))
    return RoutePropagationResult(route=route, axicon=result, metrics=metrics, output_overlap_to_canonical=tuple(overlaps))


def _peak_annulus_radius(intensity: np.ndarray, grid: Mapping[str, Any]) -> float:
    radius = np.asarray(grid["R"], dtype=float)
    bins = np.linspace(0.0, float(np.max(radius)), 96)
    labels = np.digitize(radius.ravel(), bins)
    flat = np.asarray(intensity, dtype=float).ravel()
    means: list[float] = []
    centres: list[float] = []
    for index in range(2, len(bins)):
        mask = labels == index
        if np.any(mask):
            means.append(float(np.mean(flat[mask])))
            centres.append(float(0.5 * (bins[index - 1] + bins[index])))
    return centres[int(np.argmax(means))] if centres else float(np.max(radius) / 3.0)


def _wall_continuity_from_profile(profile: np.ndarray, *, level: float = DEFAULT_WALL_CONTINUITY_LEVEL) -> float:
    prof = np.asarray(profile, dtype=float)
    peak = float(np.max(prof)) if prof.size else 0.0
    if peak <= EPS:
        return 0.0
    return float(np.mean(prof >= float(level) * peak))


def _hexagon_metrics_from_intensity_stack(
    intensity_stack: np.ndarray,
    z_values_m: Sequence[float],
    grid: Mapping[str, Any],
) -> list[HexagonPlaneMetrics]:
    """Compute raw hollow-hexagon metrics for an intensity z-stack."""

    stack = np.asarray(intensity_stack, dtype=float)
    z_values = np.asarray(z_values_m, dtype=float)
    radius = _peak_annulus_radius(stack[int(len(stack) // 2)], grid)
    curve = h6_z_curve(stack, z_values, grid, radius, angular_samples=1024)
    out: list[HexagonPlaneMetrics] = []
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    for idx, plane in enumerate(stack):
        profile = sample_ring_profile(plane, grid, radius, angular_samples=1440)
        sixfold = sixfold_from_intensity(plane, R, PHI, radius, angular_bins=720)
        accepted = hexagon_acceptance(plane, R, PHI, radius)
        ring_mask = (R >= 0.75 * radius) & (R <= 1.25 * radius)
        wall_power = float(np.sum(plane[ring_mask]))
        total = float(np.sum(plane))
        centre = tuple(s // 2 for s in plane.shape)
        core = float(plane[centre] / max(float(np.max(plane[ring_mask])) if np.any(ring_mask) else float(np.max(plane)), EPS))
        out.append(
            HexagonPlaneMetrics(
                z_m=float(z_values[idx]),
                ring_radius_m=float(radius),
                h6=float(curve["h6"][idx]),
                order6_over_order0=float(sixfold["order6_over_order0"]),
                order6_over_non_dc=float(sixfold["order6_over_non_dc"]),
                wall_continuity=_wall_continuity_from_profile(profile),
                core_darkness=core,
                wall_power_fraction=float(wall_power / max(total, EPS)),
                sidelobe_fraction=float(1.0 - wall_power / max(total, EPS)),
                hexagon_acceptance_pass=bool(accepted["pass"]),
            )
        )
    return out


def hexagon_metrics_for_stack(result: VectorAxiconResult) -> list[HexagonPlaneMetrics]:
    """Compute raw hollow-hexagon metrics for every propagated z plane."""

    return _hexagon_metrics_from_intensity_stack(result.intensity_stack, result.z_values_m, result.focal_plane.grid)


def hollow_hexagon_score(metric: HexagonPlaneMetrics) -> float:
    """Return a compact hollow-wall score; H6 remains a secondary diagnostic."""

    sixfold = max(float(metric.order6_over_non_dc), 0.0)
    sixfold = min(sixfold, 1.0)
    continuity = max(float(metric.wall_continuity), 0.0)
    dark_core = max(0.0, 1.0 - float(metric.core_darkness))
    wall_power = max(float(metric.wall_power_fraction), 0.0)
    sidelobe_penalty = max(0.0, 1.0 - float(metric.sidelobe_fraction))
    return float(sixfold * continuity * dark_core * wall_power * sidelobe_penalty)


def _component_energy_fractions(ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> tuple[float, float, float]:
    px = float(np.sum(np.abs(ex) ** 2))
    py = float(np.sum(np.abs(ey) ** 2))
    pz = float(np.sum(np.abs(ez) ** 2))
    total = max(px + py + pz, EPS)
    return px / total, py / total, pz / total


def _stokes_maps(field: VectorField) -> dict[str, np.ndarray]:
    """Return compact float32 Stokes maps for the retained focal reference plane."""

    stokes = field.stokes()
    return {key: np.asarray(value, dtype=np.float32) for key, value in stokes.items()}


def _focal_grid_from_pupil_grid(pupil_grid: Mapping[str, Any], twin: TwinConfig) -> dict[str, Any]:
    n = int(pupil_grid["N"])
    dx = float(pupil_grid["dx"])
    n_focus = float(twin.objective.immersion_n)
    dx_f = float(twin.laser.wavelength_m * twin.objective.f_eff_m / max(n_focus * n * dx, EPS))
    return make_xy_grid(n, dx_f)


def _objective_direction_cosines(
    pupil_grid: Mapping[str, Any],
    twin: TwinConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map inherited objective-pupil coordinates to direction cosines."""

    X = np.asarray(pupil_grid["X"], dtype=float)
    Y = np.asarray(pupil_grid["Y"], dtype=float)
    R = np.asarray(pupil_grid["R"], dtype=float)
    pupil_radius = float(twin.objective.pupil_radius_m)
    n_focus = float(twin.objective.immersion_n)
    sin_theta_max = min(float(twin.objective.NA) / max(n_focus, EPS), 1.0 - 1.0e-12)
    sx = sin_theta_max * X / max(pupil_radius, EPS)
    sy = sin_theta_max * Y / max(pupil_radius, EPS)
    s2 = sx * sx + sy * sy
    mask = (R <= pupil_radius) & (s2 < 1.0)
    sz = np.zeros_like(sx, dtype=float)
    sz[mask] = np.sqrt(np.maximum(1.0 - s2[mask], 0.0))
    meta = {
        "pupil_radius_m": pupil_radius,
        "objective_NA": float(twin.objective.NA),
        "objective_f_eff_m": float(twin.objective.f_eff_m),
        "focus_medium_index": n_focus,
        "sin_theta_max": float(sin_theta_max),
        "pupil_sample_count": int(np.count_nonzero(mask)),
    }
    return sx, sy, sz, mask, meta


def _project_pupil_field_to_ray_transverse_basis(
    field: VectorField,
    twin: TwinConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Project a post-axicon pupil field onto the local plane normal to each ray."""

    sx, sy, sz, mask, meta = _objective_direction_cosines(field.grid, twin)
    dot = sx * field.ex + sy * field.ey + sz * field.ez
    ex_p = field.ex - sx * dot
    ey_p = field.ey - sy * dot
    ez_p = field.ez - sz * dot
    apod = np.zeros_like(sz, dtype=float)
    apod[mask] = np.sqrt(np.maximum(sz[mask], 0.0))
    ex_p = np.where(mask, ex_p * apod, 0.0)
    ey_p = np.where(mask, ey_p * apod, 0.0)
    ez_p = np.where(mask, ez_p * apod, 0.0)
    residual = np.abs(sx * ex_p + sy * ey_p + sz * ez_p)
    amplitude = np.sqrt(np.abs(ex_p) ** 2 + np.abs(ey_p) ** 2 + np.abs(ez_p) ** 2)
    meta = {
        **meta,
        "pupil_projection": "E_perp = E - s(s dot E)",
        "apodization": "sqrt(cos(theta))",
        "pupil_projection_residual": float(np.nanmax(residual) / max(float(np.nanmax(amplitude)), EPS)),
    }
    return sx, sy, sz, ex_p, ey_p, ez_p, meta


def _direct_vectorial_pupil_sum(
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    ex_weights: np.ndarray,
    ey_weights: np.ndarray,
    ez_weights: np.ndarray,
    focal_grid: Mapping[str, Any],
    z_values_m: Sequence[float],
    *,
    chunk_size: int,
    wavelength_m: float,
    medium_index: float,
) -> tuple[VectorField, ...]:
    """Small-grid direct reference for benchmarking the FFT pupil-spectrum path."""

    x_out = np.asarray(focal_grid["X"], dtype=float).ravel()
    y_out = np.asarray(focal_grid["Y"], dtype=float).ravel()
    out_shape = np.asarray(focal_grid["X"]).shape
    fields: list[VectorField] = []
    chunk = max(1, int(chunk_size))
    for z in np.asarray(z_values_m, dtype=float):
        ex_flat = np.empty_like(x_out, dtype=np.complex128)
        ey_flat = np.empty_like(x_out, dtype=np.complex128)
        ez_flat = np.empty_like(x_out, dtype=np.complex128)
        z_phase = float(z) * kz
        for start in range(0, x_out.size, chunk):
            stop = min(start + chunk, x_out.size)
            phase = np.exp(
                1j
                * (
                    z_phase[None, :]
                    - x_out[start:stop, None] * kx[None, :]
                    - y_out[start:stop, None] * ky[None, :]
                )
            )
            ex_flat[start:stop] = phase @ ex_weights
            ey_flat[start:stop] = phase @ ey_weights
            ez_flat[start:stop] = phase @ ez_weights
        fields.append(
            VectorField(
                ex=ex_flat.reshape(out_shape),
                ey=ey_flat.reshape(out_shape),
                ez=ez_flat.reshape(out_shape),
                grid=focal_grid,
                wavelength_m=float(wavelength_m),
                medium_index=float(medium_index),
                metadata={"stage": "nathan_direct_vectorial_pupil_spectrum_reference", "z_m": float(z)},
            )
        )
    return tuple(fields)


def _fft_vectorial_pupil_spectrum(
    ex_p: np.ndarray,
    ey_p: np.ndarray,
    ez_p: np.ndarray,
    kz: np.ndarray,
    quadrature: complex,
    focal_grid: Mapping[str, Any],
    z_values_m: Sequence[float],
    *,
    wavelength_m: float,
    medium_index: float,
) -> tuple[VectorField, ...]:
    """FFT-compatible vectorial pupil-spectrum propagation."""

    fields: list[VectorField] = []
    for z in np.asarray(z_values_m, dtype=float):
        transfer = np.exp(1j * kz * float(z))
        fields.append(
            VectorField(
                ex=fft2c(ex_p * transfer) * quadrature,
                ey=fft2c(ey_p * transfer) * quadrature,
                ez=fft2c(ez_p * transfer) * quadrature,
                grid=focal_grid,
                wavelength_m=float(wavelength_m),
                medium_index=float(medium_index),
                metadata={"stage": "nathan_fft_vectorial_pupil_spectrum_reference", "z_m": float(z)},
            )
        )
    return tuple(fields)


def _vectorial_pupil_spectrum_stack(
    pupil_field: VectorField,
    twin: TwinConfig,
    z_values_m: Sequence[float],
    *,
    solver: str = "fft",
    chunk_size: int = 256,
) -> tuple[tuple[VectorField, ...], dict[str, Any]]:
    """Vectorial pupil-spectrum reference for validating the scalar focus bridge.

    This is intentionally not a global replacement for the scalar ObjectiveMap
    helper.  It is a narrow Nathan validation reference: the post-axicon pupil
    field is projected onto each ray's transverse plane, weighted by a simple
    aplanatic ``sqrt(cos(theta))`` factor, then propagated as an NA-limited
    vectorial pupil spectrum.  The default solver uses centred FFTs; the direct
    plane-wave summation is retained only for small-grid benchmarking.
    """

    sx, sy, sz, ex_p, ey_p, ez_p, meta = _project_pupil_field_to_ray_transverse_basis(pupil_field, twin)
    has_pupil_samples = int(meta["pupil_sample_count"]) > 0
    if not has_pupil_samples:
        raise ValueError("No pupil samples fall inside the inherited objective pupil.")
    valid = np.abs(ex_p) + np.abs(ey_p) + np.abs(ez_p) > 0.0
    if not np.any(valid):
        raise ValueError("Projected vectorial pupil field has zero amplitude.")

    n_focus = float(twin.objective.immersion_n)
    k = TWOPI * n_focus / float(twin.laser.wavelength_m)
    kx_2d = k * sx
    ky_2d = k * sy
    kz_2d = k * sz
    dx = float(pupil_field.grid["dx"])
    dy = float(pupil_field.grid.get("dy", dx))
    quadrature = dx * dy / (1j * float(twin.laser.wavelength_m) * float(twin.objective.f_eff_m))

    focal_grid = _focal_grid_from_pupil_grid(pupil_field.grid, twin)
    chunk = max(1, int(chunk_size))
    z_values = np.asarray(z_values_m, dtype=float)

    solver_key = str(solver).lower().strip()
    if solver_key in {"fft", "pupil_spectrum_fft", "vectorial_fft"}:
        fields = _fft_vectorial_pupil_spectrum(
            ex_p,
            ey_p,
            ez_p,
            kz_2d,
            quadrature,
            focal_grid,
            z_values,
            wavelength_m=float(twin.laser.wavelength_m),
            medium_index=n_focus,
        )
    elif solver_key in {"direct", "direct_sum", "plane_wave_sum"}:
        fields = _direct_vectorial_pupil_sum(
            kx_2d[valid],
            ky_2d[valid],
            kz_2d[valid],
            np.asarray(ex_p[valid], dtype=np.complex128) * quadrature,
            np.asarray(ey_p[valid], dtype=np.complex128) * quadrature,
            np.asarray(ez_p[valid], dtype=np.complex128) * quadrature,
            focal_grid,
            z_values,
            chunk_size=chunk,
            wavelength_m=float(twin.laser.wavelength_m),
            medium_index=n_focus,
        )
    else:
        raise ValueError(f"Unsupported vectorial pupil-spectrum solver: {solver!r}")

    fields = tuple(
        plane.replace(
            metadata={
                **dict(pupil_field.metadata),
                **dict(plane.metadata),
                "objective_model": "F2 direct vectorial plane-wave pupil-to-focus reference",
                "pupil_spectrum_solver": solver_key,
            }
        )
        for plane in fields
    )
    meta = {
        **meta,
        "method": "vectorial_plane_wave_pupil_spectrum_reference",
        "solver": solver_key,
        "quadrature": "pupil dx*dy/(i lambda f)",
        "active_pupil_samples_after_projection": int(np.count_nonzero(valid)),
        "output_grid_N": int(focal_grid["N"]),
        "output_grid_dx_m": float(focal_grid["dx"]),
        "chunk_size": chunk,
    }
    return tuple(fields), meta


def vectorial_pupil_spectrum_reference(
    pupil_field: VectorField,
    twin: TwinConfig,
    z_values_m: Sequence[float],
    *,
    solver: str = "fft",
    chunk_size: int = 256,
) -> tuple[VectorField, ...]:
    """Return focal vector fields from the scoped Nathan F2 reference."""

    fields, _ = _vectorial_pupil_spectrum_stack(pupil_field, twin, z_values_m, solver=solver, chunk_size=chunk_size)
    return fields


def _vector_downstream_result(
    field: VectorField,
    twin: TwinConfig,
    z_values_m: Sequence[float],
    *,
    control_id: str,
    route_id: str,
    route_role: str,
    expected_surface_kr_m_inv: float | None = None,
) -> DownstreamRouteResult:
    """Run the current vector-aware downstream path and retain component diagnostics.

    ``expected_surface_kr_m_inv=None`` keeps the inherited locked-fingerprint
    tripwire.  A MODE 1E redesigned candidate must pass its own declared surface
    k_r instead; the tripwire still fires on any mismatch, it is never skipped.
    """

    z_values = np.asarray(z_values_m, dtype=float)
    result = run_vector_axicon_to_surface(field, twin, z_values_m=z_values)
    if expected_surface_kr_m_inv is None:
        assert_locked_kr_fingerprint(result.parameters.k_r_surface_m_inv)
    else:
        assert_locked_kr_fingerprint(result.parameters.k_r_surface_m_inv, expected_m_inv=float(expected_surface_kr_m_inv))
    fractions = []
    z_powers: list[float] = []
    on_axis_ez: list[float] = []
    on_axis_total: list[float] = []
    for z in z_values:
        plane = propagate_vector_asm(result.focal_plane, float(z))
        fractions.append(_component_energy_fractions(plane.ex, plane.ey, plane.ez))
        z_powers.append(float(plane.power))
        centre = tuple(s // 2 for s in plane.ex.shape)
        on_axis_ez.append(float(np.abs(plane.ez[centre]) ** 2))
        on_axis_total.append(float(plane.intensity[centre]))
    ex_f, ey_f, ez_f = zip(*fractions, strict=True)
    return DownstreamRouteResult(
        route_id=route_id,
        control_id=control_id,
        z_values_m=z_values,
        intensity_stack=np.asarray(result.intensity_stack, dtype=np.float32),
        metrics=tuple(hexagon_metrics_for_stack(result)),
        ex_energy_fraction=tuple(float(v) for v in ex_f),
        ey_energy_fraction=tuple(float(v) for v in ey_f),
        ez_energy_fraction=tuple(float(v) for v in ez_f),
        transversality_residual=spectral_transversality_residual(result.focal_plane),
        metadata={
            "route_role": route_role,
            "input_field_type": "VectorField",
            "output_field_type": "DownstreamRouteResult with intensity stack plus component fractions",
            "retains_ex_ey_separately": True,
            "ez_generated_or_retained": True,
            "stokes_retained_until_intensity_stack": True,
            "reduced_to_scalar_intensity_only_at_stack": True,
            "scalar_kr_bessel_geometry": "uses TwinConfig-derived scalar kr for axicon/focus scale",
            "scalar_pupil_focus_mapping": True,
            "vector_asm_projection": True,
            "ps_fresnel_axicon": True,
            "objective_model": "scalar FFT focus applied independently to Ex and Ey",
            "source_functions": "vector_axicon.run_vector_axicon_to_surface -> apply_vector_axicon -> focus_vector_to_surface -> propagate_vector_asm",
            "pupil_power": float(field.power),
            "post_axicon_power": float(result.after_axicon.power),
            "focal_power": float(result.focal_plane.power),
            "z_power_by_plane": tuple(z_powers),
            "mean_z_power": float(np.mean(z_powers)),
            "on_axis_ez_intensity_by_z": tuple(on_axis_ez),
            "on_axis_total_intensity_by_z": tuple(on_axis_total),
            "output_grid_N": int(result.focal_plane.grid["N"]),
            "output_grid_dx_m": float(result.focal_plane.grid["dx"]),
            "absolute_scaling_status": "native current Digital Twin scalar-focus scaling",
        },
        stokes_maps=_stokes_maps(result.focal_plane),
    )


def _scalar_component_surrogate_result(
    field: VectorField,
    twin: TwinConfig,
    z_values_m: Sequence[float],
    *,
    control_id: str,
    route_id: str = "S1_scalar_component_surrogate",
    route_role: str = "diagnostic_surrogate",
) -> DownstreamRouteResult:
    """Run the S1 diagnostic: scalar axicon/focus/ASM for Ex and Ey separately."""

    z_values = np.asarray(z_values_m, dtype=float)
    params = resolve_vector_axicon_parameters(twin)
    grid = field.grid
    R = np.asarray(grid["R"], dtype=float)
    phase = np.exp(-1j * abs(float(params.k_r_pre_m_inv)) * R)
    if params.aperture_radius_m is None:
        aperture = np.ones_like(R, dtype=float)
    else:
        aperture = (R <= float(params.aperture_radius_m)).astype(float)
    ex_ax = field.ex * phase * aperture
    ey_ax = field.ey * phase * aperture
    ex_f, focal_grid = focus_to_focal_plane(ex_ax, dict(grid), twin.laser, twin.objective)
    ey_f, _ = focus_to_focal_plane(ey_ax, dict(grid), twin.laser, twin.objective)

    stack: list[np.ndarray] = []
    fractions: list[tuple[float, float, float]] = []
    on_axis_total: list[float] = []
    for z in z_values:
        ex_z = angular_spectrum_propagate_bl(ex_f, focal_grid, twin.laser.wavelength_m, float(z), n_medium=1.0)
        ey_z = angular_spectrum_propagate_bl(ey_f, focal_grid, twin.laser.wavelength_m, float(z), n_medium=1.0)
        intensity = np.abs(ex_z) ** 2 + np.abs(ey_z) ** 2
        stack.append(intensity.astype(np.float32))
        fractions.append(_component_energy_fractions(ex_z, ey_z, np.zeros_like(ex_z)))
        centre = tuple(s // 2 for s in intensity.shape)
        on_axis_total.append(float(intensity[centre]))
    intensity_stack = np.asarray(stack, dtype=np.float32)
    ex_frac, ey_frac, ez_frac = zip(*fractions, strict=True)
    metrics = _hexagon_metrics_from_intensity_stack(intensity_stack, z_values, focal_grid)
    focal_vector = VectorField(
        ex=ex_f,
        ey=ey_f,
        ez=np.zeros_like(ex_f, dtype=complex),
        grid=focal_grid,
        wavelength_m=twin.laser.wavelength_m,
        medium_index=1.0,
        metadata={"stage": "scalar_component_surrogate_focus"},
    )
    return DownstreamRouteResult(
        route_id=route_id,
        control_id=control_id,
        z_values_m=z_values,
        intensity_stack=intensity_stack,
        metrics=tuple(metrics),
        ex_energy_fraction=tuple(float(v) for v in ex_frac),
        ey_energy_fraction=tuple(float(v) for v in ey_frac),
        ez_energy_fraction=tuple(float(v) for v in ez_frac),
        transversality_residual=None,
        metadata={
            "route_role": route_role,
            "input_field_type": "VectorField split into scalar Ex/Ey arrays",
            "output_field_type": "DownstreamRouteResult with scalar Ex/Ey intensity sum",
            "retains_ex_ey_separately": True,
            "ez_generated_or_retained": False,
            "stokes_retained_until_intensity_stack": False,
            "reduced_to_scalar_intensity_only_at_stack": True,
            "scalar_kr_bessel_geometry": "uses TwinConfig-derived scalar kr for thin axicon phase only",
            "scalar_pupil_focus_mapping": True,
            "vector_asm_projection": False,
            "ps_fresnel_axicon": False,
            "objective_model": "scalar FFT focus applied independently to Ex and Ey",
            "source_functions": "resolve_vector_axicon_parameters -> focus_to_focal_plane -> angular_spectrum_propagate_bl",
            "pupil_power": float(field.power),
            "post_axicon_power": float((np.sum(np.abs(ex_ax) ** 2 + np.abs(ey_ax) ** 2) * float(grid["dx"]) * float(grid.get("dy", grid["dx"])))),
            "focal_power": float(focal_vector.power),
            "z_power_by_plane": _intensity_stack_power(intensity_stack, focal_grid),
            "mean_z_power": float(np.mean(_intensity_stack_power(intensity_stack, focal_grid))),
            "on_axis_ez_intensity_by_z": tuple(0.0 for _ in z_values),
            "on_axis_total_intensity_by_z": tuple(on_axis_total),
            "output_grid_N": int(focal_grid["N"]),
            "output_grid_dx_m": float(focal_grid["dx"]),
            "absolute_scaling_status": "native scalar-component surrogate scaling",
        },
        stokes_maps=_stokes_maps(focal_vector),
    )


def _vectorial_pupil_spectrum_reference_result(
    field: VectorField,
    twin: TwinConfig,
    z_values_m: Sequence[float],
    *,
    control_id: str,
    solver: str = "fft",
    chunk_size: int = 256,
    expected_surface_kr_m_inv: float | None = None,
) -> DownstreamRouteResult:
    """Run F2: vector axicon followed by a scoped vectorial pupil-spectrum reference.

    ``expected_surface_kr_m_inv`` mirrors :func:`_vector_downstream_result`: ``None``
    keeps the inherited locked fingerprint, a MODE 1E candidate passes its own value.
    """

    z_values = np.asarray(z_values_m, dtype=float)
    after_axicon, params, ax_meta = apply_vector_axicon(field, twin)
    if expected_surface_kr_m_inv is None:
        assert_locked_kr_fingerprint(params.k_r_surface_m_inv)
    else:
        assert_locked_kr_fingerprint(params.k_r_surface_m_inv, expected_m_inv=float(expected_surface_kr_m_inv))
    focal_fields, focus_meta = _vectorial_pupil_spectrum_stack(after_axicon, twin, z_values, solver=solver, chunk_size=chunk_size)
    intensity_stack = np.asarray([plane.intensity.astype(np.float32) for plane in focal_fields], dtype=np.float32)
    fractions = [_component_energy_fractions(plane.ex, plane.ey, plane.ez) for plane in focal_fields]
    ex_frac, ey_frac, ez_frac = zip(*fractions, strict=True)
    metrics = _hexagon_metrics_from_intensity_stack(intensity_stack, z_values, focal_fields[0].grid)
    return DownstreamRouteResult(
        route_id="F2_vectorial_pupil_spectrum_reference",
        control_id=control_id,
        z_values_m=z_values,
        intensity_stack=intensity_stack,
        metrics=tuple(metrics),
        ex_energy_fraction=tuple(float(v) for v in ex_frac),
        ey_energy_fraction=tuple(float(v) for v in ey_frac),
        ez_energy_fraction=tuple(float(v) for v in ez_frac),
        transversality_residual=float(focus_meta["pupil_projection_residual"]),
        metadata={
            "route_role": "vectorial_focus_reference",
            "input_field_type": "VectorField after vector axicon",
            "output_field_type": "DownstreamRouteResult with vectorial pupil-spectrum intensity stack plus component fractions",
            "retains_ex_ey_separately": True,
            "ez_generated_or_retained": True,
            "stokes_retained_until_intensity_stack": True,
            "reduced_to_scalar_intensity_only_at_stack": True,
            "scalar_kr_bessel_geometry": "uses inherited axicon kr only before vectorial focus reference",
            "scalar_pupil_focus_mapping": False,
            "vector_asm_projection": False,
            "ps_fresnel_axicon": True,
            "objective_model": "F2 direct vectorial plane-wave pupil-to-focus reference",
            "source_functions": "apply_vector_axicon -> vectorial_pupil_spectrum_reference",
            "absolute_scaling_status": "uncalibrated; use equal-power shape metrics before interpreting F0-F2 differences",
            "axicon": ax_meta,
            "focus_reference": focus_meta,
            "pupil_power": float(field.power),
            "post_axicon_power": float(after_axicon.power),
            "focal_power": float(focal_fields[0].power),
            "z_power_by_plane": tuple(float(plane.power) for plane in focal_fields),
            "mean_z_power": float(np.mean([plane.power for plane in focal_fields])),
            "on_axis_ez_intensity_by_z": tuple(float(np.abs(plane.ez[tuple(s // 2 for s in plane.ez.shape)]) ** 2) for plane in focal_fields),
            "on_axis_total_intensity_by_z": tuple(float(plane.intensity[tuple(s // 2 for s in plane.intensity.shape)]) for plane in focal_fields),
            "output_grid_N": int(focal_fields[0].grid["N"]),
            "output_grid_dx_m": float(focal_fields[0].grid["dx"]),
        },
        stokes_maps=_stokes_maps(focal_fields[0]),
    )


def _six_lobe_lattice_field(config: NathanHexagonConfig, grid: Mapping[str, Any]) -> VectorField:
    """Return a linearly polarised six-spot lattice control at the handoff plane."""

    R = np.asarray(grid["R"], dtype=float)
    radius = 0.45 * float(np.max(R))
    sigma = max(float(grid["dx"]) * 1.5, radius / 60.0)
    intensity = np.zeros_like(R, dtype=float)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    for angle in np.linspace(0.0, TWOPI, 6, endpoint=False):
        x0 = radius * np.cos(angle)
        y0 = radius * np.sin(angle)
        intensity += np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2.0 * sigma**2))
    amp = np.sqrt(np.maximum(intensity, 0.0))
    return VectorField(
        ex=amp.astype(complex),
        ey=np.zeros_like(amp, dtype=complex),
        ez=np.zeros_like(amp, dtype=complex),
        grid=grid,
        wavelength_m=config.vector.wavelength_m,
        medium_index=1.0,
        metadata={"stage": STAGE, "field": "six_lobe_lattice_control", "geometry_source": "diagnostic_control_only"},
    )


def downstream_control_fields(config: NathanHexagonConfig | None = None) -> dict[str, VectorField]:
    """Return same-plane handoff fields for downstream model comparison."""

    cfg = config or NathanHexagonConfig.fast()
    grid = default_nathan_grid(cfg)
    return {
        "scalar_bessel_gaussian_baseline": canonical_target_field(cfg, grid=grid, control="linear_x"),
        "all_radial": canonical_target_field(cfg, grid=grid, control="radial"),
        "all_azimuthal": canonical_target_field(cfg, grid=grid, control="azimuthal"),
        "nathan_six_sector": canonical_target_field(cfg, grid=grid),
        "six_lobe_lattice": _six_lobe_lattice_field(cfg, grid),
        "zero_vector_structure_gaussian": canonical_target_field(cfg, grid=grid, control="linear_x"),
    }


def _normalised_stack_rms(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    denom = max(float(np.sqrt(np.sum(bb**2))), EPS)
    return float(np.sqrt(np.sum((aa - bb) ** 2)) / denom)


def _intensity_stack_power(stack: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, ...]:
    dx = float(grid["dx"])
    dy = float(grid.get("dy", dx))
    return tuple(float(np.sum(plane) * dx * dy) for plane in np.asarray(stack, dtype=float))


def _equal_power_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack, dtype=float)
    denom = np.sum(arr, axis=(-2, -1), keepdims=True)
    return np.divide(arr, np.maximum(denom, EPS))


def _central_crop_stack(stack: np.ndarray, crop_fraction: float = 0.5) -> np.ndarray:
    arr = np.asarray(stack)
    frac = min(max(float(crop_fraction), EPS), 1.0)
    ny = arr.shape[-2]
    nx = arr.shape[-1]
    crop_y = max(1, int(round(frac * ny)))
    crop_x = max(1, int(round(frac * nx)))
    y0 = max(0, (ny - crop_y) // 2)
    x0 = max(0, (nx - crop_x) // 2)
    return arr[..., y0 : y0 + crop_y, x0 : x0 + crop_x]


def _equal_power_shape_metrics(a: np.ndarray, b: np.ndarray, *, crop_fraction: float = 1.0) -> dict[str, float]:
    aa = _equal_power_stack(_central_crop_stack(a, crop_fraction))
    bb = _equal_power_stack(_central_crop_stack(b, crop_fraction))
    diff = aa - bb
    denom = max(float(np.sqrt(np.sum(bb**2))), EPS)
    dot = float(np.sum(aa * bb))
    corr = dot / max(float(np.sqrt(np.sum(aa**2) * np.sum(bb**2))), EPS)
    return {
        "equal_power_shape_rms": float(np.sqrt(np.sum(diff**2)) / denom),
        "equal_power_intensity_correlation": float(corr),
    }


def _power_row(result: DownstreamRouteResult) -> dict[str, Any]:
    return {
        "route_id": result.route_id,
        "control_id": result.control_id,
        "pupil_power": result.metadata.get("pupil_power", ""),
        "post_axicon_power": result.metadata.get("post_axicon_power", ""),
        "focal_power": result.metadata.get("focal_power", ""),
        "mean_z_power": result.metadata.get("mean_z_power", ""),
        "absolute_scaling_status": result.metadata.get("absolute_scaling_status", "native route scaling"),
    }


def _best_metric_row(result: DownstreamRouteResult) -> dict[str, Any]:
    best = max(result.metrics, key=hollow_hexagon_score)
    return {
        "route_id": result.route_id,
        "control_id": result.control_id,
        "best_z_m": best.z_m,
        "best_hollow_hexagon_score": hollow_hexagon_score(best),
        "best_wall_continuity": best.wall_continuity,
        "best_core_darkness": best.core_darkness,
        "best_sidelobe_fraction": best.sidelobe_fraction,
        "best_wall_power_fraction": best.wall_power_fraction,
        "mean_ez_energy_fraction": float(np.mean(result.ez_energy_fraction)),
        "transversality_residual": "" if result.transversality_residual is None else float(result.transversality_residual),
    }


def downstream_sampling_audit_rows(config: NathanHexagonConfig | None = None) -> list[dict[str, Any]]:
    """Return numerical-sampling facts for the downstream audit."""

    cfg = config or NathanHexagonConfig.fast()
    twin = _twin_with_axial_points(cfg.twin, cfg.z_planes)
    grid = default_nathan_grid(cfg)
    design = compute_design_from_targets(twin.laser, twin.target, twin.material)
    z_values = air_z_values(twin, planes=cfg.z_planes, span_factor=cfg.z_span_factor)
    n = int(grid["N"])
    dx = float(grid["dx"])
    input_window = n * dx
    focal_dx = float(twin.laser.wavelength_m * twin.objective.f_eff_m / max(n * dx, EPS))
    carrier_cpm = float(twin.slm.carrier_cpm)
    z_step = float(np.median(np.diff(z_values))) if len(z_values) > 1 else 0.0
    radial_period = TWOPI / max(float(design.kr_sample_m_inv), EPS)
    return [
        {"item": "handoff_grid_N", "value": n, "units": "samples", "source": "default_nathan_grid"},
        {"item": "handoff_dx_m", "value": dx, "units": "m", "source": "default_nathan_grid"},
        {"item": "handoff_window_m", "value": input_window, "units": "m", "source": "N * dx"},
        {"item": "slm_pixel_pitch_m", "value": twin.slm.pixel_pitch_m, "units": "m", "source": "TwinConfig.slm.pixel_pitch_m"},
        {"item": "samples_per_slm_pixel_at_handoff", "value": float(twin.slm.pixel_pitch_m / max(dx, EPS)), "units": "samples/pixel", "source": "slm.pixel_pitch_m / handoff_dx_m"},
        {"item": "carrier_frequency_cpm", "value": carrier_cpm, "units": "cycles/m", "source": "TwinConfig.slm.carrier_cpm"},
        {"item": "carrier_nyquist_margin", "value": float((0.5 / max(dx, EPS)) / max(abs(carrier_cpm), EPS)), "units": "ratio", "source": "handoff Nyquist / carrier_cpm"},
        {"item": "focal_dx_m", "value": focal_dx, "units": "m", "source": "focus_to_focal_plane dx_f"},
        {"item": "focal_window_m", "value": n * focal_dx, "units": "m", "source": "N * focal_dx"},
        {"item": "z_plane_count", "value": int(len(z_values)), "units": "planes", "source": "air_z_values"},
        {"item": "z_step_m", "value": z_step, "units": "m", "source": "median diff(air_z_values)"},
        {"item": "kr_sample_m_inv", "value": design.kr_sample_m_inv, "units": "rad/m", "source": "compute_design_from_targets"},
        {"item": "radial_period_m", "value": radial_period, "units": "m", "source": "2*pi/kr_sample"},
        {"item": "samples_per_radial_period_at_focus", "value": float(radial_period / max(focal_dx, EPS)), "units": "samples/period", "source": "radial_period / focal_dx"},
        {"item": "single_grid_warning", "value": bool(twin.slm.pixel_pitch_m / max(dx, EPS) < 2.0), "units": "bool", "source": "True means handoff grid cannot resolve SLM pixels with >=2 samples"},
    ]


def downstream_focus_multiscale_sampling_rows(
    config: NathanHexagonConfig | None = None,
    *,
    grid_ns: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return sampling facts at multiple handoff grids for the focus gate."""

    cfg = config or NathanHexagonConfig.fast()
    if grid_ns is None:
        candidates = {max(16, int(cfg.grid_n) // 2), int(cfg.grid_n), int(cfg.grid_n) * 2}
    else:
        candidates = {int(value) for value in grid_ns}
    rows: list[dict[str, Any]] = []
    for n in sorted(value for value in candidates if value > 0):
        sub_cfg = replace(cfg, grid_n=int(n))
        for row in downstream_sampling_audit_rows(sub_cfg):
            rows.append({"grid_n": int(n), **row})
    return rows


def build_downstream_model_comparison_gate(
    config: NathanHexagonConfig | None = None,
    *,
    twin_config: TwinConfig | None = None,
    control_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the S0/S1/V1 downstream-model audit gate without expanding the study."""

    cfg = config or NathanHexagonConfig.fast()
    twin = _twin_with_axial_points(twin_config or cfg.twin, cfg.z_planes)
    z_values = air_z_values(twin, planes=cfg.z_planes, span_factor=cfg.z_span_factor)
    fields = downstream_control_fields(cfg)
    wanted = tuple(control_ids) if control_ids is not None else tuple(fields)
    results: list[DownstreamRouteResult] = []
    for control_id in wanted:
        field = fields[str(control_id)]
        s0 = _vector_downstream_result(
            field,
            twin,
            z_values,
            control_id=str(control_id),
            route_id="S0_existing_current_digital_twin_path",
            route_role="control_current_path_unchanged",
        )
        s1 = _scalar_component_surrogate_result(field, twin, z_values, control_id=str(control_id))
        v1 = replace(
            s0,
            route_id="V1_vector_downstream_reference_current",
            metadata={
                **dict(s0.metadata),
                "route_role": "vector_reference_reuses_current_vector_axicon_scalar_focus_vector_asm",
                "same_implementation_as_S0": True,
                "missing_vectorial_focus_reference": True,
            },
        )
        results.extend([s0, s1, v1])

    by_key = {(result.route_id, result.control_id): result for result in results}
    comparisons: dict[str, Any] = {}
    for route_id in ("S0_existing_current_digital_twin_path", "S1_scalar_component_surrogate", "V1_vector_downstream_reference_current"):
        radial = by_key.get((route_id, "all_radial"))
        azimuthal = by_key.get((route_id, "all_azimuthal"))
        if radial is not None and azimuthal is not None:
            comparisons[f"{route_id}:radial_vs_azimuthal_intensity_rms"] = _normalised_stack_rms(
                radial.intensity_stack, azimuthal.intensity_stack
            )
    canonical_s0 = by_key.get(("S0_existing_current_digital_twin_path", "nathan_six_sector"))
    canonical_s1 = by_key.get(("S1_scalar_component_surrogate", "nathan_six_sector"))
    canonical_v1 = by_key.get(("V1_vector_downstream_reference_current", "nathan_six_sector"))
    if canonical_s0 is not None and canonical_v1 is not None:
        comparisons["S0_minus_V1_canonical_intensity_rms"] = _normalised_stack_rms(canonical_s0.intensity_stack, canonical_v1.intensity_stack)
    if canonical_s1 is not None and canonical_v1 is not None:
        comparisons["S1_minus_V1_canonical_intensity_rms"] = _normalised_stack_rms(canonical_s1.intensity_stack, canonical_v1.intensity_stack)

    return {
        "status": "diagnostic_gate_only_no_final_claim",
        "route_results": tuple(results),
        "metric_rows": tuple(_best_metric_row(result) for result in results),
        "comparisons": comparisons,
        "sampling_audit_rows": tuple(downstream_sampling_audit_rows(cfg)),
        "model_boundary": (
            "S0 is the current Digital Twin vector downstream path.",
            "S1 is a scalar-component surrogate with no p/s Fresnel split, vector projection, or Ez generation.",
            "V1 currently reuses S0 because the implemented downstream path is vector-aware through axicon and ASM, but still uses scalar FFT focus per transverse component.",
            "No conclusion A/B/C is made by this gate alone; resolution-gated S0/S1/V1 comparisons are required.",
        ),
    }


def build_downstream_focus_validation_gate(
    config: NathanHexagonConfig | None = None,
    *,
    twin_config: TwinConfig | None = None,
    control_ids: Sequence[str] | None = None,
    f2_solver: str = "fft",
    f2_chunk_size: int = 256,
    paper_resolution_f2_converged: bool = False,
) -> dict[str, Any]:
    """Build the narrow F0/F1/F2 validation gate for the downstream focus bridge."""

    cfg = config or NathanHexagonConfig.fast()
    if paper_resolution_f2_converged and (int(cfg.grid_n) < 512 or int(cfg.z_planes) < 41):
        raise ValueError("paper_resolution_f2_converged requires at least grid_n=512 and z_planes=41.")

    twin = _twin_with_axial_points(twin_config or cfg.twin, cfg.z_planes)
    z_values = air_z_values(twin, planes=cfg.z_planes, span_factor=cfg.z_span_factor)
    fields = downstream_control_fields(cfg)
    default_controls = (
        "scalar_bessel_gaussian_baseline",
        "all_radial",
        "all_azimuthal",
        "nathan_six_sector",
    )
    wanted = tuple(control_ids) if control_ids is not None else default_controls

    results: list[DownstreamRouteResult] = []
    for control_id in wanted:
        field = fields[str(control_id)]
        f0 = _vector_downstream_result(
            field,
            twin,
            z_values,
            control_id=str(control_id),
            route_id="F0_current_scalar_focus_bridge",
            route_role="current_vector_axicon_scalar_component_focus_bridge",
        )
        f1 = _scalar_component_surrogate_result(
            field,
            twin,
            z_values,
            control_id=str(control_id),
            route_id="F1_scalar_component_surrogate",
            route_role="scalar_component_surrogate_no_ps_no_vector_projection_no_ez",
        )
        f2 = _vectorial_pupil_spectrum_reference_result(
            field,
            twin,
            z_values,
            control_id=str(control_id),
            solver=f2_solver,
            chunk_size=f2_chunk_size,
        )
        results.extend([f0, f1, f2])

    by_key = {(result.route_id, result.control_id): result for result in results}
    comparisons: dict[str, Any] = {}
    route_ids = (
        "F0_current_scalar_focus_bridge",
        "F1_scalar_component_surrogate",
        "F2_vectorial_pupil_spectrum_reference",
    )
    for route_id in route_ids:
        radial = by_key.get((route_id, "all_radial"))
        azimuthal = by_key.get((route_id, "all_azimuthal"))
        if radial is not None and azimuthal is not None:
            comparisons[f"{route_id}:radial_vs_azimuthal_intensity_rms"] = _normalised_stack_rms(
                radial.intensity_stack, azimuthal.intensity_stack
            )

    difference_maps: dict[str, np.ndarray] = {}
    raw_difference_maps: dict[str, np.ndarray] = {}
    for control_id in wanted:
        f2 = by_key.get(("F2_vectorial_pupil_spectrum_reference", str(control_id)))
        if f2 is None:
            continue
        for route_id in ("F0_current_scalar_focus_bridge", "F1_scalar_component_surrogate"):
            result = by_key.get((route_id, str(control_id)))
            if result is None:
                continue
            key = f"{route_id}_minus_F2:{control_id}"
            raw_difference_maps[key] = np.asarray(result.intensity_stack - f2.intensity_stack, dtype=np.float32)
            equal_power_difference = _equal_power_stack(result.intensity_stack) - _equal_power_stack(f2.intensity_stack)
            difference_maps[key] = np.asarray(equal_power_difference, dtype=np.float32)
            comparisons[f"{key}:raw_intensity_rms_unscaled"] = _normalised_stack_rms(result.intensity_stack, f2.intensity_stack)
            full = _equal_power_shape_metrics(result.intensity_stack, f2.intensity_stack, crop_fraction=1.0)
            crop = _equal_power_shape_metrics(result.intensity_stack, f2.intensity_stack, crop_fraction=0.5)
            for metric, value in full.items():
                comparisons[f"{key}:full_field_{metric}"] = value
            for metric, value in crop.items():
                comparisons[f"{key}:central_crop_{metric}"] = value

    return {
        "status": "focus_validation_gate_unresolved_pending_converged_F2",
        "route_results": tuple(results),
        "metric_rows": tuple(_best_metric_row(result) for result in results),
        "power_rows": tuple(_power_row(result) for result in results),
        "comparisons": comparisons,
        "difference_maps": difference_maps,
        "equal_power_difference_maps": difference_maps,
        "raw_difference_maps": raw_difference_maps,
        "sampling_audit_rows": tuple(downstream_sampling_audit_rows(cfg)),
        "multiscale_sampling_rows": tuple(downstream_focus_multiscale_sampling_rows(cfg)),
        "completion_gate": {
            "selected_conclusion": None,
            "allowed_labels": ("A", "B", "C"),
            "C_allowed": bool(paper_resolution_f2_converged),
            "C_rule": "Do not select C unless F2 has been run at declared converged paper-resolution.",
        },
        "model_boundary": (
            "F0 is the current Digital Twin vector axicon plus scalar per-component focus bridge plus vector ASM.",
            "F1 is the deliberately weaker scalar-component surrogate with no p/s Fresnel split, vector projection, or Ez generation.",
            "F2 is a scoped vectorial pupil-spectrum reference for this Nathan branch, not a global replacement of ObjectiveMap.",
            "Raw F0-F2 intensity RMS is uncalibrated; use equal-power shape metrics before interpreting focus-physics differences.",
            FOCUS_VALIDATION_UNRESOLVED_STATEMENT,
        ),
    }


def build_default_routes(
    config: NathanHexagonConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
) -> tuple[RouteFieldResult, ...]:
    """Return the default route set used by notebooks and tests."""

    cfg = config or NathanHexagonConfig.fast()
    grid_dict = dict(default_nathan_grid(cfg) if grid is None else grid)
    return (
        RouteFieldResult(
            route_id="canonical_target",
            field=canonical_target_field(cfg, grid=grid_dict),
            target=canonical_target_field(cfg, grid=grid_dict),
            comparison=compare_vector_fields(canonical_target_field(cfg, grid=grid_dict), canonical_target_field(cfg, grid=grid_dict)),
            metadata={"route_family": "analytic_target", "model_status": MODEL_STATUS},
        ),
        run_patterned_hwp_route(cfg, grid=grid_dict, hwp=PatternedHWPConfig(case="continuous"), route_id="patterned_hwp_continuous"),
        run_patterned_hwp_route(cfg, grid=grid_dict, hwp=PatternedHWPConfig(case="six_wedges"), route_id="patterned_hwp_six_wedges"),
        run_patterned_hwp_route(
            cfg,
            grid=grid_dict,
            hwp=PatternedHWPConfig(case="mosaic", tiles_per_sector=8),
            route_id="patterned_hwp_mosaic_8",
        ),
        run_serial_slm_route(cfg, grid=grid_dict, case="ideal", route_id="serial_slm_ideal"),
        run_serial_slm_route(
            replace(cfg, vector=replace(cfg.vector, ideal_components=False)),
            grid=grid_dict,
            case="panel_realistic",
            route_id="serial_slm_panel_realistic",
        ),
    )


def build_route_comparison_report(
    config: NathanHexagonConfig | None = None,
    *,
    twin_config: TwinConfig | None = None,
    include_axicon: bool = True,
) -> RouteComparisonReport:
    """Build a raw, exploratory route-equivalence report."""

    cfg = config or NathanHexagonConfig.fast()
    twin = _twin_with_axial_points(twin_config or cfg.twin, cfg.z_planes)
    z_values = air_z_values(twin, planes=cfg.z_planes, span_factor=cfg.z_span_factor)
    routes = build_default_routes(cfg)
    input_rows = tuple(_input_row(route) for route in routes)
    output_rows: list[dict[str, Any]] = []
    controls: dict[str, Any] = {}
    if include_axicon:
        canonical_prop = propagate_route_through_shared_axicon(routes[0], twin, z_values_m=z_values)
        for route in routes:
            prop = propagate_route_through_shared_axicon(route, twin, z_values_m=z_values, canonical_axicon=canonical_prop.axicon)
            output_rows.append(_output_row(prop))
        controls = build_control_suite(cfg, twin, z_values_m=z_values)
    return RouteComparisonReport(
        config=cfg,
        input_rows=input_rows,
        output_rows=tuple(output_rows),
        controls=controls,
        assumptions=(
            "Exploratory route comparison: no material response, no camera model, and no Nathan free-space millimetre geometry.",
            "Vector generator uses Nathan's sector convention at the existing Digital Twin upstream handoff plane.",
            "Downstream path inherits TwinConfig laser, SLM, axicon, relay, objective, target, grid, and z-reference conventions.",
            "Pass/fail claims use only existing metric gates or explicitly labelled exploratory controls.",
        ),
    )


def build_route_propagations(
    config: NathanHexagonConfig | None = None,
    *,
    twin_config: TwinConfig | None = None,
    route_ids: Sequence[str] | None = None,
) -> tuple[RoutePropagationResult, ...]:
    """Return propagated route objects for visual inspection.

    This is the visual counterpart to :func:`build_route_comparison_report`.
    It keeps the full z-stack so notebooks can show transverse ``xy`` planes
    and longitudinal ``xz`` slices instead of only scalar summary bars.
    """

    cfg = config or NathanHexagonConfig.fast()
    twin = _twin_with_axial_points(twin_config or cfg.twin, cfg.z_planes)
    z_values = air_z_values(twin, planes=cfg.z_planes, span_factor=cfg.z_span_factor)
    routes = build_default_routes(cfg)
    wanted = None if route_ids is None else {str(route_id) for route_id in route_ids}
    canonical_route = next((route for route in routes if route.route_id == "canonical_target"), routes[0])
    canonical_prop = propagate_route_through_shared_axicon(canonical_route, twin, z_values_m=z_values)
    out: list[RoutePropagationResult] = []
    for route in routes:
        if wanted is not None and route.route_id not in wanted:
            continue
        if route.route_id == canonical_prop.route.route_id:
            out.append(canonical_prop)
        else:
            out.append(propagate_route_through_shared_axicon(route, twin, z_values_m=z_values, canonical_axicon=canonical_prop.axicon))
    return tuple(out)


def route_xy_xz_profile_arrays(propagation: RoutePropagationResult) -> dict[str, Any]:
    """Return display-ready ``xy`` and centre-line ``xz`` intensity arrays."""

    metrics = list(propagation.metrics)
    best_idx = int(np.nanargmax([hollow_hexagon_score(metric) for metric in metrics])) if metrics else 0
    stack = np.asarray(propagation.axicon.intensity_stack, dtype=float)
    grid = propagation.axicon.focal_plane.grid
    x_m = np.asarray(grid["x"], dtype=float)
    y_m = np.asarray(grid.get("y", x_m), dtype=float)
    mid_y = int(stack.shape[1] // 2)
    return {
        "route_id": propagation.route.route_id,
        "best_index": best_idx,
        "best_metric": metrics[best_idx] if metrics else None,
        "best_hollow_hexagon_score": hollow_hexagon_score(metrics[best_idx]) if metrics else 0.0,
        "xy_intensity": stack[best_idx],
        "xz_intensity": stack[:, mid_y, :],
        "x_um": x_m / 1e-6,
        "y_um": y_m / 1e-6,
        "z_um": np.asarray(propagation.axicon.z_values_m, dtype=float) / 1e-6,
    }


def plot_route_xy_xz_profiles(
    propagations: Sequence[RoutePropagationResult],
    *,
    output_path: str | Path | None = None,
    title: str = "Nathan vector-hexagon routes: xy profile and xz path",
    common_normalisation: bool = True,
) -> tuple[Any, Any]:
    """Plot route beams as transverse ``xy`` planes and longitudinal ``xz`` paths."""

    import matplotlib.pyplot as plt

    props = tuple(propagations)
    if not props:
        raise ValueError("At least one propagated route is required.")
    arrays = [route_xy_xz_profile_arrays(prop) for prop in props]
    if common_normalisation:
        vmax = max(float(max(np.nanmax(item["xy_intensity"]), np.nanmax(item["xz_intensity"]))) for item in arrays)
        vmax = max(vmax, EPS)
    else:
        vmax = 1.0

    fig_h = max(3.2, 2.55 * len(arrays))
    fig, axes = plt.subplots(len(arrays), 2, figsize=(10.5, fig_h), constrained_layout=True)
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape(1, 2)
    for row, item in enumerate(arrays):
        metric = item["best_metric"]
        xy = np.asarray(item["xy_intensity"], dtype=float)
        xz = np.asarray(item["xz_intensity"], dtype=float)
        local_vmax = vmax if common_normalisation else max(float(max(np.nanmax(xy), np.nanmax(xz))), EPS)
        xy_display = xy / local_vmax
        xz_display = xz / local_vmax
        xy_extent = [
            float(item["x_um"][0]),
            float(item["x_um"][-1]),
            float(item["y_um"][0]),
            float(item["y_um"][-1]),
        ]
        xz_extent = [
            float(item["x_um"][0]),
            float(item["x_um"][-1]),
            float(item["z_um"][0]),
            float(item["z_um"][-1]),
        ]

        ax_xy, ax_xz = axes_arr[row]
        im_xy = ax_xy.imshow(xy_display, origin="lower", extent=xy_extent, cmap="inferno", vmin=0.0, vmax=1.0)
        z_text = "" if metric is None else (
            f"z={metric.z_m / 1e-6:.1f} um, score={item['best_hollow_hexagon_score']:.3f}, "
            f"wall={metric.wall_continuity:.3f}, core={metric.core_darkness:.3f}"
        )
        ax_xy.set_title(f"{item['route_id']} xy\n{z_text}", fontsize=8.5)
        ax_xy.set_xlabel("x (um)")
        ax_xy.set_ylabel("y (um)")
        fig.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.03, label="norm. I")

        im_xz = ax_xz.imshow(xz_display, origin="lower", aspect="auto", extent=xz_extent, cmap="inferno", vmin=0.0, vmax=1.0)
        ax_xz.set_title(f"{item['route_id']} xz centre slice", fontsize=8.5)
        ax_xz.set_xlabel("x (um)")
        ax_xz.set_ylabel("z (um)")
        fig.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.03, label="norm. I")
    fig.suptitle(title, fontsize=11)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220)
    return fig, axes


def _input_row(route: RouteFieldResult) -> dict[str, Any]:
    c = route.comparison
    return {
        "route_id": route.route_id,
        "route_family": route.metadata.get("route_family", ""),
        "complex_overlap": c.complex_overlap,
        "normalized_rms_error": c.normalized_rms_error,
        "normalized_max_error": c.normalized_max_error,
        "stokes_rms_error": c.stokes_rms_error,
        "angle_rms_rad": c.angle_rms_rad,
        "power_ratio": c.power_ratio,
        "model_status": route.metadata.get("model_status", MODEL_STATUS),
    }


def _output_row(prop: RoutePropagationResult) -> dict[str, Any]:
    best = max(prop.metrics, key=hollow_hexagon_score)
    max_h6 = max(prop.metrics, key=lambda item: item.h6)
    passes = [m for m in prop.metrics if m.hexagon_acceptance_pass]
    useful_length = 0.0
    if passes:
        useful_length = float(max(m.z_m for m in passes) - min(m.z_m for m in passes))
    return {
        "route_id": prop.route.route_id,
        "best_z_m": best.z_m,
        "best_hollow_hexagon_score": hollow_hexagon_score(best),
        "best_h6": best.h6,
        "max_h6": max_h6.h6,
        "max_h6_z_m": max_h6.z_m,
        "best_order6_over_order0": best.order6_over_order0,
        "best_order6_over_non_dc": best.order6_over_non_dc,
        "best_wall_continuity": best.wall_continuity,
        "best_core_darkness": best.core_darkness,
        "best_wall_power_fraction": best.wall_power_fraction,
        "best_sidelobe_fraction": best.sidelobe_fraction,
        "useful_axial_length_m": useful_length,
        "mean_output_overlap_to_canonical": float(np.mean(prop.output_overlap_to_canonical)) if prop.output_overlap_to_canonical else 1.0,
        "kr_surface_m_inv": prop.axicon.parameters.k_r_surface_m_inv,
    }


def build_control_suite(
    config: NathanHexagonConfig | None = None,
    twin_config: TwinConfig | None = None,
    *,
    z_values_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Run explicit controls for the hollow-hexagon report."""

    cfg = config or NathanHexagonConfig.fast()
    twin = _twin_with_axial_points(twin_config or cfg.twin, cfg.z_planes)
    grid = default_nathan_grid(cfg)
    controls: dict[str, Any] = {}
    for name in ("radial", "azimuthal", "linear_x"):
        route = RouteFieldResult(
            route_id=f"control_{name}",
            field=canonical_target_field(cfg, grid=grid, control=name),
            target=canonical_target_field(cfg, grid=grid),
            comparison=compare_vector_fields(canonical_target_field(cfg, grid=grid, control=name), canonical_target_field(cfg, grid=grid)),
            metadata={"route_family": "control", "control": name, "model_status": MODEL_STATUS},
        )
        prop = propagate_route_through_shared_axicon(route, twin, z_values_m=z_values_m)
        controls[name] = _output_row(prop)
    controls["wrong_serial_phase"] = _input_row(run_serial_slm_route(cfg, grid=grid, case="ideal", naive_psi2=True, route_id="wrong_serial_phase"))
    controls["wrong_carrier_sign"] = _input_row(
        run_serial_slm_route(
            replace(cfg, vector=replace(cfg.vector, ideal_components=False)),
            grid=grid,
            case="panel_realistic",
            wrong_carrier_sign=True,
            apply_order_filter=False,
            route_id="wrong_carrier_sign",
        )
    )
    controls["six_lobe_lattice"] = lattice_control_report(grid)
    controls["source_convention_validation"] = source_convention_validation_control(cfg)
    return controls


def lattice_control_report(grid: Mapping[str, Any]) -> dict[str, Any]:
    """Return a synthetic six-spot lattice report that must fail wall continuity."""

    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    radius = 0.45 * float(np.max(R))
    sigma = max(float(grid["dx"]) * 1.5, radius / 60.0)
    intensity = np.zeros_like(R, dtype=float)
    for angle in np.linspace(0.0, TWOPI, 6, endpoint=False):
        x0 = radius * np.cos(angle)
        y0 = radius * np.sin(angle)
        intensity += np.exp(-((np.asarray(grid["X"]) - x0) ** 2 + (np.asarray(grid["Y"]) - y0) ** 2) / (2.0 * sigma**2))
    profile = sample_ring_profile(intensity, grid, radius, angular_samples=1440)
    acceptance = hexagon_acceptance(intensity, R, PHI, radius)
    continuity = _wall_continuity_from_profile(profile)
    return {
        "hexagon_acceptance_pass": bool(acceptance["pass"]),
        "wall_continuity": continuity,
        "wall_continuity_pass": bool(continuity >= DEFAULT_WALL_CONTINUITY_MIN_FRACTION),
        "order6_over_order0": float(acceptance["order6_over_order0"]),
        "control_status": "must_fail_hollow_wall_continuity",
    }


def hwp_robustness_sweep(
    config: NathanHexagonConfig | None = None,
    *,
    family: str,
    values: Sequence[float | int],
) -> list[dict[str, Any]]:
    """Numerical sensitivity rows for patterned-HWP controls at the handoff plane."""

    cfg = config or NathanHexagonConfig.fast()
    rows: list[dict[str, Any]] = []
    for value in values:
        kwargs: dict[str, Any] = {"case": "continuous"}
        if family == "tiles_per_sector":
            kwargs = {"case": "mosaic", "tiles_per_sector": int(value)}
        elif family == "seam_width_rad":
            kwargs["seam_width_rad"] = float(value)
        elif family == "central_defect_radius_m":
            kwargs["central_defect_radius_m"] = float(value)
        elif family == "fast_axis_error_rad":
            kwargs["fast_axis_error_rad"] = float(value)
        elif family == "retardance_error_rad":
            kwargs["retardance_error_rad"] = float(value)
        elif family == "aperture_radius_m":
            kwargs["aperture_radius_m"] = float(value)
        else:
            raise ValueError(f"Unsupported HWP sweep family: {family!r}")
        route = run_patterned_hwp_route(cfg, hwp=PatternedHWPConfig(**kwargs))
        rows.append({"family": family, "value": float(value), **_input_row(route)})
    return rows


def serial_slm_robustness_sweep(
    config: NathanHexagonConfig | None = None,
    *,
    family: str,
    values: Sequence[float | int],
) -> list[dict[str, Any]]:
    """Numerical sensitivity rows for serial-SLM controls at the handoff plane."""

    cfg = config or NathanHexagonConfig.fast()
    rows: list[dict[str, Any]] = []
    for value in values:
        local = cfg
        case = "panel_realistic"
        kwargs: dict[str, Any] = {}
        if family == "piston_delta_rad":
            kwargs["piston_delta_rad"] = float(value)
        elif family == "quantisation_levels":
            v = replace(cfg.vector, slm1=replace(cfg.vector.slm1, phase_levels=int(value)), slm2=replace(cfg.vector.slm2, phase_levels=int(value)))
            local = replace(cfg, vector=replace(v, ideal_components=False))
        elif family == "fill_factor":
            v = replace(cfg.vector, slm1=replace(cfg.vector.slm1, fill_factor=float(value)), slm2=replace(cfg.vector.slm2, fill_factor=float(value)))
            local = replace(cfg, vector=replace(v, ideal_components=False))
        elif family == "waveplate_retardance_error_rad":
            v = replace(cfg.vector, hwp_retardance_error_rad=float(value), qwp_retardance_error_rad=float(value))
            local = replace(cfg, vector=replace(v, ideal_components=False))
        elif family == "carrier_sign":
            kwargs["wrong_carrier_sign"] = bool(float(value) < 0)
        else:
            rows.append({
                "family": family,
                "value": float(value),
                "status": "not_supported_by_current_geometry",
                "reason": "registration shift, magnification mismatch, phase-scale/LUT maps, and carrier/iris physical mismatch need measured component geometry.",
            })
            continue
        route = run_serial_slm_route(replace(local, vector=replace(local.vector, ideal_components=False)), case=case, **kwargs)
        rows.append({"family": family, "value": float(value), **_input_row(route)})
    return rows


def write_report_csv(report: RouteComparisonReport, output_dir: str | Path) -> dict[str, Path]:
    """Write route-comparison CSVs into the digital-twin output tree."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "input": out / "nathan_vector_hexagon_handoff_routes.csv",
        "output": out / "nathan_vector_hexagon_axicon_output_routes.csv",
        "controls": out / "nathan_vector_hexagon_controls.csv",
    }
    _write_rows(paths["input"], report.input_rows)
    _write_rows(paths["output"], report.output_rows)
    control_rows = []
    for key, value in report.controls.items():
        row = {"control_id": key}
        if isinstance(value, Mapping):
            row.update(value)
        else:
            row["value"] = repr(value)
        control_rows.append(row)
    _write_rows(paths["controls"], control_rows)
    return paths


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def route_conclusion(report: RouteComparisonReport) -> str:
    """Return a short exploratory conclusion from raw route rows."""

    if not report.output_rows:
        return "Propagation was not run; only vector-generator handoff route agreement is available."
    ranked = sorted(report.output_rows, key=lambda row: float(row.get("best_hollow_hexagon_score", 0.0)), reverse=True)
    best = ranked[0]
    input_by_id = {row["route_id"]: row for row in report.input_rows}
    best_input = input_by_id.get(best["route_id"], {})
    overlap = float(best_input.get("complex_overlap", 0.0))
    if overlap > 0.99:
        return (
            f"{best['route_id']} is closest to the canonical target and has the strongest exploratory hollow-hexagon "
            "score in this run. This is not a final claim without the declared resolution/convergence gates."
        )
    return (
        f"{best['route_id']} has the strongest exploratory hollow-hexagon score, but its handoff-plane overlap "
        f"({overlap:.3f}) is not target-equivalent under the current model."
    )


# ===========================================================================
# MODE 1 — ideal canonical P2 field through the inherited downstream Digital Twin
# ===========================================================================
#
# MODE 1 injects the ideal canonical Nathan six-sector VectorField at the common
# P2 handoff plane and propagates it through the *inherited* downstream Digital
# Twin geometry (vector axicon -> current F0 scalar-per-component focus bridge ->
# vector ASM sample z-stack), with the F2 vectorial pupil-spectrum reference as a
# diagnostic.  It bypasses ALL physical HWP/QWP/SLM generation and panel realism.
# It is not a simulation of the physical bench; it tests whether the source-
# validated (V0, docs/53) sixfold / dark-core hexagonal Bessel field survives the
# inherited micro-scale optics.  MODE 1 metrics deliberately separate
# "source-like hexagonal Bessel survival" from "clean single-wall usefulness".

MODE1_STAGE = "nathan_mode1_ideal_p2_downstream"
MODE1_ALLOWED_OUTCOMES = ("M1-A", "M1-B", "M1-C", "M1-D")
MODE1_DARK_CORE_HOLLOW_THRESHOLD = 0.60
MODE1_SIXFOLD_PRESENT_ORDER6_NON_DC = 0.10


def _mode1_git_commit_short() -> str:
    import subprocess

    try:
        root = str(Path(__file__).resolve().parents[3])
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _mode1_grid(n: int, dx: float, *, straddle: bool) -> dict[str, Any]:
    """Return a square grid dict; ``straddle`` selects the project (zero-straddling)
    convention, otherwise Nathan's axis-sampled convention (samples x=0)."""

    n = int(n)
    dx = float(dx)
    offset = 0.5 if straddle else 0.0
    x = (np.arange(n, dtype=float) - n // 2 + offset) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    return {
        "N": n, "dx": dx, "x": x, "y": x, "X": X, "Y": Y,
        "R": np.hypot(X, Y), "PHI": np.arctan2(Y, X), "FX": FX, "FY": FY,
    }


def _mode1_central_defect_field(field: VectorField, radius_m: float) -> VectorField:
    """Zero the field inside ``radius_m`` (a labelled central regularisation)."""

    R = np.asarray(field.grid["R"], dtype=float)
    mask = (R > float(radius_m)).astype(float)
    return VectorField(
        ex=field.ex * mask, ey=field.ey * mask, ez=field.ez * mask,
        grid=field.grid, wavelength_m=field.wavelength_m, medium_index=field.medium_index,
        metadata={**dict(field.metadata), "central_defect_radius_m": float(radius_m)},
    )


def _mode1_multi_ring_count(plane: np.ndarray, grid: Mapping[str, Any], cx: float = 0.0, cy: float = 0.0) -> int:
    """Count bright radial rings in the azimuthally-averaged radial profile."""

    arr = np.maximum(np.asarray(plane, dtype=float), 0.0)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    dx = float(grid["dx"])
    r = np.hypot(X - cx, Y - cy)
    rmax = 0.5 * min(float(X.max() - X.min()), float(Y.max() - Y.min()))
    nb = max(8, int(rmax / max(dx, EPS)))
    idx = np.clip((r / max(dx, EPS)).astype(int), 0, nb - 1)
    prof = np.bincount(idx.ravel(), weights=arr.ravel(), minlength=nb)
    cnt = np.bincount(idx.ravel(), minlength=nb)
    prof = prof / np.maximum(cnt, 1)
    if prof.size < 3 or prof.max() <= 0:
        return 0
    prof = prof / prof.max()
    peaks = 0
    for i in range(1, prof.size - 1):
        if prof[i] >= 0.12 and prof[i] > prof[i - 1] and prof[i] >= prof[i + 1]:
            peaks += 1
    return int(peaks)


# Three visual symmetry classes.  MODE 2A/2B may only begin from a genuine hexagon.
MODE1_SYMMETRY_CLASSES = ("visual_hexagonal_field", "triangular_lobed_field", "dark_core_structured_field")


def _mode1_ring_island_count(plane: np.ndarray, grid: Mapping[str, Any], ring_r: float) -> int:
    """Count connected bright islands on the dominant ring (3 => triangular/lobed)."""

    from scipy.ndimage import label

    R = np.asarray(grid["R"], dtype=float)
    vals = np.asarray(plane, dtype=float)
    mask = (R >= 0.70 * ring_r) & (R <= 1.30 * ring_r)
    if not np.any(mask):
        return 0
    thr = 0.5 * float(np.nanmax(vals[mask]))
    binary = (vals >= thr) & mask
    labels, _ = label(binary)
    if labels.max() == 0:
        return 0
    counts = np.bincount(labels.ravel())[1:]
    min_size = max(3, int(0.001 * vals.size))
    return int(np.sum(counts >= min_size))


def _mode1_symmetry(plane: np.ndarray, grid: Mapping[str, Any], ring_r: float) -> dict[str, Any]:
    """C3-vs-C6 diagnostics: ring angular orders, rotational self-similarity, six-sector balance.

    A triangular / C3 field can carry a strong order-6 harmonic on the ring, so order-6 content
    alone must NEVER be treated as evidence of a hexagonal beam.  The decisive discriminators are
    60 deg vs 120 deg rotational self-similarity and the six-sector energy balance."""

    from scipy.ndimage import rotate

    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    sf = sixfold_from_intensity(plane, R, PHI, ring_r, angular_bins=720)
    amps = np.abs(np.fft.rfft(np.asarray(sf["profile"], dtype=float)))

    def _amp(k: int) -> float:
        return float(amps[k]) if amps.size > k else 0.0

    o3, o6, o9, o12 = _amp(3), _amp(6), _amp(9), _amp(12)
    non_dc = float(np.sum(amps[1:])) + EPS
    vals = np.asarray(plane, dtype=float)
    n = vals.shape[0]
    c = n // 2
    h = max(4, int(0.30 * n))
    crop = vals[c - h : c + h, c - h : c + h]

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        aa = a.ravel() - a.mean()
        bb = b.ravel() - b.mean()
        return float(aa @ bb / (np.sqrt((aa @ aa) * (bb @ bb)) + EPS))

    rot = {ang: _corr(crop, rotate(crop, ang, reshape=False, order=1, mode="constant", cval=0.0)) for ang in (60, 120, 180)}
    ring_mask = (R >= 0.75 * ring_r) & (R <= 1.25 * ring_r)
    th = (PHI[ring_mask] % (2.0 * np.pi))
    w = vals[ring_mask]
    sect = np.asarray([float(w[(th >= k * np.pi / 3.0) & (th < (k + 1) * np.pi / 3.0)].sum()) for k in range(6)], dtype=float)
    stot = float(sect.sum()) + EPS
    sfrac = sect / stot
    return {
        "order3_amp": o3,
        "order6_amp": o6,
        "order9_amp": o9,
        "order12_amp": o12,
        "order3_over_order6": float(o3 / (o6 + EPS)),
        "order3_over_non_dc": float(o3 / non_dc),
        "order6_over_non_dc": float(o6 / non_dc),
        "rot_corr_60": float(rot[60]),
        "rot_corr_120": float(rot[120]),
        "rot_corr_180": float(rot[180]),
        "c120_minus_c60": float(rot[120] - rot[60]),
        "six_sector_fraction": tuple(float(v) for v in sfrac),
        "six_sector_max_over_min": float(sfrac.max() / (sfrac.min() + EPS)),
        "odd_sector_sum": float(sfrac[1::2].sum()),
        "even_sector_sum": float(sfrac[0::2].sum()),
        "three_pair_imbalance": float(abs(sfrac[1::2].sum() - sfrac[0::2].sum())),
        "ring_island_count": _mode1_ring_island_count(plane, grid, ring_r),
    }


def mode1_symmetry_class(sym: Mapping[str, Any], dark_core_ratio: float) -> str:
    """Classify a plane as visual_hexagonal / triangular_lobed / dark_core_structured.

    Order-6 content alone is NOT sufficient for hexagonal: a genuine hexagon needs the 60 deg
    rotational self-similarity to be at least as strong as the 120 deg one, six-sector energies
    reasonably balanced, and no three-lobe/triangular dominance."""

    hollow = float(dark_core_ratio) <= MODE1_DARK_CORE_HOLLOW_THRESHOLD
    c60 = float(sym["rot_corr_60"])
    c120 = float(sym["rot_corr_120"])
    o3_o6 = float(sym["order3_over_order6"])
    smax = float(sym["six_sector_max_over_min"])
    islands = int(sym["ring_island_count"])
    # The decisive discriminator is 120 deg vs 60 deg rotational self-similarity, calibrated against
    # the validated V0 hexagon (c120 - c60 ~ -0.05, o3/o6 ~ 0.08) and the failure triangle
    # (c120 - c60 ~ +0.13, o3/o6 high).  Ring-island count is unreliable for continuous multi-ring
    # patterns (V0), so it is only a triangular *trigger* (==3), never a hexagonal gate.
    triangular = (c120 - c60) >= 0.08 or o3_o6 >= 1.0 or smax >= 1.8 or islands == 3
    hexagonal = (
        hollow
        and o3_o6 < 0.8
        and c60 >= 0.55
        and (c120 - c60) <= 0.04
        and smax < 1.6
    )
    if hexagonal and not triangular:
        return "visual_hexagonal_field"
    if triangular:
        return "triangular_lobed_field"
    return "dark_core_structured_field"


def _mode1_plane_row(plane: np.ndarray, grid: Mapping[str, Any]) -> dict[str, Any]:
    """Per-plane MODE 1 diagnostics, separating hexagonal-Bessel from wall metrics."""

    diag = _v0_plane_diagnostics(plane, grid)
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    ring_r = float(diag["ring_radius_m"])
    sf = sixfold_from_intensity(plane, R, PHI, ring_r, angular_bins=720)
    prof = np.asarray(sf["profile"], dtype=float)
    fft = np.fft.rfft(prof)
    orientation = float(np.angle(fft[6]) / 6.0) if fft.size > 6 else 0.0
    sym = _mode1_symmetry(plane, grid, ring_r)
    row = {
        # source-like hexagonal Bessel descriptors
        "dark_core_ratio": float(diag["central_core_darkness"]),
        "sixfold_order6_over_non_dc": float(diag["sixfold_order6_over_non_dc"]),
        "sixfold_order6_over_order0": float(diag["sixfold_order6_over_order0"]),
        "sixfold_dominant_order": int(diag["sixfold_dominant_order"]),
        "sixfold_orientation_rad": orientation,
        "ring_radius_um": float(diag["ring_radius_um"]),
        "multi_ring_count": _mode1_multi_ring_count(plane, grid),
        # clean-wall usefulness descriptors (kept SEPARATE)
        "wall_continuity": float(diag["wall_continuity"]),
        "wall_power_fraction": float(diag["wall_power_fraction"]),
    }
    row.update(sym)
    row["symmetry_class"] = mode1_symmetry_class(sym, row["dark_core_ratio"])
    return row


def mode1_hexagonal_bessel_survival_metrics(
    intensity_stack: np.ndarray,
    z_values_m: Sequence[float],
    grid: Mapping[str, Any],
    *,
    reference_index: int = 0,
) -> dict[str, Any]:
    """Separate MODE 1 metrics: (A) source-like hexagonal Bessel survival vs
    (B) clean single-wall usefulness.  No single 'best plane' is selected."""

    stack = np.asarray(intensity_stack, dtype=float)
    rows = [_mode1_plane_row(stack[i], grid) for i in range(stack.shape[0])]
    ref = int(np.clip(reference_index, 0, len(rows) - 1))

    def _col(key: str) -> np.ndarray:
        return np.asarray([r[key] for r in rows], dtype=float)

    dark = _col("dark_core_ratio")
    o6 = _col("sixfold_order6_over_non_dc")
    dom = np.asarray([r["sixfold_dominant_order"] for r in rows], dtype=int)
    ring = _col("ring_radius_um")
    orient = _col("sixfold_orientation_rad")
    multi = _col("multi_ring_count")

    sixfold_plane = (np.isin(dom, (6, 12, 18))) | (o6 >= MODE1_SIXFOLD_PRESENT_ORDER6_NON_DC)
    hollow_plane = dark <= MODE1_DARK_CORE_HOLLOW_THRESHOLD
    survival_plane = sixfold_plane & hollow_plane
    ring_valid = ring > 0
    ring_stability = float(1.0 - (np.std(ring[ring_valid]) / max(np.mean(ring[ring_valid]), EPS))) if np.any(ring_valid) else 0.0
    orientation_resultant = float(np.abs(np.mean(np.exp(1j * 6.0 * orient)))) if orient.size else 0.0

    source_like = {
        "reference_dark_core_ratio": float(dark[ref]),
        "median_dark_core_ratio": float(np.median(dark)),
        "median_sixfold_order6_over_non_dc": float(np.median(o6)),
        "reference_dominant_order": int(dom[ref]),
        "fraction_planes_sixfold_present": float(np.mean(sixfold_plane)),
        "fraction_planes_hollow": float(np.mean(hollow_plane)),
        "dark_core_axial_persistence": float(np.mean(survival_plane)),
        "ring_radius_stability": float(np.clip(ring_stability, 0.0, 1.0)),
        "sixfold_orientation_stability": orientation_resultant,
        "median_multi_ring_count": float(np.median(multi)),
        "reference_ring_radius_um": float(ring[ref]),
    }
    clean_wall_usefulness = {
        "median_wall_continuity": float(np.median(_col("wall_continuity"))),
        "median_wall_power_fraction": float(np.median(_col("wall_power_fraction"))),
        "reference_wall_continuity": float(rows[ref]["wall_continuity"]),
        "note": "usefulness-only metric; a source-like dark-core hexagonal Bessel need NOT be a clean single wall",
    }
    classes = [str(r["symmetry_class"]) for r in rows]
    symmetry_classification = {
        "reference_symmetry_class": classes[ref],
        "per_plane_classes": tuple(classes),
        "fraction_planes_visual_hexagonal": float(np.mean([c == "visual_hexagonal_field" for c in classes])),
        "fraction_planes_triangular_lobed": float(np.mean([c == "triangular_lobed_field" for c in classes])),
        "fraction_planes_dark_core_structured": float(np.mean([c == "dark_core_structured_field" for c in classes])),
        "reference_order3_over_order6": float(rows[ref]["order3_over_order6"]),
        "reference_rot_corr_60": float(rows[ref]["rot_corr_60"]),
        "reference_rot_corr_120": float(rows[ref]["rot_corr_120"]),
        "reference_c120_minus_c60": float(rows[ref]["c120_minus_c60"]),
        "reference_six_sector_max_over_min": float(rows[ref]["six_sector_max_over_min"]),
        "reference_three_pair_imbalance": float(rows[ref]["three_pair_imbalance"]),
        "reference_ring_island_count": int(rows[ref]["ring_island_count"]),
        "note": "order-6 content alone does NOT imply a hexagon; a C3/triangular field carries a strong order-6 ring harmonic",
    }
    return {
        "reference_index": ref,
        "reference_z_m": float(np.asarray(z_values_m, dtype=float)[ref]),
        "per_plane": tuple(rows),
        "source_like_hexagonal_bessel_survival": source_like,
        "clean_single_wall_usefulness": clean_wall_usefulness,
        "symmetry_classification": symmetry_classification,
        "criteria": {
            "hollow_dark_core_ratio_max": MODE1_DARK_CORE_HOLLOW_THRESHOLD,
            "sixfold_order6_over_non_dc_min": MODE1_SIXFOLD_PRESENT_ORDER6_NON_DC,
        },
    }


def mode1_completion_gate(
    f0_survival: Mapping[str, Any],
    *,
    centre_treatment: Mapping[str, Any] | None = None,
    f0_vs_f2: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest a MODE 1 outcome M1-A/B/C/D from separated survival metrics.

    This is a *suggestion* for the operator, not an auto-applied verdict."""

    s = dict(f0_survival["source_like_hexagonal_bessel_survival"])
    sym = dict(f0_survival.get("symmetry_classification", {}))
    ref_class = str(sym.get("reference_symmetry_class", "dark_core_structured_field"))
    frac_hex = float(sym.get("fraction_planes_visual_hexagonal", 0.0))
    frac_tri = float(sym.get("fraction_planes_triangular_lobed", 0.0))

    hollow = s["median_dark_core_ratio"] <= MODE1_DARK_CORE_HOLLOW_THRESHOLD
    structured = hollow and (
        s["fraction_planes_sixfold_present"] >= 0.25 or int(s["reference_dominant_order"]) in (3, 6, 9, 12)
    )
    # A GENUINE visual hexagon requires the symmetry classifier to pass (order-6 content alone is
    # not enough; a C3/triangular field is explicitly vetoed).  MODE 2A/2B may only begin from that.
    visual_hexagon = ref_class == "visual_hexagonal_field" and frac_hex >= 0.5
    triangular = ref_class == "triangular_lobed_field" or frac_tri >= 0.5
    centre_robust = True if centre_treatment is None else bool(centre_treatment.get("sample_result_robust_to_centre_treatment", True))
    f2_consistent = True
    if f0_vs_f2 is not None:
        f2_consistent = float(f0_vs_f2.get("reference_full_field_correlation", 1.0)) >= 0.5

    reasons: list[str] = []
    if not centre_robust:
        suggested = "M1-C"
        reasons.append("sample-plane result changes qualitatively with the P2 grid centre treatment (possible numerical artefact)")
    elif not f2_consistent:
        suggested = "M1-C"
        reasons.append("F0 and F2 disagree strongly at the reference plane; focus-model discrepancy unresolved")
    elif visual_hexagon:
        suggested = "M1-A"
        reasons.append("a genuine visual hexagonal beam survives (60 deg self-similarity, balanced six-sector energy, no C3/triangular dominance)")
    elif triangular:
        suggested = "M1-B"
        reasons.append("the ideal field survives only as a dark-core TRIANGULAR / three-lobed (C3) structure, not a visual hexagon; MODE 2A/2B remain blocked")
    elif structured:
        suggested = "M1-B"
        reasons.append("a dark-core structured field survives but is not a visual hexagon; MODE 2A/2B remain blocked")
    else:
        suggested = "M1-D"
        reasons.append("no recognisable dark-core hexagonal Bessel structure survives")

    outcome_statement = {
        "M1-A": (
            "The ideal Nathan six-sector P2 field survives the inherited downstream geometry as a genuine "
            "visual hexagonal beam. MODE 2A/2B physical realisation may begin."
        ),
        "M1-B": (
            "The ideal Nathan six-sector P2 field does not produce a visually convincing micro-scale hexagonal "
            "beam under the inherited downstream Digital Twin geometry. The output is dominated by a dark-core "
            "triangular / three-lobed (C3) structure. Physical HWP/QWP/SLM realisation remains paused."
        ),
        "M1-C": (
            "The MODE 1 result is inconclusive (numerical centre-treatment or F0/F2 discrepancy). Do not proceed "
            "to physical realisation."
        ),
        "M1-D": (
            "The ideal P2 field is essentially lost by the inherited downstream system. Do not proceed to "
            "HWP/SLM realisation without changing the downstream optical design objective."
        ),
    }[suggested]

    return {
        "suggested_outcome": suggested,
        "outcome_statement": outcome_statement,
        "allowed_outcomes": MODE1_ALLOWED_OUTCOMES,
        "reference_symmetry_class": ref_class,
        "fraction_planes_visual_hexagonal": frac_hex,
        "fraction_planes_triangular_lobed": frac_tri,
        "visual_hexagon_survives": bool(visual_hexagon),
        "triangular_lobed": bool(triangular),
        "dark_core_structured_survives": bool(structured),
        "centre_treatment_robust": bool(centre_robust),
        "f0_f2_consistent": bool(f2_consistent),
        "mode2_realisation_allowed": bool(visual_hexagon),
        "reasons": tuple(reasons),
        "note": "operator must confirm the final M1-A/B/C/D outcome; MODE 2A/2B only begin from a visual_hexagonal_field",
    }


@dataclass(frozen=True)
class Mode1Result:
    """One MODE 1 run: ideal P2 field through the inherited downstream twin."""

    config: NathanHexagonConfig
    twin_preset: str
    z_values_m: np.ndarray
    reference_index: int
    output_grid: Mapping[str, Any]
    p2_diagnostics: Mapping[str, Any]
    f0: DownstreamRouteResult
    f2: DownstreamRouteResult | None
    f0_survival: Mapping[str, Any]
    f2_survival: Mapping[str, Any] | None
    centre_treatment: Mapping[str, Any]
    f0_vs_f2: Mapping[str, Any] | None
    completion: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _mode1_focal_grid(result: DownstreamRouteResult) -> dict[str, Any]:
    return make_xy_grid(int(result.metadata["output_grid_N"]), float(result.metadata["output_grid_dx_m"]))


def mode1_centre_treatment_diagnostic(
    config: NathanHexagonConfig,
    twin: TwinConfig,
    z_values_m: Sequence[float],
    *,
    reference_index: int = 0,
    defect_radius_factor: float = 0.03,
) -> dict[str, Any]:
    """Check that the sample-region result is not a numerical artefact of how the
    r=0 polarisation singularity is sampled at P2 (the V0/docs-53 lesson).

    Builds the canonical P2 field on three centre treatments and runs F0 for each:
      A) existing project grid (zero-straddling, unchanged);
      B) axis-sampled diagnostic grid (Nathan convention);
      C) existing grid + small labelled central regularisation/defect.
    """

    base_grid = default_nathan_grid(config)
    n = int(base_grid["N"])
    dx = float(base_grid["dx"])
    z_values = np.asarray(z_values_m, dtype=float)
    ref = int(np.clip(reference_index, 0, z_values.size - 1))

    treatments: dict[str, VectorField] = {
        "A_project_grid_straddling": canonical_target_field(config, grid=_mode1_grid(n, dx, straddle=True)),
        "B_axis_sampled_grid": canonical_target_field(config, grid=_mode1_grid(n, dx, straddle=False)),
    }
    defect_field = _mode1_central_defect_field(
        canonical_target_field(config, grid=_mode1_grid(n, dx, straddle=True)),
        float(defect_radius_factor) * float(config.vector.waist_m),
    )
    treatments["C_project_grid_central_regularised"] = defect_field

    rows: dict[str, Any] = {}
    for name, field in treatments.items():
        f0 = _vector_downstream_result(
            field, twin, z_values,
            control_id="nathan_six_sector", route_id="F0_current_scalar_focus_bridge",
            route_role="mode1_centre_treatment",
        )
        focal_grid = _mode1_focal_grid(f0)
        row = _mode1_plane_row(np.asarray(f0.intensity_stack[ref], dtype=float), focal_grid)
        p2_int = np.asarray(field.intensity, dtype=float)
        p2_centre = tuple(s // 2 for s in p2_int.shape)
        rows[name] = {
            "sample_dark_core_ratio": row["dark_core_ratio"],
            "sample_sixfold_order6_over_non_dc": row["sixfold_order6_over_non_dc"],
            "sample_dominant_order": row["sixfold_dominant_order"],
            "sample_ring_radius_um": row["ring_radius_um"],
            "p2_on_axis_over_peak_intensity": float(p2_int[p2_centre] / max(float(np.max(p2_int)), EPS)),
        }

    dark_a = rows["A_project_grid_straddling"]["sample_dark_core_ratio"]
    dark_b = rows["B_axis_sampled_grid"]["sample_dark_core_ratio"]
    # "robust" = A and B land in the same hollow/bright regime and agree numerically
    same_regime = (dark_a <= MODE1_DARK_CORE_HOLLOW_THRESHOLD) == (dark_b <= MODE1_DARK_CORE_HOLLOW_THRESHOLD)
    close = abs(dark_a - dark_b) <= 0.25
    robust = bool(same_regime and close)
    return {
        "treatments": rows,
        "sample_dark_core_ratio_A_minus_B": float(dark_a - dark_b),
        "sample_result_robust_to_centre_treatment": robust,
        "interpretation": (
            "robust: sample-region dark-core/sixfold does not depend on the P2 grid centre treatment"
            if robust else
            "NOT robust: sample-region result changes with the P2 grid centre treatment; treat as a numerical-artefact risk"
        ),
        "note": "P2 total intensity is Gaussian for all sector fields; the sector structure lives in polarisation, so the centre treatment only affects the propagated sample plane.",
    }


def run_mode1_ideal_p2_downstream(
    config: NathanHexagonConfig | None = None,
    *,
    twin_config: TwinConfig | None = None,
    run_f2: bool = True,
    f2_solver: str = "fft",
    f2_chunk_size: int = 256,
    reference_z_index: int | None = None,
    run_centre_treatment: bool = True,
    output_dir: str | Path | None = None,
) -> Mode1Result:
    """Run MODE 1: inject the ideal canonical Nathan six-sector P2 field into the
    inherited downstream Digital Twin geometry and report F0 (+F2) survival.

    Only the ideal P2 field is simulated; patterned-HWP / SLM1 / relay /
    intermediate-HWP / SLM2 / final-QWP generation and panel realism are bypassed.
    """

    cfg = config or NathanHexagonConfig.fast()
    twin = _twin_with_axial_points(twin_config or cfg.twin, cfg.z_planes)
    z_values = air_z_values(twin, planes=cfg.z_planes, span_factor=cfg.z_span_factor)
    # Declared reference plane = middle of the non-diffracting Bessel zone (NOT the
    # z=0 surface where the beam is still forming, and NOT a metric-led best plane).
    ref_default = int(z_values.size // 2)
    ref = ref_default if reference_z_index is None else int(np.clip(reference_z_index, 0, z_values.size - 1))

    grid = default_nathan_grid(cfg)
    field = canonical_target_field(cfg, grid=grid)
    p2 = canonical_target_diagnostics(cfg, grid=grid)

    f0 = _vector_downstream_result(
        field, twin, z_values,
        control_id="nathan_six_sector", route_id="F0_current_scalar_focus_bridge",
        route_role="mode1_ideal_p2_current_downstream_bridge",
    )
    focal_grid = _mode1_focal_grid(f0)
    f0_survival = mode1_hexagonal_bessel_survival_metrics(f0.intensity_stack, z_values, focal_grid, reference_index=ref)

    f2: DownstreamRouteResult | None = None
    f2_survival: Mapping[str, Any] | None = None
    f0_vs_f2: dict[str, Any] | None = None
    if run_f2:
        f2 = _vectorial_pupil_spectrum_reference_result(
            field, twin, z_values, control_id="nathan_six_sector", solver=f2_solver, chunk_size=f2_chunk_size,
        )
        f2_focal_grid = _mode1_focal_grid(f2)
        f2_survival = mode1_hexagonal_bessel_survival_metrics(f2.intensity_stack, z_values, f2_focal_grid, reference_index=ref)
        full = _equal_power_shape_metrics(f0.intensity_stack[ref][None, ...], f2.intensity_stack[ref][None, ...], crop_fraction=1.0)
        crop = _equal_power_shape_metrics(f0.intensity_stack[ref][None, ...], f2.intensity_stack[ref][None, ...], crop_fraction=0.5)
        f0_vs_f2 = {
            "reference_full_field_correlation": float(full["equal_power_intensity_correlation"]),
            "reference_full_field_shape_rms": float(full["equal_power_shape_rms"]),
            "reference_central_crop_correlation": float(crop["equal_power_intensity_correlation"]),
            "reference_central_crop_shape_rms": float(crop["equal_power_shape_rms"]),
            "f0_mean_ez_fraction": float(np.mean(f0.ez_energy_fraction)),
            "f2_mean_ez_fraction": float(np.mean(f2.ez_energy_fraction)),
        }

    centre_treatment: dict[str, Any] = {"status": "not_run"}
    if run_centre_treatment:
        centre_treatment = mode1_centre_treatment_diagnostic(cfg, twin, z_values, reference_index=ref)

    completion = mode1_completion_gate(f0_survival, centre_treatment=centre_treatment if run_centre_treatment else None, f0_vs_f2=f0_vs_f2)
    manifest = mode1_scope_manifest(cfg, twin, f0, f0_survival, centre_treatment, f0_vs_f2, completion, run_f2=run_f2)

    result = Mode1Result(
        config=cfg, twin_preset=str(cfg.baseline_preset), z_values_m=z_values, reference_index=ref,
        output_grid=focal_grid, p2_diagnostics=p2, f0=f0, f2=f2, f0_survival=f0_survival,
        f2_survival=f2_survival, centre_treatment=centre_treatment, f0_vs_f2=f0_vs_f2,
        completion=completion, manifest=manifest,
    )
    if output_dir is not None:
        write_mode1_scope_manifest(result, output_dir)
    return result


def mode1_scope_manifest(
    config: NathanHexagonConfig,
    twin: TwinConfig,
    f0: DownstreamRouteResult,
    f0_survival: Mapping[str, Any],
    centre_treatment: Mapping[str, Any],
    f0_vs_f2: Mapping[str, Any] | None,
    completion: Mapping[str, Any],
    *,
    run_f2: bool,
) -> dict[str, Any]:
    """Machine-readable MODE 1 simulation-scope manifest."""

    import datetime as _dt

    return {
        "mode": "MODE 1 ideal P2 downstream Digital Twin",
        "stage": MODE1_STAGE,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _mode1_git_commit_short(),
        "inherited_twinconfig_preset": str(config.baseline_preset),
        "grid_n": int(config.grid_n),
        "z_planes": int(config.z_planes),
        "input_plane": "P2 vector-generator output handoff (ideal canonical six-sector VectorField)",
        "output_plane": "inherited micro-scale sample/focal reference plane and adjacent z-stack",
        "simulated_components": (
            "ideal prescribed six-sector radial/azimuthal VectorField at P2",
            "inherited Digital Twin laser wavelength and Gaussian envelope",
            "inherited vector axicon (thin, p/s Fresnel)",
            "inherited scalar per-component ObjectiveMap/focus bridge (F0)",
            "vector ASM sample z-stack with Ez",
            "F2 vectorial pupil-spectrum reference (diagnostic only)",
        ),
        "bypassed_components": (
            "patterned HWP", "SLM1", "serial relay", "intermediate HWP", "SLM2",
            "final QWP", "panel realism", "carrier/iris realism", "HWP mosaics", "waveplate errors",
        ),
        "approximated_components": (
            "F0 focus = scalar FFT focus applied independently to Ex and Ey (not full Richards-Wolf)",
            "F2 = scoped vectorial pupil-spectrum reference, not a calibrated Richards-Wolf equivalence",
            "thin vector axicon (no thick-axicon walk-off)",
        ),
        "downstream_solver_routes": {
            "F0": "current Digital Twin bridge: vector axicon -> scalar per-component focus -> vector ASM",
            "F2": "vectorial pupil-spectrum diagnostic reference" if run_f2 else "not run in this manifest",
        },
        "represents_physical_bench": "partially, downstream only; upstream physical generation bypassed",
        "statement": (
            "MODE 1 tests the ideal target field in the inherited downstream system. "
            "It does not yet simulate HWP/QWP/SLM generation."
        ),
        "output_grid_N": int(f0.metadata["output_grid_N"]),
        "output_grid_dx_m": float(f0.metadata["output_grid_dx_m"]),
        "source_like_hexagonal_bessel_survival": dict(f0_survival["source_like_hexagonal_bessel_survival"]),
        "symmetry_classification": dict(f0_survival.get("symmetry_classification", {})),
        "clean_single_wall_usefulness": dict(f0_survival["clean_single_wall_usefulness"]),
        "centre_treatment_robust": bool(centre_treatment.get("sample_result_robust_to_centre_treatment", True)),
        "f0_vs_f2": dict(f0_vs_f2) if f0_vs_f2 is not None else None,
        "outcome": completion.get("suggested_outcome"),
        "outcome_statement": completion.get("outcome_statement"),
        "mode2_realisation_allowed": completion.get("mode2_realisation_allowed", False),
        "completion_gate": dict(completion),
        "claim_boundary": {
            "model_status": MODEL_STATUS,
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "material_model": False,
            "camera_model": False,
            "physical_generation_modelled": False,
        },
    }


def write_mode1_scope_manifest(result: Mode1Result, output_dir: str | Path) -> Path:
    """Write the MODE 1 scope manifest JSON to ``output_dir``."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "simulation_scope_manifest.json"
    path.write_text(json.dumps(_json_ready(dict(result.manifest)), indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# MODE 1 figures (package functions; notebooks call these, no duplicated physics)
# ---------------------------------------------------------------------------


def _mode1_um_axis(grid: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float) / 1e-6


def _save_fig(fig: Any, output_path: str | Path | None) -> None:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170)


def plot_mode1_p2_input_diagnostics(result: Mode1Result, *, output_path: str | Path | None = None) -> tuple[Any, Any]:
    """P2 ideal-field diagnostics: sector mask, alpha, Ex/Ey amp/phase, Stokes, intensity, centre treatment."""

    import matplotlib.pyplot as plt

    p2 = result.p2_diagnostics
    grid = p2["field"].grid
    x_um = _mode1_um_axis(grid)
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    stokes = p2["stokes"]
    fig, axes = plt.subplots(3, 4, figsize=(16.0, 11.5), constrained_layout=True)
    panels = [
        (p2["sector_mask"], "P2 sector mask (0 radial/1 azimuthal)", "viridis", None),
        (p2["alpha_rad"], "P2 alpha(theta) [rad]", "twilight", None),
        (p2["Ex_amplitude"], "|Ex|", "inferno", None),
        (p2["Ey_amplitude"], "|Ey|", "inferno", None),
        (p2["Ex_phase_rad"], "arg Ex [rad]", "twilight", None),
        (p2["Ey_phase_rad"], "arg Ey [rad]", "twilight", None),
        (stokes["S0"], "Stokes S0", "inferno", None),
        (stokes["S1"], "Stokes S1", "coolwarm", "sym"),
        (stokes["S2"], "Stokes S2", "coolwarm", "sym"),
        (stokes["S3"], "Stokes S3", "coolwarm", "sym"),
        (p2["intensity"], "P2 total intensity |E|^2 (Gaussian)", "inferno", None),
    ]
    for ax, (arr, title, cmap, mode) in zip(axes.ravel()[:11], panels, strict=False):
        a = np.asarray(arr, dtype=float)
        if mode == "sym":
            v = max(float(np.nanmax(np.abs(a))), EPS)
            im = ax.imshow(a, origin="lower", extent=ext, cmap=cmap, vmin=-v, vmax=v)
        else:
            im = ax.imshow(a, origin="lower", extent=ext, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ct = result.centre_treatment
    ax = axes.ravel()[11]
    ax.set_axis_off()
    lines = ["Centre-treatment sensitivity (sample plane)", ""]
    for name, row in ct["treatments"].items():
        lines.append(f"{name}:")
        lines.append(f"  sample dark-core/peak = {row['sample_dark_core_ratio']:.4f}")
        lines.append(f"  sample dominant order = {row['sample_dominant_order']}")
    lines.append("")
    lines.append(f"robust to centre treatment: {ct['sample_result_robust_to_centre_treatment']}")
    lines.append("P2 intensity is Gaussian; sector info is in polarisation.")
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace")
    fig.suptitle("MODE 1 - ideal canonical Nathan six-sector VectorField at P2 (bypasses HWP/QWP/SLM generation)", fontsize=12)
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1_sample_region(result: Mode1Result, route: str = "F0", *, output_path: str | Path | None = None) -> tuple[Any, Any]:
    """Sample-region output for one downstream route (F0 or F2): xy, crop, z-stack, xz, yz, profiles, on-axis vs z."""

    import matplotlib.pyplot as plt

    rr = result.f0 if route.upper() == "F0" else result.f2
    if rr is None:
        raise ValueError("F2 route was not run for this MODE 1 result.")
    grid = _mode1_focal_grid(rr)
    stack = np.asarray(rr.intensity_stack, dtype=float)
    z_um = np.asarray(result.z_values_m, dtype=float) / 1e-6
    x_um = _mode1_um_axis(grid)
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    ref = int(result.reference_index)
    mid = stack.shape[-2] // 2
    fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.2), constrained_layout=True)
    ax = axes.ravel()
    im0 = ax[0].imshow(stack[ref], origin="lower", extent=ext, cmap="inferno")
    ax[0].set_title(f"{route} xy at declared z = {z_um[ref]:.1f} um", fontsize=9); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.03)
    crop = _central_crop_2d(stack[ref], 0.35)
    ax[1].imshow(crop, origin="lower", cmap="inferno"); ax[1].set_title(f"{route} central crop (hollow core)", fontsize=9); ax[1].set_axis_off()
    zext = [float(x_um[0]), float(x_um[-1]), float(z_um[0]), float(z_um[-1])]
    ax[2].imshow(stack[:, mid, :], origin="lower", aspect="auto", extent=zext, cmap="inferno")
    ax[2].axhline(float(z_um[ref]), color="w", lw=0.7, alpha=0.7); ax[2].set_title(f"{route} x-z map", fontsize=9); ax[2].set_xlabel("x (um)"); ax[2].set_ylabel("z (um)")
    ax[3].imshow(stack[:, :, mid], origin="lower", aspect="auto", extent=zext, cmap="inferno")
    ax[3].axhline(float(z_um[ref]), color="w", lw=0.7, alpha=0.7); ax[3].set_title(f"{route} y-z map", fontsize=9); ax[3].set_xlabel("y (um)"); ax[3].set_ylabel("z (um)")
    # z-stack montage (up to 7 planes across the zone)
    n_show = min(7, stack.shape[0])
    idxs = np.linspace(0, stack.shape[0] - 1, n_show).astype(int)
    montage = np.concatenate([_central_crop_2d(stack[i], 0.5) / max(float(np.nanmax(stack[i])), EPS) for i in idxs], axis=1)
    ax[4].imshow(montage, origin="lower", cmap="inferno", vmin=0, vmax=1, aspect="auto")
    ax[4].set_title(f"{route} z-stack montage (z={z_um[idxs[0]]:.0f}..{z_um[idxs[-1]]:.0f} um)", fontsize=9); ax[4].set_axis_off()
    ax[5].plot(x_um, stack[ref, mid, :] / max(float(np.nanmax(stack[ref])), EPS), color="k", lw=1.0)
    ax[5].set_title(f"{route} x profile at ref (dark centre)", fontsize=9); ax[5].set_xlabel("x (um)"); ax[5].set_ylabel("norm. I"); ax[5].set_ylim(0, 1.05)
    ax[6].plot(x_um, stack[ref, :, mid] / max(float(np.nanmax(stack[ref])), EPS), color="k", lw=1.0)
    ax[6].set_title(f"{route} y profile at ref", fontsize=9); ax[6].set_xlabel("y (um)"); ax[6].set_ylabel("norm. I"); ax[6].set_ylim(0, 1.05)
    on_axis = stack[:, mid, mid]
    ring_peak = np.asarray([np.nanmax(stack[i]) for i in range(stack.shape[0])], dtype=float)
    ax[7].plot(z_um, on_axis / max(float(np.nanmax(ring_peak)), EPS), color="tab:red", label="on-axis")
    ax[7].plot(z_um, ring_peak / max(float(np.nanmax(ring_peak)), EPS), color="tab:blue", ls="--", label="ring peak")
    ax[7].axvline(float(z_um[ref]), color="0.5", lw=0.8)
    ax[7].set_title(f"{route} on-axis vs ring-peak vs z", fontsize=9); ax[7].set_xlabel("z (um)"); ax[7].set_ylabel("norm. I"); ax[7].legend(fontsize=7)
    s = result.f0_survival if route.upper() == "F0" else result.f2_survival
    src = s["source_like_hexagonal_bessel_survival"]
    sym = s["symmetry_classification"]
    fig.suptitle(
        f"MODE 1 {route} sample region [{sym['reference_symmetry_class']}]: dark-core ratio(ref)={src['reference_dark_core_ratio']:.3f}, "
        f"hollow planes={src['fraction_planes_hollow']:.2f}, "
        f"o3/o6={sym['reference_order3_over_order6']:.2f}, c120-c60={sym['reference_c120_minus_c60']:.3f}, "
        f"triangular planes={sym['fraction_planes_triangular_lobed']:.2f}", fontsize=10.5)
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1_f0_vs_f2(result: Mode1Result, *, output_path: str | Path | None = None) -> tuple[Any, Any]:
    """F0 vs F2 comparison: equal-power differences, Ez maps, energy fractions, correlations."""

    import matplotlib.pyplot as plt

    if result.f2 is None:
        raise ValueError("F2 route was not run for this MODE 1 result.")
    grid = _mode1_focal_grid(result.f0)
    x_um = _mode1_um_axis(grid)
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    ref = int(result.reference_index)
    f0 = np.asarray(result.f0.intensity_stack, dtype=float)
    f2 = np.asarray(result.f2.intensity_stack, dtype=float)
    mid = f0.shape[-2] // 2
    z_um = np.asarray(result.z_values_m, dtype=float) / 1e-6
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.6), constrained_layout=True)
    ax = axes.ravel()
    ax[0].imshow(f0[ref] / max(float(np.nanmax(f0[ref])), EPS), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    ax[0].set_title("F0 xy (ref)", fontsize=9); ax[0].set_xlabel("x (um)"); ax[0].set_ylabel("y (um)")
    ax[1].imshow(f2[ref] / max(float(np.nanmax(f2[ref])), EPS), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    ax[1].set_title("F2 xy (ref)", fontsize=9); ax[1].set_xlabel("x (um)"); ax[1].set_ylabel("y (um)")
    diff = _equal_power_stack(f0[ref][None, ...])[0] - _equal_power_stack(f2[ref][None, ...])[0]
    dv = max(float(np.nanmax(np.abs(diff))), EPS)
    im2 = ax[2].imshow(diff, origin="lower", extent=ext, cmap="coolwarm", vmin=-dv, vmax=dv)
    ax[2].set_title("F0-F2 equal-power diff (ref)", fontsize=9); ax[2].set_xlabel("x (um)"); ax[2].set_ylabel("y (um)")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03)
    dxz = _equal_power_stack(f0[:, mid, :]) - _equal_power_stack(f2[:, mid, :])
    xzv = max(float(np.nanmax(np.abs(dxz))), EPS)
    zext = [float(x_um[0]), float(x_um[-1]), float(z_um[0]), float(z_um[-1])]
    im3 = ax[3].imshow(dxz, origin="lower", aspect="auto", extent=zext, cmap="coolwarm", vmin=-xzv, vmax=xzv)
    ax[3].set_title("F0-F2 x-z equal-power diff", fontsize=9); ax[3].set_xlabel("x (um)"); ax[3].set_ylabel("z (um)")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.03)
    ax[4].plot(z_um, result.f0.ez_energy_fraction, label="F0 Ez frac", color="tab:red")
    ax[4].plot(z_um, result.f2.ez_energy_fraction, label="F2 Ez frac", color="tab:blue", ls="--")
    ax[4].set_title("Longitudinal Ez energy fraction vs z", fontsize=9); ax[4].set_xlabel("z (um)"); ax[4].set_ylabel("Ez fraction"); ax[4].legend(fontsize=7)
    cmp = result.f0_vs_f2 or {}
    ax[5].set_axis_off()
    txt = [
        "F0 vs F2 (equal-power shape metrics)", "",
        f"full-field corr (ref)   = {cmp.get('reference_full_field_correlation', float('nan')):.4f}",
        f"full-field shape RMS    = {cmp.get('reference_full_field_shape_rms', float('nan')):.4f}",
        f"central-crop corr (ref) = {cmp.get('reference_central_crop_correlation', float('nan')):.4f}",
        f"F0 mean Ez fraction     = {cmp.get('f0_mean_ez_fraction', float('nan')):.4f}",
        f"F2 mean Ez fraction     = {cmp.get('f2_mean_ez_fraction', float('nan')):.4f}", "",
        "F2 is a scoped vectorial pupil-spectrum",
        "diagnostic, not a calibrated Richards-Wolf",
        "reference. Use equal-power shape metrics.",
    ]
    ax[5].text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("MODE 1 F0 (current bridge) vs F2 (vectorial pupil-spectrum diagnostic)", fontsize=11)
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1_v0_to_mode1_scale(
    result: Mode1Result,
    *,
    v0_stage: VisualLadderStageResult | None = None,
    v0_grid_n: int = 384,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Conceptual scale comparison: V0 source-scale dark-core hexagon vs MODE 1 micro-scale F0."""

    import matplotlib.pyplot as plt

    if v0_stage is None:
        v0_stage = run_v0_source_parity_visual_control(NathanSourceParityConfig(grid_n=int(v0_grid_n), z_planes=25))
    v0_stack = np.asarray(v0_stage.intensity_stack, dtype=float)
    v0_grid = v0_stage.grid
    v0_x_mm = np.asarray(v0_grid.get("x", make_xy_grid(int(v0_grid["N"]), float(v0_grid["dx"]))["x"]), dtype=float) / 1e-3
    v0_ref = int(v0_stage.reference_index)
    v0_xy = _central_crop_2d(v0_stack[v0_ref], 0.10)
    v0_ext_full = float(v0_x_mm[-1] * 0.10)

    grid = _mode1_focal_grid(result.f0)
    x_um = _mode1_um_axis(grid)
    f0 = np.asarray(result.f0.intensity_stack, dtype=float)[int(result.reference_index)]
    m1_xy = _central_crop_2d(f0, 0.35)
    m1_ext = float(x_um[-1] * 0.35)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), constrained_layout=True)
    axes[0].imshow(v0_xy / max(float(np.nanmax(v0_xy)), EPS), origin="lower", cmap="inferno",
                   extent=[-v0_ext_full, v0_ext_full, -v0_ext_full, v0_ext_full], vmin=0, vmax=1)
    axes[0].set_title(f"V0 source-scale (validated)\nNathan hexagonal Bessel @ z=60 mm", fontsize=10)
    axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("y (mm)")
    axes[1].imshow(m1_xy / max(float(np.nanmax(m1_xy)), EPS), origin="lower", cmap="inferno",
                   extent=[-m1_ext, m1_ext, -m1_ext, m1_ext], vmin=0, vmax=1)
    sym = result.f0_survival["symmetry_classification"]
    axes[1].set_title(f"MODE 1 micro-scale (F0)\n[{sym['reference_symmetry_class']}] - NOT a clean hexagon", fontsize=10)
    axes[1].set_xlabel("x (um)"); axes[1].set_ylabel("y (um)")
    axes[2].set_axis_off()
    src = result.f0_survival["source_like_hexagonal_bessel_survival"]
    txt = [
        "V0 vs MODE 1 (conceptual, not an overlay)", "",
        "V0: source-scale (mm) free-space validation",
        "of Nathan's mechanism (docs/53). Total",
        "intensity, dark-core hexagonal Bessel.", "",
        "MODE 1: the SAME ideal field injected at P2",
        "and pushed through the inherited micro-scale",
        "Digital Twin optics (um). Physical HWP/QWP/SLM",
        "generation is bypassed.", "",
        f"symmetry class (ref) = {sym['reference_symmetry_class']}",
        f"triangular planes    = {sym['fraction_planes_triangular_lobed']:.2f}",
        f"order3/order6 (ref)   = {sym['reference_order3_over_order6']:.2f}",
        f"c120-c60 (ref)       = {sym['reference_c120_minus_c60']:.3f}  (>0 => C3)",
        f"dark-core ratio (ref)= {src['reference_dark_core_ratio']:.3f}",
        f"outcome              = {result.completion['suggested_outcome']} (MODE 2A/2B blocked)",
    ]
    axes[2].text(0.02, 0.98, "\n".join(txt), va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("V0 source-scale validation -> MODE 1 inherited micro-scale: dark-core TRIANGULAR/C3, not hexagonal", fontsize=10.5)
    _save_fig(fig, output_path)
    return fig, axes


# ===========================================================================
# MODE 1B — ideal downstream geometry / parameter search against the V0 hexagon
# ===========================================================================
#
# The triangle is a FAILURE MODE, not a target.  MODE 1B does NOT optimise H6 /
# order-6 / dark-core / triangle-ish metrics.  It asks the real question: can any
# physically meaningful inherited/downstream parameter set make the IDEAL P2 field
# look like a scaled version of Nathan's VALIDATED V0 hexagonal Bessel output?
#
# Reconnaissance (docs/55): the inherited high-NA objective is NOT the cause — the
# pre-objective free-space path is triangular too.  The controlling variable is the
# number of Bessel rings across the beam, n_rings = w0 * k_r / (2*pi) with
# k_r = (2*pi/lambda)*(n-1)*tan(base_angle).  Nathan's ~31 rings give a hexagon; a
# micro-scale beam gives too few rings (=> triangle) unless the axicon is steepened.
# The search therefore sweeps the two physical knobs that set the ring count: the
# axicon base angle and the beam radius.  A candidate PASSES only if it both matches
# the V0 template (scale/rotation invariant) AND classifies visual_hexagonal_field.

MODE1B_STAGE = "nathan_mode1b_geometry_target_search"
MODE1B_ALLOWED_OUTCOMES = ("M1B-A-realistic", "M1B-A-exploratory", "M1B-B", "M1B-C", "M1B-D")
MODE1B_TEMPLATE_ANGULAR_CORRELATION_PASS = 0.60
MODE1B_XY_CORRELATION_PASS = 0.45
MODE1B_ORDER3_OVER_ORDER6_MAX = 0.50
MODE1B_DARK_CORE_RATIO_MAX = 0.15


def effective_ring_count(beam_radius_m: float, k_r_m_inv: float) -> float:
    """Approximate number of Bessel rings across a Gaussian input radius."""

    return float(float(beam_radius_m) * abs(float(k_r_m_inv)) / (2.0 * np.pi))


@dataclass(frozen=True)
class Mode1BPlaneRadius:
    plane_id: str
    radius_1e_field_m: float
    radius_method: str
    grid_dx_m: float | None
    grid_n: int | None
    source: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mode1BRingCountAudit:
    rows: tuple[Mapping[str, Any], ...]
    inherited_ring_count_used_by_old_mode1b: float
    corrected_ring_count_at_p2: float | None
    corrected_ring_count_at_axicon_or_pupil: float | None
    old_new_ratio: float | None
    conclusion: str


def one_over_e_field_radius_from_intensity(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    *,
    threshold_fraction: float = 1 / np.e**2,
) -> float:
    """Estimate the Gaussian 1/e field radius from a 2D intensity moment.

    For ``A(r) = A0 exp(-r^2 / w^2)``, intensity is
    ``I(r) = I0 exp(-2 r^2 / w^2)`` and ``<r^2> = w^2 / 2``.
    The ``threshold_fraction`` argument is retained for audit readability; the
    moment estimator is used because it is less grid-threshold sensitive.
    """

    _ = threshold_fraction
    I = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    R = np.asarray(grid["R"], dtype=float)
    total = float(np.sum(I))
    if total <= 0.0:
        return float("nan")
    r2_mean = float(np.sum(I * R**2) / total)
    return float(np.sqrt(2.0 * r2_mean))


def one_over_e_field_radius_from_vector_field(field: Any, grid: Mapping[str, Any] | None = None) -> float:
    """Estimate 1/e field radius from a VectorField-like object."""

    ex_obj = getattr(field, "Ex", None)
    if ex_obj is None:
        ex_obj = getattr(field, "ex")
    ey_obj = getattr(field, "Ey", None)
    if ey_obj is None:
        ey_obj = getattr(field, "ey")
    ez_obj = getattr(field, "Ez", None)
    if ez_obj is None:
        ez_obj = getattr(field, "ez", None)
    ex = np.asarray(ex_obj, dtype=complex)
    ey = np.asarray(ey_obj, dtype=complex)
    intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2
    if ez_obj is not None:
        intensity = intensity + np.abs(np.asarray(ez_obj, dtype=complex)) ** 2
    return one_over_e_field_radius_from_intensity(intensity, field.grid if grid is None else grid)


def effective_ring_count_for_plane(
    *,
    radius_1e_field_m: float,
    k_r_m_inv: float,
) -> float:
    """Plane-labelled ring-count helper; radius and k_r must belong to the same plane."""

    return float(float(radius_1e_field_m) * abs(float(k_r_m_inv)) / (2.0 * np.pi))


def _mode1b_ring_count_row(
    *,
    plane_id: str,
    radius_1e_field_m: float,
    k_r_m_inv: float,
    radius_source: str,
    kr_source: str,
    physical_meaning: str,
    valid_for_hexagon_scaling: bool,
    radius_method: str,
    grid_dx_m: float | None = None,
    grid_n: int | None = None,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    radius = float(radius_1e_field_m)
    kr = float(k_r_m_inv)
    return {
        "plane_id": str(plane_id),
        "radius_1e_field_m": radius,
        "radius_1e_field_um": float(radius / 1e-6),
        "k_r_m_inv": kr,
        "ring_count": effective_ring_count_for_plane(radius_1e_field_m=radius, k_r_m_inv=kr),
        "radius_source": str(radius_source),
        "kr_source": str(kr_source),
        "physical_meaning": str(physical_meaning),
        "valid_for_hexagon_scaling": bool(valid_for_hexagon_scaling),
        "radius_method": str(radius_method),
        "grid_dx_m": None if grid_dx_m is None else float(grid_dx_m),
        "grid_n": None if grid_n is None else int(grid_n),
        "notes": tuple(str(note) for note in notes),
    }


def audit_mode1b_ring_count_planes(
    config: NathanHexagonConfig | None = None,
) -> Mode1BRingCountAudit:
    """Report ring count at every physically meaningful MODE 1B plane.

    This audit separates the source plane, P2/axicon input plane, and sample
    plane.  The previous MODE 1B inherited estimate mixed a sample-plane radius
    with a pre-axicon transverse wavevector, so it is retained only as an
    explicitly labelled old diagnostic.
    """

    cfg = config or NathanHexagonConfig.fast()
    twin = cfg.twin
    design = compute_design_from_targets(twin.laser, twin.target, twin.material)
    params = resolve_vector_axicon_parameters(twin)

    v0_cfg = NathanSourceParityConfig()
    p2_grid = default_nathan_grid(cfg)
    p2_field = canonical_target_field(cfg, grid=p2_grid)
    p2_radius = one_over_e_field_radius_from_vector_field(p2_field, p2_grid)
    sample_radius = float(design.w0_sample_m)
    old_ring = effective_ring_count_for_plane(radius_1e_field_m=sample_radius, k_r_m_inv=params.k_r_pre_m_inv)
    corrected_p2 = effective_ring_count_for_plane(radius_1e_field_m=p2_radius, k_r_m_inv=params.k_r_pre_m_inv)
    corrected_axicon = corrected_p2
    ratio = float(corrected_p2 / old_ring) if old_ring > 0.0 else None

    rows = [
        _mode1b_ring_count_row(
            plane_id="v0_source_plane",
            radius_1e_field_m=float(v0_cfg.beam_radius_m),
            k_r_m_inv=float(v0_cfg.k_r_m_inv),
            radius_source="NathanSourceParityConfig.beam_radius_m",
            kr_source="NathanSourceParityConfig.k_r_m_inv",
            physical_meaning="validated Nathan source-scale free-space V0 input plane",
            valid_for_hexagon_scaling=True,
            radius_method="declared 1/e field radius",
            grid_dx_m=float(v0_cfg.window_m) / float(v0_cfg.grid_n),
            grid_n=int(v0_cfg.grid_n),
        ),
        _mode1b_ring_count_row(
            plane_id="old_mode1b_mixed_sample_radius_pre_axicon_kr",
            radius_1e_field_m=sample_radius,
            k_r_m_inv=float(params.k_r_pre_m_inv),
            radius_source="compute_design_from_targets(...).w0_sample_m",
            kr_source="resolve_vector_axicon_parameters(...).k_r_pre_m_inv",
            physical_meaning="old inherited estimate; mixes sample-plane radius with pre-axicon k_r",
            valid_for_hexagon_scaling=False,
            radius_method="old sample-plane design radius",
            notes=("not physically meaningful for P2/axicon ring-count scaling",),
        ),
        _mode1b_ring_count_row(
            plane_id="p2_handoff_plane",
            radius_1e_field_m=p2_radius,
            k_r_m_inv=float(params.k_r_pre_m_inv),
            radius_source="measured canonical_target_field total intensity on default_nathan_grid",
            kr_source="resolve_vector_axicon_parameters(...).k_r_pre_m_inv",
            physical_meaning="ideal P2 field radius at the handoff/axicon-input scale",
            valid_for_hexagon_scaling=True,
            radius_method="intensity moment w=sqrt(2<r^2>)",
            grid_dx_m=float(p2_grid["dx"]),
            grid_n=int(p2_grid["N"]),
        ),
        _mode1b_ring_count_row(
            plane_id="axicon_or_pupil_input_plane",
            radius_1e_field_m=p2_radius,
            k_r_m_inv=float(params.k_r_pre_m_inv),
            radius_source="P2 radius carried into the current ideal vector-axicon handoff",
            kr_source="resolve_vector_axicon_parameters(...).k_r_pre_m_inv",
            physical_meaning="current MODE 1 vector axicon is applied to the P2-scale field",
            valid_for_hexagon_scaling=True,
            radius_method="same measured P2 1/e field radius",
            grid_dx_m=float(p2_grid["dx"]),
            grid_n=int(p2_grid["N"]),
            notes=("P2 and axicon input are not separated by a measured relay in this ideal handoff model",),
        ),
        _mode1b_ring_count_row(
            plane_id="sample_plane_design_radius",
            radius_1e_field_m=sample_radius,
            k_r_m_inv=float(params.k_r_surface_m_inv),
            radius_source="compute_design_from_targets(...).w0_sample_m",
            kr_source="resolve_vector_axicon_parameters(...).k_r_surface_m_inv",
            physical_meaning="sample-plane radius paired with sample/surface transverse k; not the axicon-input radius",
            valid_for_hexagon_scaling=False,
            radius_method="design sample-plane 1/e field radius",
            notes=("valid sample-plane bookkeeping, but must not be paired with pre-axicon k_r",),
        ),
    ]

    if ratio is not None and (ratio > 5.0 or ratio < 0.2):
        conclusion = (
            "Old MODE 1B inherited ring count used the wrong plane radius; M1B-B cannot be treated as final "
            "until the search is interpreted with plane-correct radii. Actual inherited MODE 1 remains a "
            "triangular visual failure, so MODE 2A/2B remain blocked."
        )
    else:
        conclusion = "Old MODE 1B inherited ring count is plane-consistent within the audit tolerance."

    return Mode1BRingCountAudit(
        rows=tuple(rows),
        inherited_ring_count_used_by_old_mode1b=float(old_ring),
        corrected_ring_count_at_p2=float(corrected_p2),
        corrected_ring_count_at_axicon_or_pupil=float(corrected_axicon),
        old_new_ratio=ratio,
        conclusion=conclusion,
    )


def write_mode1b_ring_count_audit(
    audit: Mode1BRingCountAudit,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode1b_geometry_search",
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "mode1b_ring_count_plane_audit.csv"
    json_path = out / "mode1b_ring_count_plane_audit.json"
    _write_rows(csv_path, audit.rows)
    json_path.write_text(json.dumps(_json_ready(asdict(audit)), indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


@dataclass(frozen=True)
class Mode1BFeasibility:
    k_r_m_inv: float
    ring_count: float
    axicon_base_angle_deg: float
    axicon_apex_angle_deg: float
    phase_period_m: float
    objective_na_required: float
    objective_na_available: float
    within_objective_na: bool
    slm_encoded_axicon_feasible: bool
    physical_axicon_candidate: bool
    thin_axicon_paraxial_warning: bool
    feasibility_class: str
    notes: tuple[str, ...]


def mode1b_feasibility(
    *,
    wavelength_m: float,
    n_axicon: float,
    n_medium: float,
    base_angle_deg: float,
    beam_radius_m: float,
    objective_na: float,
    slm_pixel_pitch_m: float | None = None,
    min_slm_pixels_per_2pi: float = 4.0,
) -> Mode1BFeasibility:
    """Feasibility report for an ideal MODE 1B geometry candidate.

    The visual model may use a thin axicon at exploratory angles, but this report
    keeps that distinct from an existing-lab-realistic claim.
    """

    k0 = 2.0 * np.pi / float(wavelength_m)
    beta = np.deg2rad(float(base_angle_deg))
    k_r = float(k0 * (float(n_axicon) - float(n_medium)) * np.tan(beta))
    phase_period = float(2.0 * np.pi / max(abs(k_r), EPS))
    ring_count = effective_ring_count(float(beam_radius_m), k_r)
    na_required = float(abs(k_r) / max(k0, EPS))
    within_na = bool(na_required <= float(objective_na) + 1e-15)
    slm_ok = False if slm_pixel_pitch_m is None else bool(
        phase_period >= float(min_slm_pixels_per_2pi) * float(slm_pixel_pitch_m)
    )
    high_angle = bool(float(base_angle_deg) > 8.0)
    notes: list[str] = []
    if not within_na:
        notes.append("required transverse k exceeds objective NA")
    if slm_pixel_pitch_m is not None and not slm_ok:
        notes.append("phase period too fine for SLM encoded axicon")
    if high_angle:
        notes.append("large axicon base angle; thin/paraxial approximation must be treated as exploratory")
    if ring_count < 5.0:
        notes.append("low ring count; likely triangular/lobed failure mode")

    physical_candidate = bool(within_na and ring_count >= 5.0)
    if within_na and ring_count >= 5.0 and not high_angle:
        feasibility_class = "physically_plausible_existing_model"
    elif within_na and ring_count >= 5.0:
        feasibility_class = "exploratory_high_angle_redesign"
    else:
        feasibility_class = "not_feasible_or_not_useful"

    return Mode1BFeasibility(
        k_r_m_inv=k_r,
        ring_count=float(ring_count),
        axicon_base_angle_deg=float(base_angle_deg),
        axicon_apex_angle_deg=float(180.0 - 2.0 * float(base_angle_deg)),
        phase_period_m=phase_period,
        objective_na_required=na_required,
        objective_na_available=float(objective_na),
        within_objective_na=within_na,
        slm_encoded_axicon_feasible=bool(slm_ok),
        physical_axicon_candidate=physical_candidate,
        thin_axicon_paraxial_warning=high_angle,
        feasibility_class=feasibility_class,
        notes=tuple(notes),
    )


def _mode1b_even_axis_crop(image: np.ndarray, grid: Mapping[str, Any], crop_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Central crop that preserves an axis-sampled centre pixel for even source grids."""

    arr = np.asarray(image)
    n = int(arr.shape[-1])
    frac = min(max(float(crop_fraction), EPS), 1.0)
    side = max(4, 2 * int(round(0.5 * frac * n)))
    side = min(side, n if n % 2 == 0 else n - 1)
    start = n // 2 - side // 2
    stop = start + side
    crop = arr[start:stop, start:stop]
    x_full = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float)
    x = x_full[start:stop]
    X, Y = np.meshgrid(x, x, indexing="xy")
    fx = np.fft.fftshift(np.fft.fftfreq(side, d=float(grid["dx"])))
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    cropped_grid = {
        **dict(grid),
        "N": int(side),
        "x": x,
        "y": x,
        "X": X,
        "Y": Y,
        "R": np.hypot(X, Y),
        "PHI": np.arctan2(Y, X),
        "FX": FX,
        "FY": FY,
    }
    return crop, cropped_grid


def _mode1b_dark_core_ratio_for_ring(image: np.ndarray, grid: Mapping[str, Any], ring_radius_m: float) -> float:
    arr = np.asarray(image, dtype=float)
    R = np.asarray(grid["R"], dtype=float)
    centre = tuple(s // 2 for s in arr.shape)
    mask = (R >= 0.75 * float(ring_radius_m)) & (R <= 1.25 * float(ring_radius_m))
    ring_peak = float(np.nanmax(arr[mask])) if np.any(mask) else float(np.nanmax(arr))
    return float(arr[centre] / max(ring_peak, EPS))


def _normalise_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr - float(np.nanmean(arr))
    norm = float(np.sqrt(np.nansum(arr * arr)))
    return arr / norm if norm > 0.0 else np.zeros_like(arr, dtype=float)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalise_vector(np.asarray(a, dtype=float))
    bb = _normalise_vector(np.asarray(b, dtype=float))
    if aa.shape != bb.shape:
        raise ValueError(f"correlation arrays must have matching shape, got {aa.shape} and {bb.shape}")
    return float(np.nansum(aa * bb))


def angular_profile_on_ring(
    image: np.ndarray,
    grid: Mapping[str, Any],
    ring_radius_m: float,
    *,
    ring_width_frac: float = 0.25,
    angular_bins: int = 720,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an annular angular intensity profile around ``ring_radius_m``."""

    arr = np.asarray(image, dtype=float)
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float) % (2.0 * np.pi)
    rr = float(ring_radius_m)
    half_width = max(float(ring_width_frac) * rr, float(grid["dx"]))
    mask = (R >= rr - half_width) & (R <= rr + half_width)
    bins = np.linspace(0.0, 2.0 * np.pi, int(angular_bins) + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    if not np.any(mask):
        return centres, np.zeros(int(angular_bins), dtype=float)
    idx = np.clip(np.digitize(PHI[mask], bins) - 1, 0, int(angular_bins) - 1)
    weights = arr[mask]
    prof = np.bincount(idx, weights=weights, minlength=int(angular_bins)).astype(float)
    counts = np.bincount(idx, minlength=int(angular_bins)).astype(float)
    prof = prof / np.maximum(counts, 1.0)
    return centres, prof


def circular_profile_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Maximum Pearson-like correlation over integer circular shifts of ``b``."""

    aa = _normalise_vector(np.asarray(a, dtype=float))
    bb = _normalise_vector(np.asarray(b, dtype=float))
    if aa.size != bb.size:
        raise ValueError("circular profile correlation requires equal-length profiles")
    best_corr = -np.inf
    best_shift = 0
    for shift in range(int(aa.size)):
        corr = float(np.nansum(aa * np.roll(bb, shift)))
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return float(best_corr), int(best_shift)


def _polar_signature(
    plane: np.ndarray,
    grid: Mapping[str, Any],
    ring_r: float,
    *,
    n_r: int = 48,
    n_theta: int = 180,
    r_max_factor: float = 2.2,
) -> np.ndarray:
    """Scale-normalised polar intensity signature for template matching."""

    x = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float)
    dx = float(grid["dx"])
    x0 = float(x[0])
    rr = np.linspace(0.0, float(r_max_factor), int(n_r))[:, None] * float(ring_r)
    th = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)[None, :]
    xs = rr * np.cos(th)
    ys = rr * np.sin(th)
    cols = (xs - x0) / max(dx, EPS)
    rows = (ys - x0) / max(dx, EPS)
    sig = map_coordinates(
        np.asarray(plane, dtype=float),
        [rows.ravel(), cols.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    )
    sig = sig.reshape(int(n_r), int(n_theta))
    s = float(np.nansum(sig))
    return sig / s if s > 0.0 else sig


def _normalised_axis_profile(
    image: np.ndarray,
    grid: Mapping[str, Any],
    ring_radius_m: float,
    *,
    axis: str,
    samples: int = 512,
    extent_factor: float = 2.2,
) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    x = np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float)
    dx = float(grid["dx"])
    x0 = float(x[0])
    coord = np.linspace(-float(extent_factor), float(extent_factor), int(samples)) * float(ring_radius_m)
    if axis == "x":
        xs = coord
        ys = np.zeros_like(coord)
    elif axis == "y":
        xs = np.zeros_like(coord)
        ys = coord
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    cols = (xs - x0) / max(dx, EPS)
    rows = (ys - x0) / max(dx, EPS)
    prof = map_coordinates(arr, [rows, cols], order=1, mode="constant", cval=0.0)
    peak = float(np.nanmax(prof))
    return prof / peak if peak > 0.0 else prof


@dataclass(frozen=True)
class Mode1BTargetTemplate:
    intensity_xy: np.ndarray
    grid: Mapping[str, Any]
    z_reference_m: float
    ring_radius_m: float
    dark_core_ratio: float
    angular_profile: np.ndarray
    x_profile: np.ndarray
    y_profile: np.ndarray
    symmetry: Mapping[str, float]
    classification: str
    metadata: Mapping[str, Any]

    @property
    def symmetry_class(self) -> str:
        return self.classification


@dataclass(frozen=True)
class Mode1BTemplateScore:
    angular_profile_correlation: float
    xy_correlation: float | None
    x_profile_correlation: float
    y_profile_correlation: float
    best_rotation_deg: float
    candidate_ring_radius_m: float
    target_ring_radius_m: float
    scale_factor: float


def build_mode1b_target_template(
    grid_n: int = 512,
    z_planes: int = 41,
    crop_fraction: float = 0.10,
) -> Mode1BTargetTemplate:
    """Build the V0 target template for MODE 1B.

    The ring radius is anchored on the validated full V0 reference plane, then a
    fixed central crop is used for the template views/profiles.  The crop helper
    preserves the axis-sampled centre pixel so the V0 grid-centring fix is not
    undone by an odd crop.
    """

    cfg = NathanSourceParityConfig(grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_v0_source_parity_visual_control(cfg)
    ref = int(v0.reference_index)
    full_xy = np.asarray(v0.intensity_stack[ref], dtype=float)
    full_grid = v0.grid
    full_diag = _v0_plane_diagnostics(full_xy, full_grid)
    ring = float(full_diag["ring_radius_m"])
    crop_xy, crop_grid = _mode1b_even_axis_crop(full_xy, full_grid, float(crop_fraction))
    dark_core = _mode1b_dark_core_ratio_for_ring(crop_xy, crop_grid, ring)
    sym = _mode1_symmetry(crop_xy, crop_grid, ring)
    cls = mode1_symmetry_class(sym, dark_core)
    if cls != "visual_hexagonal_field":
        raise ValueError(f"V0 template did not classify as visual_hexagonal_field (got {cls!r}); check the classifier.")
    _, angular = angular_profile_on_ring(crop_xy, crop_grid, ring)
    return Mode1BTargetTemplate(
        intensity_xy=crop_xy,
        grid=crop_grid,
        z_reference_m=float(v0.z_values_m[ref]),
        ring_radius_m=ring,
        dark_core_ratio=float(dark_core),
        angular_profile=angular,
        x_profile=_normalised_axis_profile(crop_xy, crop_grid, ring, axis="x"),
        y_profile=_normalised_axis_profile(crop_xy, crop_grid, ring, axis="y"),
        symmetry={str(k): float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in sym.items()},
        classification=cls,
        metadata={
            "stage": MODE1B_STAGE,
            "source": "validated V0 Nathan source parity visual control",
            "grid_n": int(grid_n),
            "z_planes": int(z_planes),
            "crop_fraction_requested": float(crop_fraction),
            "crop_grid_n": int(crop_grid["N"]),
            "full_ring_radius_m": ring,
            "full_dark_core_ratio": float(full_diag["central_core_darkness"]),
            "v0_k_r_m_inv": float(cfg.k_r_m_inv),
            "v0_effective_ring_count": effective_ring_count(cfg.beam_radius_m, cfg.k_r_m_inv),
            "note": "V0 is the calibration anchor; a MODE 1B target regression is an error, not a candidate outcome.",
        },
    )


def compare_to_v0_template(
    image: np.ndarray,
    grid: Mapping[str, Any],
    template: Mode1BTargetTemplate,
    *,
    candidate_ring_radius_m: float | None = None,
) -> Mode1BTemplateScore:
    """Compare a candidate plane against the V0 target with allowed nuisances only."""

    ring = float(candidate_ring_radius_m) if candidate_ring_radius_m is not None else float(_v0_plane_diagnostics(image, grid)["ring_radius_m"])
    _, cand_ang = angular_profile_on_ring(image, grid, ring, angular_bins=int(template.angular_profile.size))
    ang_corr, best_shift = circular_profile_correlation(cand_ang, template.angular_profile)
    best_rotation = float(360.0 * best_shift / max(int(template.angular_profile.size), 1))
    x_prof = _normalised_axis_profile(image, grid, ring, axis="x", samples=int(template.x_profile.size))
    y_prof = _normalised_axis_profile(image, grid, ring, axis="y", samples=int(template.y_profile.size))
    x_corr = _safe_corr(x_prof, template.x_profile)
    y_corr = _safe_corr(y_prof, template.y_profile)

    cand_sig = _polar_signature(image, grid, ring, n_theta=180)
    tmpl_sig = _polar_signature(template.intensity_xy, template.grid, template.ring_radius_m, n_theta=180)
    best_xy = -np.inf
    for shift in range(cand_sig.shape[1]):
        best_xy = max(best_xy, _safe_corr(np.roll(cand_sig, shift, axis=1).ravel(), tmpl_sig.ravel()))

    return Mode1BTemplateScore(
        angular_profile_correlation=float(ang_corr),
        xy_correlation=float(best_xy),
        x_profile_correlation=float(x_corr),
        y_profile_correlation=float(y_corr),
        best_rotation_deg=best_rotation,
        candidate_ring_radius_m=ring,
        target_ring_radius_m=float(template.ring_radius_m),
        scale_factor=float(ring / max(float(template.ring_radius_m), EPS)),
    )


def mode1b_candidate_passes_hexagon_gate(
    *,
    symmetry_class: str,
    symmetry: Mapping[str, float],
    template_score: Mode1BTemplateScore,
    dark_core_ratio: float,
    feasibility: Mode1BFeasibility,
    require_existing_realism: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    """Visual/template pass gate with an optional existing-realism requirement."""

    reasons: list[str] = []
    if str(symmetry_class) != "visual_hexagonal_field":
        reasons.append(f"class is {symmetry_class}, not visual_hexagonal_field")
    if float(symmetry.get("rot_corr_120", 0.0)) - float(symmetry.get("rot_corr_60", 0.0)) > 0.02:
        reasons.append("C3/triangular self-similarity exceeds C6")
    if float(symmetry.get("order3_over_order6", np.inf)) > MODE1B_ORDER3_OVER_ORDER6_MAX:
        reasons.append("order-3 content too strong relative to order-6")
    if float(template_score.angular_profile_correlation) < MODE1B_TEMPLATE_ANGULAR_CORRELATION_PASS:
        reasons.append("low angular-profile similarity to V0 template")
    if template_score.xy_correlation is not None and float(template_score.xy_correlation) < MODE1B_XY_CORRELATION_PASS:
        reasons.append("low scale/rotation-normalised XY similarity to V0 template")
    if float(dark_core_ratio) > MODE1B_DARK_CORE_RATIO_MAX:
        reasons.append("central core not sufficiently dark")
    if require_existing_realism and feasibility.feasibility_class != "physically_plausible_existing_model":
        reasons.append("candidate is exploratory, not existing-realistic")
    return (len(reasons) == 0), tuple(reasons)


@dataclass(frozen=True)
class Mode1BCandidate:
    candidate_id: str
    beam_radius_m: float
    base_angle_deg: float
    sector_rotation_rad: float
    z_reference_m: float
    grid_n: int
    symmetry_class: str
    template_score: Mode1BTemplateScore
    feasibility: Mode1BFeasibility
    dark_core_ratio: float
    ring_radius_m: float
    pass_hexagon_gate: bool
    fail_reasons: tuple[str, ...]
    output_paths: Mapping[str, str] = field(default_factory=dict)
    tier: str = "candidate"
    model_family: str = "plane_corrected_free_space_continuation"
    radius_plane_id: str = "candidate_input_plane"
    radius_1e_field_m: float = float("nan")
    k_r_m_inv: float = float("nan")
    ring_count: float = float("nan")
    symmetry: Mapping[str, Any] = field(default_factory=dict)
    intensity_xy: np.ndarray | None = field(default=None, repr=False, compare=False)
    grid: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    z_values_m: np.ndarray | None = field(default=None, repr=False, compare=False)
    reference_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_hexagonal(self) -> bool:
        return self.symmetry_class == "visual_hexagonal_field"

    @property
    def n_rings(self) -> float:
        return float(self.ring_count if np.isfinite(self.ring_count) else self.feasibility.ring_count)


@dataclass(frozen=True)
class Mode1BSearchResult:
    target_template: Mode1BTargetTemplate
    candidates: tuple[Mode1BCandidate, ...]
    shortlist_candidates: tuple[Mode1BCandidate, ...]
    summary_rows: tuple[Mapping[str, Any], ...]
    outcome_report: Mapping[str, Any]
    search_space: Mapping[str, Any]
    output_paths: Mapping[str, str] = field(default_factory=dict)


def _mode1b_candidate_id(tier: str, base_angle_deg: float, beam_radius_m: float, sector_rotation_rad: float) -> str:
    rot_deg = float(np.rad2deg(sector_rotation_rad))
    text = f"{tier}_base{float(base_angle_deg):05.2f}deg_w{float(beam_radius_m)/1e-6:05.1f}um_rot{rot_deg:04.1f}deg"
    return text.replace("-", "m").replace(".", "p")


def run_mode1b_candidate(
    template: Mode1BTargetTemplate,
    *,
    base_angle_deg: float,
    beam_radius_m: float,
    tier: str = "candidate",
    candidate_id: str | None = None,
    window_factor: float = 8.0,
    grid_n: int = 256,
    z_planes: int = 13,
    z_span_factor: float = 2.5,
    wavelength_m: float = 1030e-9,
    n_axicon: float = 1.458,
    n_medium: float = 1.0,
    objective_na: float = 0.45,
    slm_pixel_pitch_m: float | None = 8e-6,
    sector_rotation_rad: float = 0.0,
    model_family: str | None = None,
    radius_plane_id: str = "candidate_input_plane",
) -> Mode1BCandidate:
    """Run one ideal MODE 1B free-space downstream geometry candidate."""

    cid = candidate_id or _mode1b_candidate_id(str(tier), float(base_angle_deg), float(beam_radius_m), float(sector_rotation_rad))
    window_m = float(window_factor) * float(beam_radius_m)
    grid = _mode1_grid(int(grid_n), window_m / int(grid_n), straddle=False)
    field, _ = nathan_literal_segmented_ra_input(
        grid,
        wavelength_m=float(wavelength_m),
        beam_radius_m=float(beam_radius_m),
        n_pairs=3,
        sector_theta_rad=np.pi / 3.0,
        sector_rotation_rad=float(sector_rotation_rad),
    )
    base = np.deg2rad(float(base_angle_deg))
    after, meta = _apply_free_space_vector_axicon(
        field,
        n_axicon=float(n_axicon),
        n_medium=float(n_medium),
        base_angle_rad=base,
    )
    z_ref = float(beam_radius_m) / max(np.tan(base), 1e-9)
    z_values = np.linspace(0.05 * z_ref, float(z_span_factor) * z_ref, int(z_planes))
    ref = int(np.argmin(np.abs(z_values - z_ref)))
    stack, _ = _free_space_intensity_stack(after, z_values)
    plane = np.asarray(stack[ref], dtype=float)
    diag = _v0_plane_diagnostics(plane, grid)
    ring = float(diag["ring_radius_m"])
    dark = float(diag["central_core_darkness"])
    sym = _mode1_symmetry(plane, grid, ring)
    cls = mode1_symmetry_class(sym, dark)
    score = compare_to_v0_template(plane, grid, template, candidate_ring_radius_m=ring)
    feasibility = mode1b_feasibility(
        wavelength_m=float(wavelength_m),
        n_axicon=float(n_axicon),
        n_medium=float(n_medium),
        base_angle_deg=float(base_angle_deg),
        beam_radius_m=float(beam_radius_m),
        objective_na=float(objective_na),
        slm_pixel_pitch_m=slm_pixel_pitch_m,
    )
    passed, reasons = mode1b_candidate_passes_hexagon_gate(
        symmetry_class=cls,
        symmetry=sym,
        template_score=score,
        dark_core_ratio=dark,
        feasibility=feasibility,
    )
    family = model_family or (
        "exploratory_high_angle_redesign"
        if float(base_angle_deg) > 8.0
        else "plane_corrected_free_space_continuation"
    )
    return Mode1BCandidate(
        candidate_id=cid,
        beam_radius_m=float(beam_radius_m),
        base_angle_deg=float(base_angle_deg),
        sector_rotation_rad=float(sector_rotation_rad),
        z_reference_m=float(z_values[ref]),
        grid_n=int(grid_n),
        symmetry_class=cls,
        template_score=score,
        feasibility=feasibility,
        dark_core_ratio=dark,
        ring_radius_m=ring,
        pass_hexagon_gate=bool(passed),
        fail_reasons=tuple(reasons),
        tier=str(tier),
        model_family=str(family),
        radius_plane_id=str(radius_plane_id),
        radius_1e_field_m=float(beam_radius_m),
        k_r_m_inv=float(feasibility.k_r_m_inv),
        ring_count=float(feasibility.ring_count),
        symmetry=sym,
        intensity_xy=plane.astype(np.float32),
        grid=grid,
        z_values_m=z_values,
        reference_index=ref,
        metadata={
            "window_m": window_m,
            "z_planes": int(z_planes),
            "z_span_factor": float(z_span_factor),
            "n_axicon": float(n_axicon),
            "n_medium": float(n_medium),
            "wavelength_m": float(wavelength_m),
            "axicon_meta": dict(meta),
            "candidate_type": (
                "existing architecture candidate"
                if feasibility.feasibility_class == "physically_plausible_existing_model"
                else (
                    "ideal exploratory redesign candidate"
                    if feasibility.feasibility_class == "exploratory_high_angle_redesign"
                    else "not feasible or not useful candidate"
                )
            ),
        },
    )


def mode1b_candidate_row(cand: Mode1BCandidate) -> dict[str, Any]:
    score = cand.template_score
    feas = cand.feasibility
    return {
        "candidate_id": cand.candidate_id,
        "tier": cand.tier,
        "model_family": cand.model_family,
        "radius_plane_id": cand.radius_plane_id,
        "radius_1e_field_m": float(cand.radius_1e_field_m),
        "radius_1e_field_um": float(cand.radius_1e_field_m / 1e-6),
        "base_angle_deg": float(cand.base_angle_deg),
        "axicon_apex_angle_deg": float(feas.axicon_apex_angle_deg),
        "apex_angle_deg": float(feas.axicon_apex_angle_deg),
        "beam_radius_um": float(cand.beam_radius_m / 1e-6),
        "sector_rotation_deg": float(np.rad2deg(cand.sector_rotation_rad)),
        "grid_n": int(cand.grid_n),
        "z_reference_mm": float(cand.z_reference_m / 1e-3),
        "ring_radius_um": float(cand.ring_radius_m / 1e-6),
        "dark_core_ratio": float(cand.dark_core_ratio),
        "symmetry_class": str(cand.symmetry_class),
        "pass_hexagon_gate": bool(cand.pass_hexagon_gate),
        "fail_reasons": "; ".join(cand.fail_reasons),
        "angular_profile_correlation": float(score.angular_profile_correlation),
        "xy_correlation": "" if score.xy_correlation is None else float(score.xy_correlation),
        "x_profile_correlation": float(score.x_profile_correlation),
        "y_profile_correlation": float(score.y_profile_correlation),
        "best_rotation_deg": float(score.best_rotation_deg),
        "scale_factor": float(score.scale_factor),
        "k_r_m_inv": float(feas.k_r_m_inv),
        "ring_count": float(cand.n_rings),
        "n_rings": float(cand.n_rings),
        "candidate_k_r_m_inv": float(cand.k_r_m_inv),
        "phase_period_m": float(feas.phase_period_m),
        "phase_period_um": float(feas.phase_period_m / 1e-6),
        "objective_na_required": float(feas.objective_na_required),
        "objective_na_available": float(feas.objective_na_available),
        "within_objective_na": bool(feas.within_objective_na),
        "slm_encoded_axicon_feasible": bool(feas.slm_encoded_axicon_feasible),
        "physical_axicon_candidate": bool(feas.physical_axicon_candidate),
        "thin_axicon_paraxial_warning": bool(feas.thin_axicon_paraxial_warning),
        "feasibility_class": str(feas.feasibility_class),
        "feasibility_notes": "; ".join(feas.notes),
        "candidate_type": str(cand.metadata.get("candidate_type", "")),
        "order3_over_order6": float(cand.symmetry.get("order3_over_order6", np.nan)),
        "rot_corr_60": float(cand.symmetry.get("rot_corr_60", np.nan)),
        "rot_corr_120": float(cand.symmetry.get("rot_corr_120", np.nan)),
        "c120_minus_c60": float(cand.symmetry.get("c120_minus_c60", np.nan)),
        "six_sector_max_over_min": float(cand.symmetry.get("six_sector_max_over_min", np.nan)),
        "ring_island_count": int(cand.symmetry.get("ring_island_count", -1)),
    }


def _mode1b_score_for_ranking(cand: Mode1BCandidate) -> float:
    score = cand.template_score
    base = float(score.angular_profile_correlation)
    if score.xy_correlation is not None:
        base += 0.5 * float(score.xy_correlation)
    base += 0.25 * (float(score.x_profile_correlation) + float(score.y_profile_correlation))
    base += 1.0 if cand.symmetry_class == "visual_hexagonal_field" else 0.0
    base += 2.0 if cand.pass_hexagon_gate else 0.0
    return float(base)


def _mode1b_existing_defaults() -> dict[str, Any]:
    cfg = NathanHexagonConfig.fast()
    twin = cfg.twin
    design = compute_design_from_targets(twin.laser, twin.target, twin.material)
    params = resolve_vector_axicon_parameters(twin)
    p2_grid = default_nathan_grid(cfg)
    p2_field = canonical_target_field(cfg, grid=p2_grid)
    p2_radius = one_over_e_field_radius_from_vector_field(p2_field, p2_grid)
    old_sample_radius = float(design.w0_sample_m)
    return {
        "wavelength_m": float(twin.laser.wavelength_m),
        "base_angle_deg": float(np.rad2deg(params.base_angle_rad)),
        "beam_radius_m": float(p2_radius),
        "radius_plane_id": "p2_handoff_plane",
        "radius_source": "measured canonical_target_field total intensity on default_nathan_grid",
        "old_sample_radius_m": old_sample_radius,
        "old_mixed_plane_ring_count": effective_ring_count_for_plane(
            radius_1e_field_m=old_sample_radius,
            k_r_m_inv=float(params.k_r_pre_m_inv),
        ),
        "sample_radius_with_sample_kr_ring_count": effective_ring_count_for_plane(
            radius_1e_field_m=old_sample_radius,
            k_r_m_inv=float(params.k_r_surface_m_inv),
        ),
        "k_r_m_inv": float(params.k_r_pre_m_inv),
        "k_r_surface_m_inv": float(params.k_r_surface_m_inv),
        "objective_na": float(twin.objective.NA),
        "slm_pixel_pitch_m": float(twin.slm.pixel_pitch_m),
        "n_axicon": float(params.n_axicon),
        "n_medium": float(params.n_medium),
    }


def run_mode1b_geometry_search(
    template: Mode1BTargetTemplate | None = None,
    *,
    target_grid_n: int = 512,
    target_z_planes: int = 41,
    target_crop_fraction: float = 0.10,
    grid_n: int = 224,
    z_planes: int = 13,
    tier1_base_angles_deg: Sequence[float] | None = None,
    tier1_beam_radii_m: Sequence[float] | None = None,
    tier2_base_angles_deg: Sequence[float] = (2.0, 4.0, 8.0, 12.0, 16.0, 20.0),
    tier2_beam_radii_m: Sequence[float] | None = None,
    sector_rotation_deg: Sequence[float] = (0.0, 15.0, 30.0),
    max_shortlist: int = 8,
) -> Mode1BSearchResult:
    """Two-tier MODE 1B geometry search against the V0 target template."""

    tmpl = template or build_mode1b_target_template(
        grid_n=int(target_grid_n),
        z_planes=int(target_z_planes),
        crop_fraction=float(target_crop_fraction),
    )
    existing = _mode1b_existing_defaults()
    current_base = float(existing["base_angle_deg"])
    current_w0 = float(existing["beam_radius_m"])
    t1_bases = tuple(tier1_base_angles_deg) if tier1_base_angles_deg is not None else (
        0.75 * current_base,
        current_base,
        1.25 * current_base,
        0.50,
    )
    t1_radii = tuple(tier1_beam_radii_m) if tier1_beam_radii_m is not None else (
        0.75 * current_w0,
        current_w0,
        1.50 * current_w0,
        2.00 * current_w0,
    )
    t2_radii = tuple(tier2_beam_radii_m) if tier2_beam_radii_m is not None else (
        0.25 * current_w0,
        0.50 * current_w0,
        current_w0,
        1.50 * current_w0,
    )
    candidates: list[Mode1BCandidate] = []
    common = {
        "grid_n": int(grid_n),
        "z_planes": int(z_planes),
        "wavelength_m": float(existing["wavelength_m"]),
        "n_axicon": float(existing["n_axicon"]),
        "n_medium": float(existing["n_medium"]),
        "objective_na": float(existing["objective_na"]),
        "slm_pixel_pitch_m": float(existing["slm_pixel_pitch_m"]),
    }
    for base in t1_bases:
        for radius in t1_radii:
            candidates.append(
                run_mode1b_candidate(
                    tmpl,
                    base_angle_deg=float(base),
                    beam_radius_m=float(radius),
                    tier="tier1_existing_near_current",
                    sector_rotation_rad=0.0,
                    model_family="plane_corrected_free_space_continuation",
                    radius_plane_id=str(existing["radius_plane_id"]),
                    **common,
                )
            )
    for base in tier2_base_angles_deg:
        for radius in t2_radii:
            for rot_deg in sector_rotation_deg:
                candidates.append(
                    run_mode1b_candidate(
                        tmpl,
                        base_angle_deg=float(base),
                        beam_radius_m=float(radius),
                        tier="tier2_exploratory_redesign",
                        sector_rotation_rad=float(np.deg2rad(rot_deg)),
                        model_family=(
                            "exploratory_high_angle_redesign"
                            if float(base) > 8.0
                            else "plane_corrected_free_space_continuation"
                        ),
                        radius_plane_id=str(existing["radius_plane_id"]),
                        **common,
                    )
                )
    ranked = tuple(sorted(candidates, key=_mode1b_score_for_ranking, reverse=True))
    shortlist = ranked[: int(max_shortlist)]
    rows = tuple(mode1b_candidate_row(c) for c in candidates)
    search_space = {
        "tier1": {
            "label": "inherited-realistic / near-current",
            "base_angles_deg": tuple(float(v) for v in t1_bases),
            "beam_radii_um": tuple(float(v) / 1e-6 for v in t1_radii),
            "sector_rotation_deg": (0.0,),
        },
        "tier2": {
            "label": "ideal exploratory redesign candidate",
            "base_angles_deg": tuple(float(v) for v in tier2_base_angles_deg),
            "beam_radii_um": tuple(float(v) / 1e-6 for v in t2_radii),
            "sector_rotation_deg": tuple(float(v) for v in sector_rotation_deg),
        },
        "grid_n": int(grid_n),
        "z_planes": int(z_planes),
        "v0_ring_count": float(tmpl.metadata.get("v0_effective_ring_count", np.nan)),
        "inherited_mode1_old_mixed_plane_ring_count_estimate": float(existing["old_mixed_plane_ring_count"]),
        "inherited_mode1_ring_count_estimate": effective_ring_count_for_plane(
            radius_1e_field_m=current_w0,
            k_r_m_inv=float(existing["k_r_m_inv"]),
        ),
        "inherited_mode1_radius_plane_id": str(existing["radius_plane_id"]),
        "existing_defaults": existing,
    }
    outcome = mode1b_completion_gate(candidates, shortlist)
    return Mode1BSearchResult(
        target_template=tmpl,
        candidates=tuple(candidates),
        shortlist_candidates=shortlist,
        summary_rows=rows,
        outcome_report=outcome,
        search_space=search_space,
    )


def mode1b_completion_gate(
    candidates: Sequence[Mode1BCandidate],
    shortlist: Sequence[Mode1BCandidate] | None = None,
) -> dict[str, Any]:
    """Select the final MODE 1B outcome, separating visual success from realism."""

    all_candidates = tuple(candidates)
    passing = tuple(c for c in all_candidates if c.pass_hexagon_gate)
    realistic = tuple(
        c for c in passing
        if c.model_family == "actual_inherited_downstream"
        and c.feasibility.feasibility_class == "physically_plausible_existing_model"
    )
    exploratory = tuple(c for c in passing if c.feasibility.feasibility_class != "physically_plausible_existing_model")
    realistic_pool = tuple(
        c for c in all_candidates
        if c.tier == "tier1_existing_near_current" or c.feasibility.feasibility_class == "physically_plausible_existing_model"
    )
    exploratory_pool = tuple(
        c for c in all_candidates
        if c.tier == "tier2_exploratory_redesign" or c.feasibility.feasibility_class == "exploratory_high_angle_redesign"
    )
    any_hex = any(c.symmetry_class == "visual_hexagonal_field" for c in all_candidates)
    any_structured = any(c.symmetry_class in ("visual_hexagonal_field", "dark_core_structured_field") for c in all_candidates)
    best_realistic = max(realistic, key=_mode1b_score_for_ranking) if realistic else None
    best_exploratory = max(exploratory, key=_mode1b_score_for_ranking) if exploratory else None
    best_realistic_any = max(realistic_pool, key=_mode1b_score_for_ranking) if realistic_pool else None
    best_exploratory_any = max(exploratory_pool, key=_mode1b_score_for_ranking) if exploratory_pool else None
    best_any = max(all_candidates, key=_mode1b_score_for_ranking) if all_candidates else None

    if best_realistic is not None:
        outcome = "M1B-A-realistic"
        statement = (
            "A physically meaningful configuration inside the existing architecture produces a visually acceptable "
            "micro-scale version of the V0 hexagonal target. MODE 2A/2B may begin for that configuration."
        )
        mode2_allowed = True
    elif best_exploratory is not None:
        outcome = "M1B-A-exploratory"
        statement = (
            "An ideal exploratory redesign candidate produces a visual hexagon, but it requires optics/parameters "
            "outside the current inherited architecture or outside the validated thin/paraxial model. MODE 2A/2B "
            "for the current architecture remains blocked; redesign study may begin."
        )
        mode2_allowed = False
    elif any_hex or any_structured:
        outcome = "M1B-B"
        statement = (
            "Some candidates produce dark-core structured fields, but all remain triangular/lobed or non-hexagonal "
            "under the V0-template gate. MODE 2A/2B remains blocked."
        )
        mode2_allowed = False
    elif all_candidates:
        outcome = "M1B-D"
        statement = (
            "The tested downstream design space cannot produce the target from the ideal P2 field. Optical design "
            "must change before physical realisation."
        )
        mode2_allowed = False
    else:
        outcome = "M1B-C"
        statement = "Search inconclusive because no candidates were evaluated. MODE 2A/2B remains blocked."
        mode2_allowed = False

    def _brief(cand: Mode1BCandidate | None) -> Mapping[str, Any] | None:
        if cand is None:
            return None
        row = mode1b_candidate_row(cand)
        keep = (
            "candidate_id",
            "tier",
            "model_family",
            "radius_plane_id",
            "base_angle_deg",
            "beam_radius_um",
            "sector_rotation_deg",
            "ring_count",
            "symmetry_class",
            "pass_hexagon_gate",
            "angular_profile_correlation",
            "xy_correlation",
            "dark_core_ratio",
            "feasibility_class",
            "candidate_type",
            "fail_reasons",
        )
        return {k: row[k] for k in keep}

    shortlisted = tuple(shortlist or ())
    return {
        "stage": MODE1B_STAGE,
        "suggested_outcome": outcome,
        "outcome_statement": statement,
        "allowed_outcomes": MODE1B_ALLOWED_OUTCOMES,
        "mode2_realisation_allowed": bool(mode2_allowed),
        "mode2a_2b_gate": "open" if mode2_allowed else "blocked",
        "n_candidates": int(len(all_candidates)),
        "n_visual_hexagonal_class": int(sum(c.symmetry_class == "visual_hexagonal_field" for c in all_candidates)),
        "n_pass_hexagon_gate": int(len(passing)),
        "n_realistic_pass": int(len(realistic)),
        "n_exploratory_pass": int(len(exploratory)),
        "best_realistic_candidate": _brief(best_realistic_any),
        "best_exploratory_candidate": _brief(best_exploratory_any),
        "best_realistic_passing_candidate": _brief(best_realistic),
        "best_exploratory_passing_candidate": _brief(best_exploratory),
        "best_overall_candidate": _brief(best_any),
        "shortlist_ids": tuple(c.candidate_id for c in shortlisted),
        "note": (
            "A high-angle visual success is an ideal exploratory redesign candidate, not permission to start "
            "HWP/QWP/SLM realism for the current architecture."
        ),
    }


def mode1b_parameter_search(
    template: Mode1BTargetTemplate | None = None,
    *,
    base_angles_deg: Sequence[float] = (0.24, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0),
    beam_radii_m: Sequence[float] = (30e-6, 50e-6, 80e-6, 120e-6),
    grid_n_coarse: int = 224,
    grid_n_fine: int | None = None,
    z_planes: int = 13,
    max_shortlist: int = 8,
) -> dict[str, Any]:
    """Backward-compatible wrapper around the MODE 1B geometry search."""

    search = run_mode1b_geometry_search(
        template=template,
        grid_n=int(grid_n_coarse if grid_n_fine is None else grid_n_fine),
        z_planes=int(z_planes),
        tier1_base_angles_deg=(float(base_angles_deg[0]),),
        tier1_beam_radii_m=(float(beam_radii_m[0]),),
        tier2_base_angles_deg=tuple(float(v) for v in base_angles_deg[1:]),
        tier2_beam_radii_m=beam_radii_m,
        max_shortlist=int(max_shortlist),
    )
    return {
        "template": search.target_template,
        "candidates": search.candidates,
        "summary_rows": search.summary_rows,
        "shortlist_candidates": search.shortlist_candidates,
        "passing_candidates": tuple(c for c in search.candidates if c.pass_hexagon_gate),
        "completion": search.outcome_report,
        "search_result": search,
    }


def mode1b_scope_manifest(search: Mode1BSearchResult) -> dict[str, Any]:
    import datetime as _dt

    audit = audit_mode1b_ring_count_planes()
    return {
        "mode": "MODE 1B ideal downstream geometry target search",
        "stage": MODE1B_STAGE,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _mode1_git_commit_short(),
        "target": {
            "source": "validated V0 Nathan hexagonal Bessel total-intensity field",
            "classification": search.target_template.classification,
            "ring_radius_m": float(search.target_template.ring_radius_m),
            "dark_core_ratio": float(search.target_template.dark_core_ratio),
            "ring_count": float(search.target_template.metadata.get("v0_effective_ring_count", np.nan)),
        },
        "search_space": dict(search.search_space),
        "ring_count_plane_audit": asdict(audit),
        "simulated_components": (
            "ideal prescribed six-sector radial/azimuthal VectorField",
            "ideal free-space thin vector axicon candidates",
            "free-space vector ASM total-intensity output planes",
            "objective NA / SLM phase-period feasibility diagnostics",
        ),
        "bypassed_components": (
            "patterned HWP",
            "QWP",
            "SLM panel realism",
            "4F carrier and iris",
            "physical route generation",
        ),
        "claim_boundary": {
            "no_physical_hwp_qwp_slm_realisation": True,
            "visual_success_is_separate_from_feasibility": True,
            "mode2_realisation_allowed": bool(search.outcome_report.get("mode2_realisation_allowed", False)),
        },
        "outcome": dict(search.outcome_report),
    }


def plot_mode1b_target_template(
    template: Mode1BTargetTemplate,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    x_um = np.asarray(template.grid["x"], dtype=float) / 1e-6
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    axes[0].imshow(
        template.intensity_xy / max(float(np.nanmax(template.intensity_xy)), EPS),
        origin="lower",
        extent=ext,
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axes[0].set_title("V0 target template")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    theta = np.linspace(0.0, 360.0, int(template.angular_profile.size), endpoint=False)
    axes[1].plot(theta, template.angular_profile / max(float(np.nanmax(template.angular_profile)), EPS), color="k")
    axes[1].set_title("Angular ring profile")
    axes[1].set_xlabel("theta (deg)")
    axes[1].set_ylabel("norm. I")
    axis = np.linspace(-2.2, 2.2, int(template.x_profile.size))
    axes[2].plot(axis, template.x_profile, label="x", color="tab:blue")
    axes[2].plot(axis, template.y_profile, label="y", color="tab:orange", ls="--")
    axes[2].set_title("Normalised profiles")
    axes[2].set_xlabel("coordinate / ring radius")
    axes[2].set_ylabel("norm. I")
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"MODE 1B V0 target: {template.classification}, "
        f"ring={template.ring_radius_m/1e-6:.1f} um, core={template.dark_core_ratio:.4f}"
    )
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1b_candidate(
    candidate: Mode1BCandidate,
    template: Mode1BTargetTemplate,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    if candidate.intensity_xy is None:
        raise ValueError("candidate does not carry an intensity plane for plotting")
    x_um = np.asarray(candidate.grid["x"], dtype=float) / 1e-6
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    arr = np.asarray(candidate.intensity_xy, dtype=float)
    axes[0].imshow(arr / max(float(np.nanmax(arr)), EPS), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(candidate.candidate_id)
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    _, prof = angular_profile_on_ring(arr, candidate.grid, candidate.ring_radius_m, angular_bins=int(template.angular_profile.size))
    theta = np.linspace(0.0, 360.0, int(template.angular_profile.size), endpoint=False)
    axes[1].plot(theta, template.angular_profile / max(float(np.nanmax(template.angular_profile)), EPS), label="V0", color="k")
    axes[1].plot(theta, prof / max(float(np.nanmax(prof)), EPS), label="candidate", color="tab:red", alpha=0.8)
    axes[1].set_title(f"angular corr={candidate.template_score.angular_profile_correlation:.3f}")
    axes[1].set_xlabel("theta (deg)")
    axes[1].legend(fontsize=8)
    axes[2].set_axis_off()
    txt = [
        f"class: {candidate.symmetry_class}",
        f"gate pass: {candidate.pass_hexagon_gate}",
        f"fail: {'; '.join(candidate.fail_reasons) if candidate.fail_reasons else 'none'}",
        f"base angle: {candidate.base_angle_deg:.2f} deg",
        f"beam radius: {candidate.beam_radius_m/1e-6:.1f} um",
        f"ring count: {candidate.feasibility.ring_count:.2f}",
        f"feasibility: {candidate.feasibility.feasibility_class}",
        f"candidate type: {candidate.metadata.get('candidate_type', '')}",
        f"phase period: {candidate.feasibility.phase_period_m/1e-6:.2f} um",
        f"objective NA req: {candidate.feasibility.objective_na_required:.3f}",
    ]
    axes[2].text(0.02, 0.98, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=8.5)
    fig.suptitle("MODE 1B shortlist candidate")
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1b_current_inherited_failure(
    *,
    output_path: str | Path | None = None,
    grid_n: int = 96,
    z_planes: int = 9,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    cfg = NathanHexagonConfig.fast(grid_n=int(grid_n), z_planes=int(z_planes), angular_samples=360)
    result = run_mode1_ideal_p2_downstream(cfg, run_f2=False, run_centre_treatment=False)
    grid = _mode1_focal_grid(result.f0)
    plane = np.asarray(result.f0.intensity_stack[int(result.reference_index)], dtype=float)
    row = _mode1_plane_row(plane, grid)
    x_um = _mode1_um_axis(grid)
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)
    axes[0].imshow(plane / max(float(np.nanmax(plane)), EPS), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("Inherited MODE 1 reference")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[1].set_axis_off()
    txt = [
        f"class: {row['symmetry_class']}",
        f"dark-core ratio: {row['dark_core_ratio']:.3f}",
        f"order3/order6: {row['order3_over_order6']:.3f}",
        f"c120-c60: {row['c120_minus_c60']:.3f}",
        f"triangular veto: {row['symmetry_class'] != 'visual_hexagonal_field'}",
        "MODE 2A/2B: blocked",
    ]
    axes[1].text(0.02, 0.98, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=9)
    fig.suptitle("MODE 1B current inherited failure control: triangular/C3 is not the target")
    _save_fig(fig, output_path)
    return fig, axes


def write_mode1b_geometry_search_outputs(
    search: Mode1BSearchResult,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode1b_geometry_search",
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "target_template": out / "mode1b_target_template.png",
        "current_inherited_failure": out / "mode1b_current_inherited_failure.png",
        "summary_csv": out / "mode1b_search_summary.csv",
        "summary_json": out / "mode1b_search_summary.json",
        "outcome_report": out / "mode1b_outcome_report.json",
        "scope_manifest": out / "simulation_scope_manifest.json",
    }
    audit = audit_mode1b_ring_count_planes()
    audit_paths = write_mode1b_ring_count_audit(audit, out)
    paths["ring_count_audit_csv"] = audit_paths["csv"]
    paths["ring_count_audit_json"] = audit_paths["json"]
    fig, _ = plot_mode1b_target_template(search.target_template, output_path=paths["target_template"])
    plt.close(fig)
    fig, _ = plot_mode1b_current_inherited_failure(output_path=paths["current_inherited_failure"])
    plt.close(fig)
    _write_rows(paths["summary_csv"], search.summary_rows)
    paths["summary_json"].write_text(json.dumps(_json_ready(search.summary_rows), indent=2), encoding="utf-8")
    report = dict(search.outcome_report)
    report["search_space"] = dict(search.search_space)
    report["ring_count_plane_audit"] = asdict(audit)
    paths["outcome_report"].write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    paths["scope_manifest"].write_text(json.dumps(_json_ready(mode1b_scope_manifest(search)), indent=2), encoding="utf-8")
    for cand in search.shortlist_candidates:
        p = out / f"mode1b_shortlist_{cand.candidate_id}.png"
        fig, _ = plot_mode1b_candidate(cand, search.target_template, output_path=p)
        plt.close(fig)
        paths[f"shortlist_{cand.candidate_id}"] = p
    return paths


# ===========================================================================
# MODE 1C - k-space / aperture / NA feasibility audit
# ===========================================================================

MODE1C_STAGE = "nathan_mode1c_kr_aperture_feasibility"
MODE1C_ALLOWED_OUTCOMES = ("M1C-A", "M1C-B", "M1C-C", "M1C-D")


@dataclass(frozen=True)
class Mode1CKMapping:
    k0_m_inv: float
    wavelength_m: float
    k_r_pre_m_inv: float
    k_r_surface_m_inv: float
    k_scale_surface_over_pre: float
    objective_na: float
    k_r_surface_na_limit_m_inv: float
    k_r_pre_na_limited_m_inv: float
    current_surface_na_fraction: float
    current_pre_phase_period_m: float
    current_surface_radial_period_m: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Mode1CApertureRingLimit:
    p2_radius_current_m: float
    p2_radius_max_slm_short_axis_m: float
    p2_radius_max_with_safety_m: float
    k_r_pre_current_m_inv: float
    k_r_pre_na_limited_m_inv: float
    ring_count_current: float
    ring_count_max_current_radius_na_limited: float
    ring_count_max_slm_radius_current_kr: float
    ring_count_max_slm_radius_na_limited: float
    v0_ring_count: float
    fraction_of_v0_current: float
    fraction_of_v0_max_possible: float
    conclusion: str


@dataclass(frozen=True)
class Mode1CConstrainedCandidate:
    target_ring_count: float
    p2_radius_m: float
    k_r_pre_m_inv: float
    k_r_surface_m_inv: float
    surface_na_required: float
    within_objective_na: bool
    within_slm_aperture: bool
    phase_period_m: float
    slm_encoded_phase_ok: bool
    run_status: str
    symmetry_class: str | None
    template_gate_pass: bool | None
    failure_reason: tuple[str, ...]


def audit_mode1c_kr_mapping(config: NathanHexagonConfig | None = None) -> Mode1CKMapping:
    """Audit current pre-axicon, surface, objective-NA, and phase-period mapping."""

    cfg = config or NathanHexagonConfig.fast()
    twin = cfg.twin
    params = resolve_vector_axicon_parameters(twin)
    k0 = float(2.0 * np.pi / twin.laser.wavelength_m)
    k_pre = float(params.k_r_pre_m_inv)
    k_surface = float(params.k_r_surface_m_inv)
    scale = float(k_surface / max(k_pre, EPS))
    k_surface_limit = float(twin.objective.NA * k0)
    k_pre_limit = float(k_surface_limit / max(scale, EPS))
    notes = [
        "physical_axicon.axicon_base_angle_deg cleanly overrides pre-axicon k_r only",
        "surface k_r is target/design-derived from TwinConfig.target and the ObjectiveMap",
        "a high pre-axicon k_r proxy must still satisfy the surface objective NA after mapping",
    ]
    if k_surface > k_surface_limit:
        notes.append("current surface k_r exceeds objective NA")
    return Mode1CKMapping(
        k0_m_inv=k0,
        wavelength_m=float(twin.laser.wavelength_m),
        k_r_pre_m_inv=k_pre,
        k_r_surface_m_inv=k_surface,
        k_scale_surface_over_pre=scale,
        objective_na=float(twin.objective.NA),
        k_r_surface_na_limit_m_inv=k_surface_limit,
        k_r_pre_na_limited_m_inv=k_pre_limit,
        current_surface_na_fraction=float(k_surface / max(k_surface_limit, EPS)),
        current_pre_phase_period_m=float(2.0 * np.pi / max(abs(k_pre), EPS)),
        current_surface_radial_period_m=float(2.0 * np.pi / max(abs(k_surface), EPS)),
        notes=tuple(notes),
    )


def _mode1c_current_p2_radius(config: NathanHexagonConfig) -> float:
    grid = default_nathan_grid(config)
    field = canonical_target_field(config, grid=grid)
    return one_over_e_field_radius_from_vector_field(field, grid)


def audit_mode1c_aperture_ring_limit(
    config: NathanHexagonConfig | None = None,
    *,
    safety: float = 0.90,
) -> Mode1CApertureRingLimit:
    """Audit maximum plane-correct ring count allowed by P2 size, SLM aperture, and NA."""

    cfg = config or NathanHexagonConfig.fast()
    twin = cfg.twin
    mapping = audit_mode1c_kr_mapping(cfg)
    p2_radius = _mode1c_current_p2_radius(cfg)
    active_w = float(twin.slm.resolution_x) * float(twin.slm.pixel_pitch_m)
    active_h = float(twin.slm.resolution_y) * float(twin.slm.pixel_pitch_m)
    r_slm_short = 0.5 * min(active_w, active_h)
    r_slm_safe = float(safety) * r_slm_short
    v0_cfg = NathanSourceParityConfig()
    v0_rings = effective_ring_count_for_plane(radius_1e_field_m=v0_cfg.beam_radius_m, k_r_m_inv=v0_cfg.k_r_m_inv)
    n_current = effective_ring_count_for_plane(radius_1e_field_m=p2_radius, k_r_m_inv=mapping.k_r_pre_m_inv)
    n_current_radius_na = effective_ring_count_for_plane(radius_1e_field_m=p2_radius, k_r_m_inv=mapping.k_r_pre_na_limited_m_inv)
    n_slm_current = effective_ring_count_for_plane(radius_1e_field_m=r_slm_safe, k_r_m_inv=mapping.k_r_pre_m_inv)
    n_slm_na = effective_ring_count_for_plane(radius_1e_field_m=r_slm_safe, k_r_m_inv=mapping.k_r_pre_na_limited_m_inv)
    if n_slm_na < 0.5 * v0_rings:
        conclusion = "current aperture/NA cannot reach V0-like ring count"
    elif n_slm_na < v0_rings:
        conclusion = "current aperture/NA reaches only partial V0 ring count"
    else:
        conclusion = "current aperture/NA can in principle reach V0 ring count"
    return Mode1CApertureRingLimit(
        p2_radius_current_m=float(p2_radius),
        p2_radius_max_slm_short_axis_m=float(r_slm_short),
        p2_radius_max_with_safety_m=float(r_slm_safe),
        k_r_pre_current_m_inv=float(mapping.k_r_pre_m_inv),
        k_r_pre_na_limited_m_inv=float(mapping.k_r_pre_na_limited_m_inv),
        ring_count_current=float(n_current),
        ring_count_max_current_radius_na_limited=float(n_current_radius_na),
        ring_count_max_slm_radius_current_kr=float(n_slm_current),
        ring_count_max_slm_radius_na_limited=float(n_slm_na),
        v0_ring_count=float(v0_rings),
        fraction_of_v0_current=float(n_current / max(v0_rings, EPS)),
        fraction_of_v0_max_possible=float(n_slm_na / max(v0_rings, EPS)),
        conclusion=conclusion,
    )


def make_mode1c_twin_with_axicon_base_angle(twin: TwinConfig, base_angle_deg: float) -> TwinConfig:
    """Return a copy of TwinConfig with only the physical axicon base angle changed."""

    return replace(
        twin,
        physical_axicon=replace(twin.physical_axicon, axicon_base_angle_deg=float(base_angle_deg)),
    )


def make_mode1c_twin_with_target_kr(twin: TwinConfig, target_k_r_surface_m_inv: float) -> TwinConfig:
    """Return a target-design copy whose equivalent ell=0 target implies ``target_k_r_surface_m_inv``.

    The existing MODE 1 downstream path contains a locked surface-k_r fingerprint,
    so this helper is legitimate design plumbing but cannot be used as an
    unqualified current-architecture run without also revisiting that tripwire.
    """

    kr = max(float(target_k_r_surface_m_inv), EPS)
    target_core_diameter_m = float(2.0 * J0_FIRST_ZERO / kr)
    return replace(twin, target=replace(twin.target, target_core_diameter_m=target_core_diameter_m))


def _mode1c_base_angle_for_pre_kr(*, twin: TwinConfig, k_r_pre_m_inv: float) -> float:
    n_axicon = float(twin.target.n_axicon if twin.physical_axicon.n_axicon is None else twin.physical_axicon.n_axicon)
    n_medium = float(getattr(twin.physical_axicon, "axicon_medium_n", twin.target.hologram_medium_n))
    denom = float(twin.laser.k0 * (n_axicon - n_medium))
    return float(np.rad2deg(np.arctan(float(k_r_pre_m_inv) / max(denom, EPS))))


def mode1c_candidate_row(cand: Mode1CConstrainedCandidate) -> dict[str, Any]:
    return {
        "target_ring_count": float(cand.target_ring_count),
        "p2_radius_um": float(cand.p2_radius_m / 1e-6),
        "k_r_pre_m_inv": float(cand.k_r_pre_m_inv),
        "k_r_surface_m_inv": float(cand.k_r_surface_m_inv),
        "surface_na_required": float(cand.surface_na_required),
        "within_objective_na": bool(cand.within_objective_na),
        "within_slm_aperture": bool(cand.within_slm_aperture),
        "phase_period_um": float(cand.phase_period_m / 1e-6),
        "slm_encoded_phase_ok": bool(cand.slm_encoded_phase_ok),
        "run_status": str(cand.run_status),
        "symmetry_class": "" if cand.symmetry_class is None else str(cand.symmetry_class),
        "template_gate_pass": "" if cand.template_gate_pass is None else bool(cand.template_gate_pass),
        "failure_reason": "; ".join(cand.failure_reason),
    }


def run_mode1c_constrained_search(
    template: Mode1BTargetTemplate | None = None,
    *,
    config: NathanHexagonConfig | None = None,
    target_ring_counts: Sequence[float] = (4, 6, 8, 10, 12, 16, 20, 24, 31),
    p2_radius_options_m: Sequence[float] | None = None,
    slm_safety: float = 0.90,
    grid_n_proxy: int = 160,
    z_planes_proxy: int = 9,
    run_proxy: bool = True,
) -> tuple[Mode1CConstrainedCandidate, ...]:
    """Evaluate architecture-constrained ring-count targets.

    Feasible candidates are run only as plane-corrected free-space proxies.  The
    actual inherited target-k_r override is not run because the current MODE 1
    path locks the surface k_r fingerprint to the inherited design.
    """

    cfg = config or NathanHexagonConfig.fast()
    twin = cfg.twin
    tmpl = template or build_mode1b_target_template(grid_n=384, z_planes=21)
    mapping = audit_mode1c_kr_mapping(cfg)
    limit = audit_mode1c_aperture_ring_limit(cfg, safety=slm_safety)
    radius_options = tuple(p2_radius_options_m) if p2_radius_options_m is not None else (
        limit.p2_radius_current_m,
        0.75 * limit.p2_radius_max_with_safety_m,
        limit.p2_radius_max_with_safety_m,
    )
    rows: list[Mode1CConstrainedCandidate] = []
    min_slm_phase_period = 4.0 * float(twin.slm.pixel_pitch_m)
    for radius in radius_options:
        for n_ring in target_ring_counts:
            r = float(radius)
            target = float(n_ring)
            k_pre = float(2.0 * np.pi * target / max(r, EPS))
            k_surface = float(k_pre * mapping.k_scale_surface_over_pre)
            na_required = float(k_surface / max(mapping.k0_m_inv, EPS))
            within_na = bool(na_required <= float(mapping.objective_na) + 1e-15)
            within_aperture = bool(r <= limit.p2_radius_max_with_safety_m + 1e-15)
            phase_period = float(2.0 * np.pi / max(abs(k_pre), EPS))
            slm_ok = bool(phase_period >= min_slm_phase_period)
            failures: list[str] = []
            if not within_na:
                failures.append("surface NA exceeds objective NA")
            if not within_aperture:
                failures.append("P2 radius exceeds SLM-safe aperture")
            if not slm_ok:
                failures.append("pre-axicon phase period too fine for 4-pixel SLM encoding")

            run_status = "not_run"
            symmetry_class: str | None = None
            pass_gate: bool | None = None
            if failures:
                run_status = "infeasible_not_run"
                pass_gate = False
            elif run_proxy:
                beta_deg = _mode1c_base_angle_for_pre_kr(twin=twin, k_r_pre_m_inv=k_pre)
                proxy = run_mode1b_candidate(
                    tmpl,
                    base_angle_deg=beta_deg,
                    beam_radius_m=r,
                    tier="mode1c_constrained_proxy",
                    candidate_id=f"mode1c_proxy_N{target:.1f}_R{r/1e-6:.1f}um".replace(".", "p"),
                    grid_n=int(grid_n_proxy),
                    z_planes=int(z_planes_proxy),
                    wavelength_m=float(twin.laser.wavelength_m),
                    n_axicon=float(twin.target.n_axicon if twin.physical_axicon.n_axicon is None else twin.physical_axicon.n_axicon),
                    n_medium=float(twin.physical_axicon.axicon_medium_n),
                    objective_na=float(twin.objective.NA),
                    slm_pixel_pitch_m=float(twin.slm.pixel_pitch_m),
                    model_family="plane_corrected_free_space_continuation",
                    radius_plane_id="mode1c_candidate_p2_radius",
                )
                symmetry_class = proxy.symmetry_class
                pass_gate = bool(proxy.pass_hexagon_gate)
                run_status = "proxy_only_run"
                failures.extend(proxy.fail_reasons)
                failures.append("actual inherited target-k_r override not run because MODE 1 locks surface k_r fingerprint")
            else:
                run_status = "feasible_proxy_not_requested"
                pass_gate = None

            rows.append(
                Mode1CConstrainedCandidate(
                    target_ring_count=target,
                    p2_radius_m=r,
                    k_r_pre_m_inv=k_pre,
                    k_r_surface_m_inv=k_surface,
                    surface_na_required=na_required,
                    within_objective_na=within_na,
                    within_slm_aperture=within_aperture,
                    phase_period_m=phase_period,
                    slm_encoded_phase_ok=slm_ok,
                    run_status=run_status,
                    symmetry_class=symmetry_class,
                    template_gate_pass=pass_gate,
                    failure_reason=tuple(failures),
                )
            )
    return tuple(rows)


def mode1c_outcome_report(
    *,
    mapping: Mode1CKMapping,
    aperture: Mode1CApertureRingLimit,
    candidates: Sequence[Mode1CConstrainedCandidate],
) -> dict[str, Any]:
    actual_pass = any(c.run_status.startswith("actual") and c.template_gate_pass is True for c in candidates)
    proxy_pass = any(c.run_status == "proxy_only_run" and c.template_gate_pass is True for c in candidates)
    theoretical_budget = aperture.ring_count_max_slm_radius_na_limited >= aperture.v0_ring_count
    if actual_pass:
        outcome = "M1C-A"
        statement = "Current architecture reaches a V0-like ring-count regime and an actual inherited candidate passes."
        mode2_allowed = True
    elif theoretical_budget:
        outcome = "M1C-B"
        statement = "Current architecture has enough theoretical ring-count budget, but no actual inherited candidate passed."
        mode2_allowed = False
    else:
        outcome = "M1C-C"
        statement = (
            "Current architecture cannot reach the required V0-like ring-count regime within objective NA, "
            "SLM aperture, and inherited k_r mapping. MODE 2A/2B remains blocked; optical redesign required."
        )
        mode2_allowed = False
    return {
        "stage": MODE1C_STAGE,
        "suggested_outcome": outcome,
        "allowed_outcomes": MODE1C_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "mode2_realisation_allowed": bool(mode2_allowed),
        "mode2a_2b_gate": "open" if mode2_allowed else "blocked",
        "current_p2_radius_m": float(aperture.p2_radius_current_m),
        "slm_safe_radius_m": float(aperture.p2_radius_max_with_safety_m),
        "current_k_r_pre_m_inv": float(mapping.k_r_pre_m_inv),
        "current_k_r_surface_m_inv": float(mapping.k_r_surface_m_inv),
        "k_scale_surface_over_pre": float(mapping.k_scale_surface_over_pre),
        "objective_na": float(mapping.objective_na),
        "objective_surface_k_limit_m_inv": float(mapping.k_r_surface_na_limit_m_inv),
        "max_ring_count_current_radius_na_limited": float(aperture.ring_count_max_current_radius_na_limited),
        "max_ring_count_slm_safe_na_limited": float(aperture.ring_count_max_slm_radius_na_limited),
        "v0_ring_count": float(aperture.v0_ring_count),
        "n_candidates": int(len(tuple(candidates))),
        "n_within_na": int(sum(c.within_objective_na for c in candidates)),
        "n_proxy_runs": int(sum(c.run_status == "proxy_only_run" for c in candidates)),
        "n_proxy_pass": int(sum(c.run_status == "proxy_only_run" and c.template_gate_pass is True for c in candidates)),
        "n_actual_pass": int(sum(c.run_status.startswith("actual") and c.template_gate_pass is True for c in candidates)),
        "proxy_pass_found": bool(proxy_pass),
        "note": "MODE 1C is ideal downstream only; proxy passes do not authorise HWP/QWP/SLM realisation.",
    }


def plot_mode1c_feasibility(
    aperture: Mode1CApertureRingLimit,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    labels = ["current", "current+NA", "SLM safe+current k", "SLM safe+NA", "V0"]
    values = [
        aperture.ring_count_current,
        aperture.ring_count_max_current_radius_na_limited,
        aperture.ring_count_max_slm_radius_current_kr,
        aperture.ring_count_max_slm_radius_na_limited,
        aperture.v0_ring_count,
    ]
    colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:red", "0.25"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.bar(labels, values, color=colors)
    ax.axhline(aperture.v0_ring_count, color="0.2", lw=1.0, ls="--")
    ax.set_ylabel("effective ring count")
    ax.set_title("MODE 1C aperture / objective-NA ring-count budget")
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    _save_fig(fig, output_path)
    return fig, ax


def write_mode1c_outputs(
    *,
    mapping: Mode1CKMapping,
    aperture: Mode1CApertureRingLimit,
    candidates: Sequence[Mode1CConstrainedCandidate],
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode1c_kr_aperture",
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = tuple(mode1c_candidate_row(c) for c in candidates)
    outcome = mode1c_outcome_report(mapping=mapping, aperture=aperture, candidates=candidates)
    paths = {
        "kr_mapping": out / "mode1c_kr_mapping.json",
        "aperture_ring_limit": out / "mode1c_aperture_ring_limit.json",
        "constrained_search_csv": out / "mode1c_constrained_search.csv",
        "constrained_search_json": out / "mode1c_constrained_search.json",
        "feasibility_plot": out / "mode1c_feasibility_plot.png",
        "outcome_report": out / "mode1c_outcome_report.json",
    }
    paths["kr_mapping"].write_text(json.dumps(_json_ready(asdict(mapping)), indent=2), encoding="utf-8")
    paths["aperture_ring_limit"].write_text(json.dumps(_json_ready(asdict(aperture)), indent=2), encoding="utf-8")
    _write_rows(paths["constrained_search_csv"], rows)
    paths["constrained_search_json"].write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")
    paths["outcome_report"].write_text(json.dumps(_json_ready(outcome), indent=2), encoding="utf-8")
    fig, _ = plot_mode1c_feasibility(aperture, output_path=paths["feasibility_plot"])
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# MODE 1D: inverse optical redesign requirements
# ---------------------------------------------------------------------------

MODE1D_STAGE = "nathan_mode1d_inverse_redesign_requirements"
MODE1D_ALLOWED_OUTCOMES = ("M1D-A", "M1D-B", "M1D-C", "M1D-D")
MODE1D_DEFAULT_RING_SWEEP = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 31.0)
MODE1D_ACCEPTABLE_TEMPLATE_CORRELATION = 0.55
MODE1D_ACCEPTABLE_XY_CORRELATION = MODE1B_XY_CORRELATION_PASS
MODE1D_ACCEPTABLE_DARK_CORE_RATIO = MODE1_DARK_CORE_HOLLOW_THRESHOLD
MODE1D_SCOPE_STATEMENT = (
    "MODE 1D inverse redesign audit only: analytic downstream k-space/NA/aperture requirements "
    "plus validated source-style lower-ring sweep. No HWP/QWP/SLM panel realism and no MODE 2A/2B "
    "physical-route approval."
)


@dataclass(frozen=True)
class Mode1DDesignContext:
    wavelength_m: float
    k0_m_inv: float
    current_objective_na: float
    current_mapping_factor: float
    current_p2_radius_m: float
    slm_safe_radius_m: float
    slm_half_short_axis_radius_m: float
    slm_half_long_axis_radius_m: float
    current_pre_kr_m_inv: float
    current_surface_kr_m_inv: float
    current_pre_kr_na_limited_m_inv: float
    v0_ring_count: float
    mode1c_current_ring_count: float
    mode1c_max_current_radius_na_limited: float
    mode1c_max_slm_safe_na_limited: float


@dataclass(frozen=True)
class Mode1DSourceSweepCase:
    ring_count_target: float
    ring_count_actual: float
    k_r_m_inv: float
    axicon_base_angle_deg: float
    axicon_apex_angle_deg: float
    z_reference_m: float
    grid_n: int
    classification: str
    accepted_hexagon: bool
    dark_core_ratio: float
    template_score: Mode1BTemplateScore
    symmetry: Mapping[str, Any]
    intensity_xy: np.ndarray
    xz_map: np.ndarray
    grid: Mapping[str, Any]
    z_values_m: np.ndarray


def required_pre_kr_for_ring_count(ring_count: float, radius_m: float) -> float:
    return float(2.0 * np.pi * float(ring_count) / max(float(radius_m), EPS))


def required_surface_kr(pre_kr: float, k_scale_surface_over_pre: float) -> float:
    return float(float(pre_kr) * float(k_scale_surface_over_pre))


def required_na(surface_kr: float, wavelength_m: float) -> float:
    k0 = 2.0 * np.pi / float(wavelength_m)
    return float(float(surface_kr) / max(k0, EPS))


def required_radius_for_ring_count(ring_count: float, pre_kr: float) -> float:
    return float(2.0 * np.pi * float(ring_count) / max(float(pre_kr), EPS))


def required_mapping_for_ring_count(
    *,
    ring_count: float,
    radius_m: float,
    wavelength_m: float,
    objective_na: float,
) -> float:
    k0 = 2.0 * np.pi / float(wavelength_m)
    pre_kr = required_pre_kr_for_ring_count(ring_count, radius_m)
    max_surface_kr = float(objective_na) * k0
    return float(max_surface_kr / max(pre_kr, EPS))


def mode1d_design_context(config: NathanHexagonConfig | None = None) -> Mode1DDesignContext:
    cfg = config or NathanHexagonConfig.fast()
    mapping = audit_mode1c_kr_mapping(cfg)
    aperture = audit_mode1c_aperture_ring_limit(cfg)
    twin = cfg.twin
    active_w = float(twin.slm.resolution_x) * float(twin.slm.pixel_pitch_m)
    active_h = float(twin.slm.resolution_y) * float(twin.slm.pixel_pitch_m)
    return Mode1DDesignContext(
        wavelength_m=float(mapping.wavelength_m),
        k0_m_inv=float(mapping.k0_m_inv),
        current_objective_na=float(mapping.objective_na),
        current_mapping_factor=float(mapping.k_scale_surface_over_pre),
        current_p2_radius_m=float(aperture.p2_radius_current_m),
        slm_safe_radius_m=float(aperture.p2_radius_max_with_safety_m),
        slm_half_short_axis_radius_m=float(aperture.p2_radius_max_slm_short_axis_m),
        slm_half_long_axis_radius_m=0.5 * max(active_w, active_h),
        current_pre_kr_m_inv=float(mapping.k_r_pre_m_inv),
        current_surface_kr_m_inv=float(mapping.k_r_surface_m_inv),
        current_pre_kr_na_limited_m_inv=float(mapping.k_r_pre_na_limited_m_inv),
        v0_ring_count=float(aperture.v0_ring_count),
        mode1c_current_ring_count=float(aperture.ring_count_current),
        mode1c_max_current_radius_na_limited=float(aperture.ring_count_max_current_radius_na_limited),
        mode1c_max_slm_safe_na_limited=float(aperture.ring_count_max_slm_radius_na_limited),
    )


def _mode1d_radius_comment(required_na_value: float) -> str:
    na = float(required_na_value)
    if na <= 0.45:
        return "within current objective NA"
    if na <= 0.70:
        return "requires moderate high-NA redesign"
    if na <= 0.95:
        return "requires aggressive high-NA air redesign"
    if na <= 1.20:
        return "requires immersion-class or exploratory NA"
    return "not realistic for this architecture"


def mode1d_required_na_table(context: Mode1DDesignContext | None = None) -> tuple[dict[str, Any], ...]:
    ctx = context or mode1d_design_context()
    rows: list[dict[str, Any]] = []
    radius_rows = (
        ("current_p2_radius", ctx.current_p2_radius_m),
        ("0p75_slm_safe_radius", 0.75 * ctx.slm_safe_radius_m),
        ("slm_safe_radius", ctx.slm_safe_radius_m),
        ("full_half_short_axis_slm_radius", ctx.slm_half_short_axis_radius_m),
    )
    for label, radius in radius_rows:
        pre = required_pre_kr_for_ring_count(ctx.v0_ring_count, radius)
        surface = required_surface_kr(pre, ctx.current_mapping_factor)
        na = required_na(surface, ctx.wavelength_m)
        rows.append(
            {
                "radius_case": label,
                "radius_m": float(radius),
                "radius_um": float(radius / 1e-6),
                "required_pre_kr_m_inv": float(pre),
                "required_surface_kr_m_inv": float(surface),
                "required_NA_for_V0_ring_count": float(na),
                "feasible_with_air_objective": bool(na <= 0.95),
                "feasible_with_high_NA_immersion": bool(na <= 1.20),
                "comment": _mode1d_radius_comment(na),
            }
        )
    return tuple(rows)


def mode1d_required_radius_table(
    context: Mode1DDesignContext | None = None,
    *,
    na_values: Sequence[float] = (0.45, 0.7, 0.9, 1.0, 1.2),
) -> tuple[dict[str, Any], ...]:
    ctx = context or mode1d_design_context()
    rows: list[dict[str, Any]] = []
    for na in na_values:
        max_surface = float(na) * ctx.k0_m_inv
        max_pre = max_surface / max(ctx.current_mapping_factor, EPS)
        radius = required_radius_for_ring_count(ctx.v0_ring_count, max_pre)
        rows.append(
            {
                "NA": float(na),
                "max_surface_kr_m_inv": float(max_surface),
                "max_pre_kr_under_current_mapping_m_inv": float(max_pre),
                "required_radius_for_V0_ring_count_m": float(radius),
                "required_radius_for_V0_ring_count_um": float(radius / 1e-6),
                "required_diameter_m": float(2.0 * radius),
                "required_diameter_um": float(2.0 * radius / 1e-6),
                "fits_SLM_short_axis": bool(radius <= ctx.slm_half_short_axis_radius_m),
                "fits_SLM_long_axis": bool(radius <= ctx.slm_half_long_axis_radius_m),
                "comment": (
                    "fits SLM but current mapping still requires this NA"
                    if radius <= ctx.slm_half_short_axis_radius_m
                    else "requires a larger P2 beam/aperture than the SLM short axis"
                ),
            }
        )
    return tuple(rows)


def mode1d_required_mapping_table(
    context: Mode1DDesignContext | None = None,
    *,
    na_values: Sequence[float] = (0.45, 0.7, 0.9),
) -> tuple[dict[str, Any], ...]:
    ctx = context or mode1d_design_context()
    rows: list[dict[str, Any]] = []
    for label, radius in (("current_p2_radius", ctx.current_p2_radius_m), ("slm_safe_radius", ctx.slm_safe_radius_m)):
        row: dict[str, Any] = {
            "radius_case": label,
            "radius_m": float(radius),
            "radius_um": float(radius / 1e-6),
            "current_mapping_factor": float(ctx.current_mapping_factor),
        }
        required_current_na = None
        for na in na_values:
            req = required_mapping_for_ring_count(
                ring_count=ctx.v0_ring_count,
                radius_m=radius,
                wavelength_m=ctx.wavelength_m,
                objective_na=float(na),
            )
            key = str(na).replace(".", "p")
            row[f"required_mapping_factor_for_V0_at_NA_{key}"] = float(req)
            if abs(float(na) - ctx.current_objective_na) < 1.0e-12:
                required_current_na = req
        if required_current_na is None:
            required_current_na = required_mapping_for_ring_count(
                ring_count=ctx.v0_ring_count,
                radius_m=radius,
                wavelength_m=ctx.wavelength_m,
                objective_na=ctx.current_objective_na,
            )
        row["reduction_factor_needed_at_current_NA"] = float(ctx.current_mapping_factor / max(required_current_na, EPS))
        row["comment"] = (
            "mapping/demagnification must be reduced substantially"
            if row["reduction_factor_needed_at_current_NA"] > 2.0
            else "mapping reduction is modest"
        )
        rows.append(row)
    return tuple(rows)


def mode1d_achievable_ring_count_table(
    context: Mode1DDesignContext | None = None,
    *,
    na_values: Sequence[float] = (0.45, 0.7, 0.9, 1.0, 1.2),
) -> tuple[dict[str, Any], ...]:
    ctx = context or mode1d_design_context()
    cases: list[tuple[str, float, float]] = [("current_NA_current_radius", ctx.current_objective_na, ctx.current_p2_radius_m)]
    for na in na_values:
        if abs(float(na) - ctx.current_objective_na) < 1.0e-12:
            cases.append(("current_NA_slm_safe_radius", float(na), ctx.slm_safe_radius_m))
        else:
            key = str(float(na)).replace(".", "p")
            cases.append((f"NA_{key}_slm_safe_radius", float(na), ctx.slm_safe_radius_m))
    rows: list[dict[str, Any]] = []
    for label, na, radius in cases:
        max_surface = float(na) * ctx.k0_m_inv
        max_pre = max_surface / max(ctx.current_mapping_factor, EPS)
        rings = effective_ring_count_for_plane(radius_1e_field_m=radius, k_r_m_inv=max_pre)
        fraction = rings / max(ctx.v0_ring_count, EPS)
        if fraction < 0.45:
            regime = "low_ring_triangular"
        elif fraction < 0.85:
            regime = "intermediate_uncertain"
        else:
            regime = "V0_like_possible"
        rows.append(
            {
                "case": label,
                "NA": float(na),
                "radius_m": float(radius),
                "radius_um": float(radius / 1e-6),
                "max_surface_kr_m_inv": float(max_surface),
                "max_pre_kr_m_inv": float(max_pre),
                "ring_count": float(rings),
                "fraction_of_V0": float(fraction),
                "likely_regime": regime,
            }
        )
    return tuple(rows)


def _mode1d_source_base_angle_for_ring_count(ring_count: float, config: NathanSourceParityConfig) -> float:
    k_pre = required_pre_kr_for_ring_count(float(ring_count), float(config.beam_radius_m))
    denom = (2.0 * np.pi / float(config.wavelength_m)) * (float(config.axicon_n) - float(config.medium_n))
    return float(np.arctan(k_pre / max(denom, EPS)))


def _mode1d_source_case_config(
    ring_count: float,
    *,
    base_config: NathanSourceParityConfig | None = None,
    grid_n: int = 384,
    z_planes: int = 21,
) -> NathanSourceParityConfig:
    base = base_config or NathanSourceParityConfig()
    base_angle = _mode1d_source_base_angle_for_ring_count(float(ring_count), base)
    z_ref = float(base.beam_radius_m) / max(float(np.tan(base_angle)), 1.0e-9)
    return replace(
        base,
        grid_n=int(grid_n),
        z_planes=int(z_planes),
        axicon_apex_angle_deg=float(180.0 - 2.0 * np.rad2deg(base_angle)),
        z_reference_m=z_ref,
        z_start_m=0.05 * z_ref,
        z_end_m=2.25 * z_ref,
        z_span_m=None,
    )


def run_mode1d_source_ring_count_case(
    ring_count: float,
    template: Mode1BTargetTemplate | None = None,
    *,
    base_config: NathanSourceParityConfig | None = None,
    grid_n: int = 384,
    z_planes: int = 21,
) -> Mode1DSourceSweepCase:
    """Run one validated-source-style lower-ring-count candidate."""

    cfg = _mode1d_source_case_config(
        float(ring_count),
        base_config=base_config,
        grid_n=int(grid_n),
        z_planes=int(z_planes),
    )
    tmpl = template or build_mode1b_target_template(grid_n=int(grid_n), z_planes=max(21, int(z_planes)))
    grid = source_parity_grid(cfg)
    field, _ = nathan_literal_segmented_ra_input(
        grid,
        wavelength_m=float(cfg.wavelength_m),
        beam_radius_m=float(cfg.beam_radius_m),
        n_pairs=int(cfg.n_pairs),
        sector_theta_rad=float(cfg.sector_theta_rad),
        sector_rotation_rad=float(cfg.sector_rotation_rad),
    )
    after, _ = _apply_free_space_vector_axicon(
        field,
        n_axicon=float(cfg.axicon_n),
        n_medium=float(cfg.medium_n),
        base_angle_rad=float(cfg.axicon_base_angle_rad),
    )
    z_values = _v0_z_values(cfg)
    stack, _ = _free_space_intensity_stack(after, z_values)
    ref = _nearest_z_index(z_values, cfg.z_reference_m)
    plane = np.asarray(stack[ref], dtype=float)
    diag = _v0_plane_diagnostics(plane, grid)
    ring_radius = float(diag["ring_radius_m"])
    dark = float(diag["central_core_darkness"])
    sym = _mode1_symmetry(plane, grid, ring_radius)
    cls = mode1_symmetry_class(sym, dark)
    score = compare_to_v0_template(plane, grid, tmpl, candidate_ring_radius_m=ring_radius)
    accepted = bool(
        cls == "visual_hexagonal_field"
        and dark <= MODE1D_ACCEPTABLE_DARK_CORE_RATIO
        and float(score.angular_profile_correlation) >= MODE1D_ACCEPTABLE_TEMPLATE_CORRELATION
        and (score.xy_correlation is not None and float(score.xy_correlation) >= MODE1D_ACCEPTABLE_XY_CORRELATION)
    )
    mid = plane.shape[0] // 2
    return Mode1DSourceSweepCase(
        ring_count_target=float(ring_count),
        ring_count_actual=effective_ring_count_for_plane(radius_1e_field_m=cfg.beam_radius_m, k_r_m_inv=cfg.k_r_m_inv),
        k_r_m_inv=float(cfg.k_r_m_inv),
        axicon_base_angle_deg=float(np.rad2deg(cfg.axicon_base_angle_rad)),
        axicon_apex_angle_deg=float(cfg.axicon_apex_angle_deg),
        z_reference_m=float(z_values[ref]),
        grid_n=int(grid_n),
        classification=cls,
        accepted_hexagon=accepted,
        dark_core_ratio=dark,
        template_score=score,
        symmetry={str(k): v for k, v in sym.items()},
        intensity_xy=plane.astype(np.float32),
        xz_map=np.asarray(stack[:, mid, :], dtype=np.float32),
        grid=grid,
        z_values_m=z_values,
    )


def mode1d_source_sweep_row(case: Mode1DSourceSweepCase) -> dict[str, Any]:
    sym = case.symmetry
    score = case.template_score
    return {
        "ring_count_target": float(case.ring_count_target),
        "ring_count_actual": float(case.ring_count_actual),
        "k_r_m_inv": float(case.k_r_m_inv),
        "axicon_base_angle_deg": float(case.axicon_base_angle_deg),
        "axicon_apex_angle_deg": float(case.axicon_apex_angle_deg),
        "z_reference_m": float(case.z_reference_m),
        "z_reference_mm": float(case.z_reference_m / 1e-3),
        "grid_n": int(case.grid_n),
        "classification": str(case.classification),
        "accepted_hexagon": bool(case.accepted_hexagon),
        "dark_core_ratio": float(case.dark_core_ratio),
        "template_angular_correlation": float(score.angular_profile_correlation),
        "template_xy_correlation": float(score.xy_correlation if score.xy_correlation is not None else np.nan),
        "template_x_profile_correlation": float(score.x_profile_correlation),
        "template_y_profile_correlation": float(score.y_profile_correlation),
        "best_rotation_deg": float(score.best_rotation_deg),
        "scale_factor_to_v0_ring": float(score.scale_factor),
        "c60": float(sym.get("rot_corr_60", np.nan)),
        "c120": float(sym.get("rot_corr_120", np.nan)),
        "c120_minus_c60": float(sym.get("c120_minus_c60", np.nan)),
        "order3_over_order6": float(sym.get("order3_over_order6", np.nan)),
        "sector_balance_max_over_min": float(sym.get("six_sector_max_over_min", np.nan)),
        "three_pair_imbalance": float(sym.get("three_pair_imbalance", np.nan)),
        "ring_island_count": int(sym.get("ring_island_count", -1)),
    }


def run_mode1d_source_ring_count_sweep(
    ring_counts: Sequence[float] = MODE1D_DEFAULT_RING_SWEEP,
    *,
    grid_n: int = 384,
    z_planes: int = 21,
    template: Mode1BTargetTemplate | None = None,
) -> tuple[Mode1DSourceSweepCase, ...]:
    tmpl = template or build_mode1b_target_template(grid_n=max(384, int(grid_n)), z_planes=max(21, int(z_planes)))
    return tuple(
        run_mode1d_source_ring_count_case(
            float(ring),
            tmpl,
            grid_n=int(grid_n),
            z_planes=int(z_planes),
        )
        for ring in ring_counts
    )


def mode1d_minimum_accepted_ring_count(cases: Sequence[Mode1DSourceSweepCase]) -> float | None:
    accepted = [float(case.ring_count_target) for case in cases if case.accepted_hexagon]
    return None if not accepted else float(min(accepted))


def mode1d_outcome_report(
    context: Mode1DDesignContext,
    source_cases: Sequence[Mode1DSourceSweepCase],
    achievable_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    min_required = mode1d_minimum_accepted_ring_count(source_cases)
    max_plausible = max(
        (
            float(row["ring_count"])
            for row in achievable_rows
            if float(row["NA"]) <= 0.9 and str(row["case"]).endswith("slm_safe_radius")
        ),
        default=0.0,
    )
    max_exploratory = max(float(row["ring_count"]) for row in achievable_rows) if achievable_rows else 0.0
    if min_required is None:
        outcome = "M1D-C"
        statement = "No lower-ring source candidate in the sweep produced an acceptable Nathan-style hexagon."
    elif min_required <= max_plausible:
        outcome = "M1D-A"
        statement = (
            "A redesigned downstream system with available SLM aperture and plausible high NA can reach "
            "the minimum lower-ring Nathan-style regime. A redesigned MODE 1 downstream simulation is required next."
        )
    elif min_required <= max_exploratory or context.v0_ring_count <= max_exploratory:
        outcome = "M1D-B"
        statement = (
            "The mechanism appears reachable only with major optical changes such as immersion-class NA, "
            "mapping reduction, larger P2 radius, or changed axicon/focus placement."
        )
    else:
        outcome = "M1D-B"
        statement = (
            "A convincing source target exists, but it lies beyond the current and plausible redesigned "
            "ring-count budget; major optical redesign is required."
        )
    return {
        "stage": MODE1D_STAGE,
        "suggested_outcome": outcome,
        "allowed_outcomes": MODE1D_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "minimum_accepted_source_ring_count": min_required,
        "current_actual_ring_count": float(context.mode1c_current_ring_count),
        "current_na_current_radius_max_ring_count": float(context.mode1c_max_current_radius_na_limited),
        "current_na_slm_safe_max_ring_count": float(context.mode1c_max_slm_safe_na_limited),
        "max_plausible_na0p9_slm_safe_ring_count": float(max_plausible),
        "max_exploratory_ring_count_in_table": float(max_exploratory),
        "current_inherited_actual_reaches_minimum": bool(min_required is not None and context.mode1c_current_ring_count >= min_required),
        "current_na_current_radius_reaches_minimum": bool(min_required is not None and context.mode1c_max_current_radius_na_limited >= min_required),
        "current_na_slm_safe_radius_reaches_minimum": bool(min_required is not None and context.mode1c_max_slm_safe_na_limited >= min_required),
        "mode2a_2b_realisation_allowed": False,
        "mode2a_2b_gate": "blocked_pending_redesigned_mode1_confirmation",
        "scope": MODE1D_SCOPE_STATEMENT,
    }


def mode1d_scope_manifest(outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage": MODE1D_STAGE,
        "scope": MODE1D_SCOPE_STATEMENT,
        "inverse_redesign_requirements_only": True,
        "hwp_qwp_slm_panel_realism": False,
        "downstream_redesigned_simulation_confirmation": False,
        "physical_route_approval": False,
        "mode2a_2b_realisation_allowed": bool(outcome.get("mode2a_2b_realisation_allowed", False)) if outcome else False,
        "blocking_dependency": "MODE 2A/2B remains blocked unless M1D-A is followed by a confirming redesigned MODE 1 downstream simulation.",
    }


def run_mode1d_inverse_redesign(
    config: NathanHexagonConfig | None = None,
    *,
    ring_counts: Sequence[float] = MODE1D_DEFAULT_RING_SWEEP,
    source_grid_n: int = 384,
    source_z_planes: int = 21,
) -> dict[str, Any]:
    ctx = mode1d_design_context(config)
    required_na_rows = mode1d_required_na_table(ctx)
    required_radius_rows = mode1d_required_radius_table(ctx)
    required_mapping_rows = mode1d_required_mapping_table(ctx)
    achievable_rows = mode1d_achievable_ring_count_table(ctx)
    source_cases = run_mode1d_source_ring_count_sweep(
        ring_counts,
        grid_n=int(source_grid_n),
        z_planes=int(source_z_planes),
    )
    source_rows = tuple(mode1d_source_sweep_row(case) for case in source_cases)
    outcome = mode1d_outcome_report(ctx, source_cases, achievable_rows)
    return {
        "context": ctx,
        "required_na_rows": required_na_rows,
        "required_radius_rows": required_radius_rows,
        "required_mapping_rows": required_mapping_rows,
        "achievable_ring_count_rows": achievable_rows,
        "source_cases": source_cases,
        "source_ring_count_rows": source_rows,
        "outcome": outcome,
        "manifest": mode1d_scope_manifest(outcome),
    }


def plot_mode1d_source_ring_count_sweep(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    x = np.asarray([float(row["ring_count_target"]) for row in rows], dtype=float)
    dark = np.asarray([float(row["dark_core_ratio"]) for row in rows], dtype=float)
    ang = np.asarray([float(row["template_angular_correlation"]) for row in rows], dtype=float)
    cdelta = np.asarray([float(row["c120_minus_c60"]) for row in rows], dtype=float)
    accepted = np.asarray([bool(row["accepted_hexagon"]) for row in rows], dtype=bool)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    axes[0, 0].plot(x, dark, marker="o", color="tab:blue")
    axes[0, 0].axhline(MODE1D_ACCEPTABLE_DARK_CORE_RATIO, color="0.35", lw=0.8, ls="--")
    axes[0, 0].set_title("dark-core ratio")
    axes[0, 0].set_xlabel("source ring count")
    axes[0, 0].set_ylabel("core / ring peak")
    axes[0, 1].plot(x, ang, marker="o", color="tab:green")
    axes[0, 1].axhline(MODE1D_ACCEPTABLE_TEMPLATE_CORRELATION, color="0.35", lw=0.8, ls="--")
    axes[0, 1].set_title("V0 angular-template similarity")
    axes[0, 1].set_xlabel("source ring count")
    axes[0, 1].set_ylabel("correlation")
    axes[1, 0].plot(x, cdelta, marker="o", color="tab:red")
    axes[1, 0].axhline(0.04, color="0.35", lw=0.8, ls="--")
    axes[1, 0].set_title("C120 - C60")
    axes[1, 0].set_xlabel("source ring count")
    axes[1, 0].set_ylabel("rotational-correlation delta")
    axes[1, 1].scatter(x[~accepted], np.zeros(np.count_nonzero(~accepted)), label="not accepted", color="0.55")
    axes[1, 1].scatter(x[accepted], np.ones(np.count_nonzero(accepted)), label="accepted", color="tab:green")
    axes[1, 1].set_ylim(-0.25, 1.25)
    axes[1, 1].set_yticks([0, 1], labels=["no", "yes"])
    axes[1, 1].set_title("accepted Nathan-style hexagon")
    axes[1, 1].set_xlabel("source ring count")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("MODE 1D lower-ring source sweep")
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1d_redesign_budget(
    context: Mode1DDesignContext,
    achievable_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_ring_count: float | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    labels = [str(row["case"]).replace("_", "\n") for row in achievable_rows]
    values = [float(row["ring_count"]) for row in achievable_rows]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.bar(labels, values, color=["tab:blue" if v < context.v0_ring_count else "tab:green" for v in values])
    ax.axhline(context.v0_ring_count, color="0.2", lw=1.0, ls="--", label=f"V0 {context.v0_ring_count:.1f}")
    if minimum_ring_count is not None:
        ax.axhline(float(minimum_ring_count), color="tab:red", lw=1.0, ls=":", label=f"min accepted {float(minimum_ring_count):.1f}")
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("achievable ring count")
    ax.set_title("MODE 1D redesign ring-count budget under current mapping")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    _save_fig(fig, output_path)
    return fig, ax


def plot_mode1d_source_case(
    case: Mode1DSourceSweepCase,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    x = np.asarray(case.grid["x"], dtype=float) / 1e-3
    z = np.asarray(case.z_values_m, dtype=float) / 1e-3
    ext_xy = [float(x[0]), float(x[-1]), float(x[0]), float(x[-1])]
    ext_xz = [float(x[0]), float(x[-1]), float(z[0]), float(z[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].imshow(_normalise_image(case.intensity_xy, local=True), origin="lower", extent=ext_xy, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"xy, N={case.ring_count_target:g}\n{case.classification}")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[1].imshow(_normalise_image(case.xz_map, local=True), origin="lower", aspect="auto", extent=ext_xz, cmap="inferno", vmin=0, vmax=1)
    axes[1].axhline(case.z_reference_m / 1e-3, color="white", lw=0.8, alpha=0.8)
    axes[1].set_title("x-z propagation")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")
    fig.suptitle(f"MODE 1D source-ring case: accepted={case.accepted_hexagon}")
    _save_fig(fig, output_path)
    return fig, axes


def write_mode1d_inverse_redesign_outputs(
    config: NathanHexagonConfig | None = None,
    *,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode1d_inverse_redesign",
    source_grid_n: int = 384,
    source_z_planes: int = 21,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = run_mode1d_inverse_redesign(config, source_grid_n=int(source_grid_n), source_z_planes=int(source_z_planes))
    paths = {
        "required_na_csv": out / "mode1d_required_na_table.csv",
        "required_na_json": out / "mode1d_required_na_table.json",
        "required_radius_csv": out / "mode1d_required_radius_table.csv",
        "required_radius_json": out / "mode1d_required_radius_table.json",
        "required_mapping_csv": out / "mode1d_required_mapping_table.csv",
        "required_mapping_json": out / "mode1d_required_mapping_table.json",
        "achievable_ring_count_csv": out / "mode1d_achievable_ring_count_table.csv",
        "achievable_ring_count_json": out / "mode1d_achievable_ring_count_table.json",
        "source_ring_count_sweep_csv": out / "mode1d_source_ring_count_sweep.csv",
        "source_ring_count_sweep_json": out / "mode1d_source_ring_count_sweep.json",
        "source_ring_count_sweep_summary": out / "mode1d_source_ring_count_sweep_summary.png",
        "redesign_budget_plot": out / "mode1d_redesign_budget_plot.png",
        "outcome_report": out / "mode1d_outcome_report.json",
        "scope_manifest": out / "simulation_scope_manifest.json",
    }
    table_pairs = (
        (paths["required_na_csv"], paths["required_na_json"], report["required_na_rows"]),
        (paths["required_radius_csv"], paths["required_radius_json"], report["required_radius_rows"]),
        (paths["required_mapping_csv"], paths["required_mapping_json"], report["required_mapping_rows"]),
        (paths["achievable_ring_count_csv"], paths["achievable_ring_count_json"], report["achievable_ring_count_rows"]),
        (paths["source_ring_count_sweep_csv"], paths["source_ring_count_sweep_json"], report["source_ring_count_rows"]),
    )
    for csv_path, json_path, rows in table_pairs:
        _write_rows(csv_path, rows)
        json_path.write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")
    fig, _ = plot_mode1d_source_ring_count_sweep(report["source_ring_count_rows"], output_path=paths["source_ring_count_sweep_summary"])
    plt.close(fig)
    fig, _ = plot_mode1d_redesign_budget(
        report["context"],
        report["achievable_ring_count_rows"],
        minimum_ring_count=report["outcome"]["minimum_accepted_source_ring_count"],
        output_path=paths["redesign_budget_plot"],
    )
    plt.close(fig)
    accepted = [case for case in report["source_cases"] if case.accepted_hexagon]
    shortlist = accepted[:2] if accepted else sorted(report["source_cases"], key=lambda case: case.template_score.angular_profile_correlation, reverse=True)[:2]
    for case in shortlist:
        key = f"source_case_N{float(case.ring_count_target):04.1f}".replace(".", "p")
        path = out / f"mode1d_{key}_xy_xz.png"
        fig, _ = plot_mode1d_source_case(case, output_path=path)
        plt.close(fig)
        paths[key] = path
    paths["outcome_report"].write_text(json.dumps(_json_ready(report["outcome"]), indent=2), encoding="utf-8")
    paths["scope_manifest"].write_text(json.dumps(_json_ready(report["manifest"]), indent=2), encoding="utf-8")
    return paths


# ---------------------------------------------------------------------------
# MODE 2-preflight: component-level Jones synthesis only
# ---------------------------------------------------------------------------

MODE2P_STAGE = "nathan_mode2_preflight_jones_synthesis"
MODE2P_ALLOWED_OUTCOMES = ("M2P-A", "M2P-B", "M2P-C", "M2P-D")
MODE2P_ACCEPTANCE_OVERLAP = 0.999
MODE2P_SCOPE_STATEMENT = (
    "MODE 2-preflight only: component-level Jones-chain equivalence at P2. "
    "No downstream propagation, no carrier/iris/panel realism, and no MODE 2A/2B physical-route approval."
)


@dataclass(frozen=True)
class JonesChainResult:
    route_id: str
    Ex: np.ndarray
    Ey: np.ndarray
    metadata: dict
    overlap_to_target: float
    rms_error: float
    global_phase_rad: float


def nathan_alpha_map(
    theta: np.ndarray,
    *,
    sector_num_pairs: int = 3,
    sector_theta: float = np.pi / 3.0,
    sector_rotation: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Nathan's six-sector alpha map and the radial-sector mask.

    The convention is the validated V0 one: each 120 degree cell starts with an
    azimuthal sector and ends with the radial sector of width ``sector_theta``.
    """

    theta_arr = np.asarray(theta, dtype=float)
    cell_angle = TWOPI / float(sector_num_pairs)
    phi = np.mod(theta_arr - float(sector_rotation), TWOPI)
    phi_cell = np.mod(phi, cell_angle)
    radial_mask = phi_cell >= (cell_angle - float(sector_theta))
    phi0 = np.where(radial_mask, 0.0, 0.5 * np.pi)
    alpha = theta_arr + phi0
    return alpha, radial_mask


def rot(theta: float) -> np.ndarray:
    """Return the passive 2D rotation matrix used by the Jones retarder helpers."""

    c = np.cos(float(theta))
    s = np.sin(float(theta))
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def linear_retarder(delta: float, beta: Any) -> np.ndarray:
    """Return ``R(-beta) diag(exp(-i delta/2), exp(+i delta/2)) R(beta)``."""

    beta_arr = np.asarray(beta, dtype=float)
    c = np.cos(beta_arr)
    s = np.sin(beta_arr)
    fast = np.exp(-0.5j * float(delta))
    slow = np.exp(0.5j * float(delta))
    jxx = fast * c * c + slow * s * s
    jxy = (slow - fast) * s * c
    jyx = jxy
    jyy = fast * s * s + slow * c * c
    return np.asarray([[jxx, jxy], [jyx, jyy]], dtype=np.complex128)


def hwp(beta: Any) -> np.ndarray:
    return linear_retarder(np.pi, beta)


def qwp(beta: Any) -> np.ndarray:
    return linear_retarder(0.5 * np.pi, beta)


def apply_jones_map(
    Jxx: Any,
    Jxy: Any,
    Jyx: Any,
    Jyy: Any,
    Ex: Any,
    Ey: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a possibly spatially varying 2x2 Jones map to a linear field."""

    ex = np.asarray(Ex, dtype=np.complex128)
    ey = np.asarray(Ey, dtype=np.complex128)
    return (
        np.asarray(Jxx, dtype=np.complex128) * ex + np.asarray(Jxy, dtype=np.complex128) * ey,
        np.asarray(Jyx, dtype=np.complex128) * ex + np.asarray(Jyy, dtype=np.complex128) * ey,
    )


def apply_uniform_jones(J: Any, Ex: Any, Ey: Any) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(J, dtype=np.complex128)
    if arr.shape[0:2] != (2, 2):
        raise ValueError("J must have leading shape (2, 2).")
    return apply_jones_map(arr[0, 0], arr[0, 1], arr[1, 0], arr[1, 1], Ex, Ey)


def apply_hwp_map_to_horizontal(A: Any, beta: Any) -> tuple[np.ndarray, np.ndarray]:
    """Ideal continuous HWP synthesis from horizontal input, up to a global phase."""

    amp = np.asarray(A, dtype=float)
    axis = np.asarray(beta, dtype=float)
    return (
        (amp * np.cos(2.0 * axis)).astype(np.complex128),
        (amp * np.sin(2.0 * axis)).astype(np.complex128),
    )


def wrap_2pi(phase_rad: Any) -> np.ndarray:
    return np.mod(np.asarray(phase_rad, dtype=float), TWOPI)


def mode2p_target_arrays(
    config: NathanHexagonConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
    amplitude: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the MODE 2P P2 target arrays without invoking downstream optics."""

    cfg = config or NathanHexagonConfig.fast()
    grid_dict = dict(default_nathan_grid(cfg) if grid is None else grid)
    theta = np.asarray(grid_dict["PHI"], dtype=float)
    sector_theta = float(cfg.vector.sector_duty) * TWOPI / float(cfg.vector.n_pairs)
    alpha, radial_mask = nathan_alpha_map(
        theta,
        sector_num_pairs=int(cfg.vector.n_pairs),
        sector_theta=sector_theta,
        sector_rotation=float(cfg.vector.sector_rotation_rad),
    )
    A = gaussian_envelope(grid_dict, cfg.vector) if amplitude is None else np.asarray(amplitude, dtype=float)
    Ex_t = (A * np.cos(alpha)).astype(np.complex128)
    Ey_t = (A * np.sin(alpha)).astype(np.complex128)
    metric_mask = A > max(float(np.max(np.abs(A))), EPS) * 1.0e-6
    return {
        "config": cfg,
        "grid": grid_dict,
        "A": A,
        "alpha": alpha,
        "radial_mask": radial_mask,
        "target": (Ex_t, Ey_t),
        "metric_mask": metric_mask,
        "centre_policy": {
            "policy": "shared_target_and_route_alpha_map; no centre zeroing in component-level proof",
            "axis_sampled": bool(np.any(np.hypot(np.asarray(grid_dict["X"], dtype=float), np.asarray(grid_dict["Y"], dtype=float)) <= EPS)),
            "metrics_report_centre_sensitivity": True,
        },
    }


def complex_vector_inner(
    Ex1: Any,
    Ey1: Any,
    Ex2: Any,
    Ey2: Any,
    mask: np.ndarray | None = None,
) -> complex:
    ex1 = np.asarray(Ex1, dtype=np.complex128)
    ey1 = np.asarray(Ey1, dtype=np.complex128)
    ex2 = np.asarray(Ex2, dtype=np.complex128)
    ey2 = np.asarray(Ey2, dtype=np.complex128)
    if mask is None:
        use = np.ones_like(ex1, dtype=bool)
    else:
        use = np.asarray(mask, dtype=bool)
    return complex(np.vdot(ex1[use], ex2[use]) + np.vdot(ey1[use], ey2[use]))


def complex_vector_overlap(
    field: tuple[Any, Any],
    target: tuple[Any, Any],
    mask: np.ndarray | None = None,
) -> float:
    Ex, Ey = field
    Tx, Ty = target
    num = abs(complex_vector_inner(Ex, Ey, Tx, Ty, mask)) ** 2
    den = (
        complex_vector_inner(Ex, Ey, Ex, Ey, mask).real
        * complex_vector_inner(Tx, Ty, Tx, Ty, mask).real
    )
    value = float(num / max(float(den), EPS))
    return float(min(1.0, max(0.0, value)))


def best_global_phase(
    field: tuple[Any, Any],
    target: tuple[Any, Any],
    mask: np.ndarray | None = None,
) -> float:
    """Return the constant piston on ``field`` relative to ``target``."""

    ip = complex_vector_inner(target[0], target[1], field[0], field[1], mask)
    return float(np.angle(ip))


def phase_aligned_rms(
    field: tuple[Any, Any],
    target: tuple[Any, Any],
    mask: np.ndarray | None = None,
) -> float:
    gamma = best_global_phase(field, target, mask)
    use = np.ones_like(np.asarray(target[0]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    Ex = np.asarray(field[0], dtype=np.complex128) * np.exp(-1j * gamma)
    Ey = np.asarray(field[1], dtype=np.complex128) * np.exp(-1j * gamma)
    Tx = np.asarray(target[0], dtype=np.complex128)
    Ty = np.asarray(target[1], dtype=np.complex128)
    diff = np.abs(Ex[use] - Tx[use]) ** 2 + np.abs(Ey[use] - Ty[use]) ** 2
    ref = np.abs(Tx[use]) ** 2 + np.abs(Ty[use]) ** 2
    return float(np.sqrt(np.sum(diff) / max(float(np.sum(ref)), EPS)))


def jones_stokes_rms(
    field: tuple[Any, Any],
    target: tuple[Any, Any],
    mask: np.ndarray | None = None,
) -> float:
    use = np.ones_like(np.asarray(target[0]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    st_f = stokes_from_linear_components(field[0], field[1])
    st_t = stokes_from_linear_components(target[0], target[1])
    diff = np.zeros(int(np.count_nonzero(use)), dtype=float)
    for key in ("S0", "S1", "S2", "S3"):
        diff += (np.asarray(st_f[key], dtype=float)[use] - np.asarray(st_t[key], dtype=float)[use]) ** 2
    scale = max(float(np.sqrt(np.mean(np.asarray(st_t["S0"], dtype=float)[use] ** 2))), EPS)
    return float(np.sqrt(np.mean(diff)) / scale)


def alpha_angle_rms_mod_pi(
    field: tuple[Any, Any],
    target: tuple[Any, Any],
    mask: np.ndarray | None = None,
) -> float:
    use = np.ones_like(np.asarray(target[0]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    st_f = stokes_from_linear_components(field[0], field[1])
    st_t = stokes_from_linear_components(target[0], target[1])
    ang_f = 0.5 * np.arctan2(st_f["S2"], st_f["S1"])
    ang_t = 0.5 * np.arctan2(st_t["S2"], st_t["S1"])
    err = 0.5 * np.angle(np.exp(2j * (ang_f[use] - ang_t[use])))
    return float(np.sqrt(np.mean(err**2))) if err.size else 0.0


def jones_power_ratio(
    field: tuple[Any, Any],
    target: tuple[Any, Any],
    mask: np.ndarray | None = None,
) -> float:
    use = np.ones_like(np.asarray(target[0]), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    f = np.abs(np.asarray(field[0], dtype=np.complex128)[use]) ** 2 + np.abs(np.asarray(field[1], dtype=np.complex128)[use]) ** 2
    t = np.abs(np.asarray(target[0], dtype=np.complex128)[use]) ** 2 + np.abs(np.asarray(target[1], dtype=np.complex128)[use]) ** 2
    ratio = float(np.sum(f) / max(float(np.sum(t)), EPS))
    return 1.0 if abs(ratio - 1.0) < 1.0e-12 else ratio


def jones_metric_row(
    route_id: str,
    field: tuple[Any, Any],
    target: tuple[Any, Any],
    *,
    mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "complex_vector_overlap": complex_vector_overlap(field, target, mask),
        "phase_aligned_rms": phase_aligned_rms(field, target, mask),
        "global_phase_rad": best_global_phase(field, target, mask),
        "stokes_rms": jones_stokes_rms(field, target, mask),
        "alpha_angle_rms_mod_pi": alpha_angle_rms_mod_pi(field, target, mask),
        "power_ratio": jones_power_ratio(field, target, mask),
        **dict(metadata or {}),
    }


def _make_jones_result(
    route_id: str,
    Ex: np.ndarray,
    Ey: np.ndarray,
    target: tuple[Any, Any],
    *,
    mask: np.ndarray | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> JonesChainResult:
    field = (np.asarray(Ex, dtype=np.complex128), np.asarray(Ey, dtype=np.complex128))
    return JonesChainResult(
        route_id=route_id,
        Ex=field[0],
        Ey=field[1],
        metadata=dict(metadata or {}),
        overlap_to_target=complex_vector_overlap(field, target, mask),
        rms_error=phase_aligned_rms(field, target, mask),
        global_phase_rad=best_global_phase(field, target, mask),
    )


def synthesize_with_patterned_hwp(A: Any, alpha: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    beta = 0.5 * np.asarray(alpha, dtype=float)
    Ex, Ey = apply_hwp_map_to_horizontal(A, beta)
    return Ex, Ey, {"beta_rad": beta, "input_jones": "horizontal", "hwp_axis_rule": "beta = alpha/2"}


def route_patterned_hwp_ideal(
    A: Any,
    alpha: Any,
    target: tuple[Any, Any],
    *,
    mask: np.ndarray | None = None,
) -> JonesChainResult:
    Ex, Ey, meta = synthesize_with_patterned_hwp(A, alpha)
    return _make_jones_result(
        "route_patterned_hwp_ideal",
        Ex,
        Ey,
        target,
        mask=mask,
        metadata={
            "route_family": "patterned_hwp",
            "component_level_only": True,
            **meta,
        },
    )


def synthesize_from_circular_components(
    A: Any,
    alpha: Any,
    *,
    sign: int = +1,
    piston: float = 0.0,
    Phi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    amp = np.asarray(A, dtype=float)
    phase = np.zeros_like(amp, dtype=float) if Phi is None else np.asarray(Phi, dtype=float)
    a = np.asarray(alpha, dtype=float)
    s = 1 if int(sign) >= 0 else -1
    ER = amp / np.sqrt(2.0) * np.exp(1j * (phase + s * a))
    EL = amp / np.sqrt(2.0) * np.exp(1j * (phase - s * a + float(piston)))
    Ex = (ER + EL) / np.sqrt(2.0)
    Ey = (-1j * ER + 1j * EL) / np.sqrt(2.0)
    return Ex, Ey, {
        "circular_amplitude_normalisation": "A/sqrt(2) per circular channel",
        "sign": int(s),
        "piston_rad": float(piston),
        "slm1_phase_rad": wrap_2pi(phase + s * a),
        "slm2_phase_rad": wrap_2pi(phase - s * a + float(piston)),
    }


def route_dual_slm_circular_ideal(
    A: Any,
    alpha: Any,
    target: tuple[Any, Any],
    *,
    Phi: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> JonesChainResult:
    candidates: list[tuple[float, float, int, float, np.ndarray, np.ndarray, dict[str, Any]]] = []
    for sign in (+1, -1):
        for piston in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
            Ex, Ey, meta = synthesize_from_circular_components(A, alpha, sign=sign, piston=piston, Phi=Phi)
            score = complex_vector_overlap((Ex, Ey), target, mask)
            rms = phase_aligned_rms((Ex, Ey), target, mask)
            candidates.append((score, -rms, sign, piston, Ex, Ey, meta))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    score, neg_rms, sign, piston, Ex, Ey, meta = candidates[0]
    sweep = [
        {
            "sign": int(row[2]),
            "piston_rad": float(row[3]),
            "complex_vector_overlap": float(row[0]),
            "phase_aligned_rms": float(-row[1]),
        }
        for row in candidates
    ]
    return _make_jones_result(
        "route_dual_slm_circular_ideal",
        Ex,
        Ey,
        target,
        mask=mask,
        metadata={
            "route_family": "dual_slm_circular_identity",
            "component_level_only": True,
            "selected_sign": int(sign),
            "selected_piston_rad": float(piston),
            "convention_sweep": sweep,
            **meta,
        },
    )


def route_dual_slm_linear_then_qwp_ideal(
    A: Any,
    alpha: Any,
    target: tuple[Any, Any],
    *,
    Phi: np.ndarray | None = None,
    qwp_angle: float = np.pi / 4.0,
    mask: np.ndarray | None = None,
) -> JonesChainResult:
    amp = np.asarray(A, dtype=float)
    a = np.asarray(alpha, dtype=float)
    phase = np.zeros_like(amp, dtype=float) if Phi is None else np.asarray(Phi, dtype=float)
    qwp_angles = tuple(dict.fromkeys((float(qwp_angle), -float(qwp_angle))))
    candidates: list[dict[str, Any]] = []
    for h_sign, v_sign in ((+1, -1), (-1, +1)):
        for v_piston in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
            EH = amp / np.sqrt(2.0) * np.exp(1j * (phase + h_sign * a))
            EV = amp / np.sqrt(2.0) * np.exp(1j * (phase + v_sign * a + float(v_piston)))
            for angle in qwp_angles:
                Ex, Ey = apply_uniform_jones(qwp(angle), EH, EV)
                field = (Ex, Ey)
                candidates.append(
                    {
                        "complex_vector_overlap": complex_vector_overlap(field, target, mask),
                        "phase_aligned_rms": phase_aligned_rms(field, target, mask),
                        "h_phase_sign": int(h_sign),
                        "v_phase_sign": int(v_sign),
                        "h_piston_rad": 0.0,
                        "v_piston_rad": float(v_piston),
                        "qwp_angle_rad": float(angle),
                        "Ex": Ex,
                        "Ey": Ey,
                        "h_phase_mask_rad": wrap_2pi(phase + h_sign * a),
                        "v_phase_mask_rad": wrap_2pi(phase + v_sign * a + float(v_piston)),
                    }
                )
    candidates.sort(key=lambda row: (float(row["complex_vector_overlap"]), -float(row["phase_aligned_rms"])), reverse=True)
    best = candidates[0]
    sweep = [
        {
            "h_phase_sign": int(row["h_phase_sign"]),
            "v_phase_sign": int(row["v_phase_sign"]),
            "v_piston_rad": float(row["v_piston_rad"]),
            "qwp_angle_rad": float(row["qwp_angle_rad"]),
            "complex_vector_overlap": float(row["complex_vector_overlap"]),
            "phase_aligned_rms": float(row["phase_aligned_rms"]),
        }
        for row in candidates
    ]
    return _make_jones_result(
        "route_dual_slm_linear_then_qwp_ideal",
        np.asarray(best["Ex"], dtype=np.complex128),
        np.asarray(best["Ey"], dtype=np.complex128),
        target,
        mask=mask,
        metadata={
            "route_family": "dual_slm_linear_channels_then_uniform_qwp",
            "component_level_only": True,
            "linear_channel_amplitude_normalisation": "A/sqrt(2) per H/V channel",
            "selected_h_phase_sign": int(best["h_phase_sign"]),
            "selected_v_phase_sign": int(best["v_phase_sign"]),
            "selected_h_piston_rad": float(best["h_piston_rad"]),
            "selected_v_piston_rad": float(best["v_piston_rad"]),
            "selected_qwp_angle_rad": float(best["qwp_angle_rad"]),
            "slm1_phase_rad": best["h_phase_mask_rad"],
            "slm2_phase_rad": best["v_phase_mask_rad"],
            "convention_sweep": sweep,
        },
    )


def mode2p_centre_treatment_report(
    result: JonesChainResult,
    target: tuple[Any, Any],
    grid: Mapping[str, Any],
    *,
    metric_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    R = np.hypot(np.asarray(grid["X"], dtype=float), np.asarray(grid["Y"], dtype=float))
    centre_mask = R <= max(float(np.min(R)), EPS)
    include_mask = np.ones_like(R, dtype=bool) if metric_mask is None else np.asarray(metric_mask, dtype=bool)
    exclude_mask = include_mask & ~centre_mask
    field = (result.Ex, result.Ey)
    centre_power = np.abs(result.Ex[centre_mask]) ** 2 + np.abs(result.Ey[centre_mask]) ** 2
    target_power = np.abs(np.asarray(target[0])[centre_mask]) ** 2 + np.abs(np.asarray(target[1])[centre_mask]) ** 2
    return {
        "route_id": result.route_id,
        "axis_sampled": bool(np.any(R <= EPS)),
        "centre_pixel_count": int(np.count_nonzero(centre_mask)),
        "centre_power_sum": float(np.sum(centre_power)),
        "target_centre_power_sum": float(np.sum(target_power)),
        "overlap_with_centre_policy": complex_vector_overlap(field, target, include_mask),
        "overlap_excluding_centre_pixels": complex_vector_overlap(field, target, exclude_mask),
        "rms_with_centre_policy": phase_aligned_rms(field, target, include_mask),
        "rms_excluding_centre_pixels": phase_aligned_rms(field, target, exclude_mask),
        "centre_policy": "reported sensitivity; no route-specific centre editing",
    }


def mode2p_outcome_report(
    patterned_hwp: JonesChainResult,
    dual_slm_qwp: JonesChainResult,
    *,
    circular_identity: JonesChainResult | None = None,
    acceptance_overlap: float = MODE2P_ACCEPTANCE_OVERLAP,
    mode1c_outcome: str = "M1C-C",
) -> dict[str, Any]:
    hwp_pass = patterned_hwp.overlap_to_target >= float(acceptance_overlap)
    dual_pass = dual_slm_qwp.overlap_to_target >= float(acceptance_overlap)
    if hwp_pass and dual_pass:
        outcome = "M2P-A"
        statement = (
            "Both ideal patterned-HWP and ideal dual-SLM/QWP routes reproduce the Nathan P2 target. "
            "Upstream synthesis mathematics is validated only."
        )
    elif hwp_pass:
        outcome = "M2P-B"
        statement = "Patterned-HWP synthesis passes, but the dual-SLM/QWP convention does not."
    elif dual_pass:
        outcome = "M2P-C"
        statement = "Dual-SLM/QWP synthesis passes, but the patterned-HWP convention does not."
    else:
        outcome = "M2P-D"
        statement = "Neither ideal component route reproduces the Nathan P2 target."
    downstream_allowed = bool(outcome == "M2P-A" and str(mode1c_outcome) == "M1C-A")
    return {
        "stage": MODE2P_STAGE,
        "suggested_outcome": outcome,
        "allowed_outcomes": MODE2P_ALLOWED_OUTCOMES,
        "acceptance_overlap": float(acceptance_overlap),
        "outcome_statement": statement,
        "patterned_hwp_pass": bool(hwp_pass),
        "dual_slm_qwp_pass": bool(dual_pass),
        "circular_identity_pass": None if circular_identity is None else bool(circular_identity.overlap_to_target >= float(acceptance_overlap)),
        "patterned_hwp_overlap": float(patterned_hwp.overlap_to_target),
        "dual_slm_qwp_overlap": float(dual_slm_qwp.overlap_to_target),
        "circular_identity_overlap": None if circular_identity is None else float(circular_identity.overlap_to_target),
        "mode1c_active_outcome": str(mode1c_outcome),
        "mode2a_2b_realisation_allowed": bool(downstream_allowed),
        "mode2a_2b_gate": "open" if downstream_allowed else f"blocked_by_{mode1c_outcome}",
        "downstream_statement": (
            "Upstream target synthesis is mathematically valid; current downstream ring-count/NA/aperture "
            "constraints still block full physical realisation."
            if not downstream_allowed
            else "Downstream gate is open only because MODE 1C is M1C-A."
        ),
        "scope": MODE2P_SCOPE_STATEMENT,
    }


def mode2p_scope_manifest(outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage": MODE2P_STAGE,
        "scope": MODE2P_SCOPE_STATEMENT,
        "mode2_preflight_only": True,
        "downstream_propagation": False,
        "physical_route_approval": False,
        "carrier_iris_panel_realism": False,
        "tests_component_level_mathematical_equivalence_only": True,
        "mode2a_2b_realisation_allowed": bool(outcome.get("mode2a_2b_realisation_allowed", False)) if outcome else False,
        "blocking_dependency": "MODE 1C remains blocked unless a later M1C-A outcome is produced.",
    }


def run_mode2p_jones_synthesis(
    config: NathanHexagonConfig | None = None,
    *,
    grid: Mapping[str, Any] | None = None,
    mode1c_outcome: str = "M1C-C",
) -> dict[str, Any]:
    target_data = mode2p_target_arrays(config, grid=grid)
    A = target_data["A"]
    alpha = target_data["alpha"]
    target = target_data["target"]
    mask = target_data["metric_mask"]
    hwp_result = route_patterned_hwp_ideal(A, alpha, target, mask=mask)
    circular_result = route_dual_slm_circular_ideal(A, alpha, target, mask=mask)
    qwp_result = route_dual_slm_linear_then_qwp_ideal(A, alpha, target, mask=mask)
    outcome = mode2p_outcome_report(
        hwp_result,
        qwp_result,
        circular_identity=circular_result,
        mode1c_outcome=mode1c_outcome,
    )
    route_rows = [
        jones_metric_row(
            hwp_result.route_id,
            (hwp_result.Ex, hwp_result.Ey),
            target,
            mask=mask,
            metadata={"route_family": hwp_result.metadata["route_family"]},
        ),
        jones_metric_row(
            circular_result.route_id,
            (circular_result.Ex, circular_result.Ey),
            target,
            mask=mask,
            metadata={
                "route_family": circular_result.metadata["route_family"],
                "selected_sign": circular_result.metadata["selected_sign"],
                "selected_piston_rad": circular_result.metadata["selected_piston_rad"],
            },
        ),
        jones_metric_row(
            qwp_result.route_id,
            (qwp_result.Ex, qwp_result.Ey),
            target,
            mask=mask,
            metadata={
                "route_family": qwp_result.metadata["route_family"],
                "selected_h_phase_sign": qwp_result.metadata["selected_h_phase_sign"],
                "selected_v_phase_sign": qwp_result.metadata["selected_v_phase_sign"],
                "selected_v_piston_rad": qwp_result.metadata["selected_v_piston_rad"],
                "selected_qwp_angle_rad": qwp_result.metadata["selected_qwp_angle_rad"],
            },
        ),
    ]
    centre_reports = {
        "patterned_hwp": mode2p_centre_treatment_report(hwp_result, target, target_data["grid"], metric_mask=mask),
        "dual_slm_qwp": mode2p_centre_treatment_report(qwp_result, target, target_data["grid"], metric_mask=mask),
    }
    return {
        "target_data": target_data,
        "patterned_hwp": hwp_result,
        "circular_identity": circular_result,
        "dual_slm_qwp": qwp_result,
        "route_rows": route_rows,
        "centre_reports": centre_reports,
        "outcome": outcome,
        "manifest": mode2p_scope_manifest(outcome),
    }


def _mode2p_axis_um(grid: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(grid.get("x", make_xy_grid(int(grid["N"]), float(grid["dx"]))["x"]), dtype=float) / 1e-6


def plot_mode2p_target_alpha_and_sector_map(
    target_data: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = target_data["grid"]
    x_um = _mode2p_axis_um(grid)
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    panels = [
        (target_data["A"], "A(r)", "inferno"),
        (np.mod(target_data["alpha"], np.pi), "alpha mod pi", "twilight"),
        (np.asarray(target_data["radial_mask"], dtype=float), "radial sector mask", "viridis"),
    ]
    for ax, (image, title, cmap) in zip(axes, panels, strict=True):
        im = ax.imshow(image, origin="lower", extent=ext, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, shrink=0.82)
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2p_route_vs_target(
    result: JonesChainResult,
    target_data: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = target_data["grid"]
    target = target_data["target"]
    mask = target_data["metric_mask"]
    gamma = best_global_phase((result.Ex, result.Ey), target, mask)
    Ex = result.Ex * np.exp(-1j * gamma)
    Ey = result.Ey * np.exp(-1j * gamma)
    Tx, Ty = target
    x_um = _mode2p_axis_um(grid)
    ext = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    route_s0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    target_s0 = np.abs(Tx) ** 2 + np.abs(Ty) ** 2
    diff = np.sqrt(np.abs(Ex - Tx) ** 2 + np.abs(Ey - Ty) ** 2)
    route_st = stokes_from_linear_components(Ex, Ey)
    target_st = stokes_from_linear_components(Tx, Ty)
    route_angle = 0.5 * np.arctan2(route_st["S2"], route_st["S1"])
    target_angle = 0.5 * np.arctan2(target_st["S2"], target_st["S1"])
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), constrained_layout=True)
    panels = [
        (target_s0, "target S0", "inferno", None),
        (route_s0, "route S0", "inferno", None),
        (diff, "aligned vector error", "magma", None),
        (target_angle, "target alpha mod pi", "twilight", None),
        (route_angle, "route alpha mod pi", "twilight", None),
        (route_st["S3"] - target_st["S3"], "S3 difference", "coolwarm", "sym"),
    ]
    for ax, (image, title, cmap, mode) in zip(axes.ravel(), panels, strict=True):
        arr = np.asarray(image, dtype=float)
        if mode == "sym":
            lim = max(float(np.max(np.abs(arr))), EPS)
            im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(im, ax=ax, shrink=0.78)
    fig.suptitle(
        f"{result.route_id}: overlap={result.overlap_to_target:.6f}, rms={result.rms_error:.3e}",
        fontsize=11,
    )
    _save_fig(fig, output_path)
    return fig, axes


def write_mode2p_jones_synthesis_outputs(
    config: NathanHexagonConfig | None = None,
    *,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode2_preflight_jones",
    mode1c_outcome: str = "M1C-C",
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = run_mode2p_jones_synthesis(config, mode1c_outcome=mode1c_outcome)
    paths = {
        "target_alpha_and_sector_map": out / "target_alpha_and_sector_map.png",
        "route_patterned_hwp_ideal_vs_target": out / "route_patterned_hwp_ideal_vs_target.png",
        "route_dual_slm_qwp_ideal_vs_target": out / "route_dual_slm_qwp_ideal_vs_target.png",
        "summary_json": out / "jones_synthesis_summary.json",
        "summary_csv": out / "jones_synthesis_summary.csv",
        "scope_manifest": out / "simulation_scope_manifest.json",
    }
    fig, _ = plot_mode2p_target_alpha_and_sector_map(report["target_data"], output_path=paths["target_alpha_and_sector_map"])
    plt.close(fig)
    fig, _ = plot_mode2p_route_vs_target(report["patterned_hwp"], report["target_data"], output_path=paths["route_patterned_hwp_ideal_vs_target"])
    plt.close(fig)
    fig, _ = plot_mode2p_route_vs_target(report["dual_slm_qwp"], report["target_data"], output_path=paths["route_dual_slm_qwp_ideal_vs_target"])
    plt.close(fig)
    summary = {
        "stage": MODE2P_STAGE,
        "scope": MODE2P_SCOPE_STATEMENT,
        "target": {
            "grid_n": int(report["target_data"]["grid"]["N"]),
            "dx_m": float(report["target_data"]["grid"]["dx"]),
            "sector_num_pairs": int(report["target_data"]["config"].vector.n_pairs),
            "sector_theta_rad": float(report["target_data"]["config"].vector.sector_duty) * TWOPI / float(report["target_data"]["config"].vector.n_pairs),
            "sector_rotation_rad": float(report["target_data"]["config"].vector.sector_rotation_rad),
            "centre_policy": report["target_data"]["centre_policy"],
        },
        "routes": report["route_rows"],
        "selected_dual_slm_qwp_convention": {
            "h_phase_sign": report["dual_slm_qwp"].metadata["selected_h_phase_sign"],
            "v_phase_sign": report["dual_slm_qwp"].metadata["selected_v_phase_sign"],
            "v_piston_rad": report["dual_slm_qwp"].metadata["selected_v_piston_rad"],
            "qwp_angle_rad": report["dual_slm_qwp"].metadata["selected_qwp_angle_rad"],
        },
        "selected_circular_convention": {
            "sign": report["circular_identity"].metadata["selected_sign"],
            "piston_rad": report["circular_identity"].metadata["selected_piston_rad"],
        },
        "centre_reports": report["centre_reports"],
        "outcome": report["outcome"],
    }
    paths["summary_json"].write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    _write_rows(paths["summary_csv"], report["route_rows"])
    paths["scope_manifest"].write_text(json.dumps(_json_ready(report["manifest"]), indent=2), encoding="utf-8")
    return paths


# ===========================================================================
# MODE 1E - actual redesigned downstream confirmation
# ===========================================================================
#
# MODE 1D established a plausible redesign budget (minimum accepted source ring
# count ~12 within the SLM-safe P2 radius and objective NA).  MODE 1E asks the
# confirming question: does a legitimate, fully-resolved redesigned downstream
# configuration - redesigned pre-axicon k_r AND consistent redesigned target /
# surface k_r, run through the ACTUAL inherited MODE 1 downstream machinery -
# produce a sample-plane micro-scale version of the minimum accepted N=12
# Nathan-style hexagonal Bessel target?  Ideal P2 input only; a free-space
# proxy run, a source-template pass, or a P2 Jones pass can NEVER yield M1E-A.

MODE1E_STAGE = "nathan_mode1e_redesigned_downstream_confirmation"
MODE1E_ALLOWED_OUTCOMES = ("M1E-A", "M1E-B", "M1E-C", "M1E-D")
MODE1E_PRIMARY_TEMPLATE_RING_COUNT = 12.0
MODE1E_REFERENCE_TEMPLATE_RING_COUNT = 31.0
MODE1E_DEFAULT_RING_COUNTS = (10.0, 11.0, 12.0, 13.0, 13.5)
MODE1E_DEFAULT_P2_RADIUS_FACTORS = (0.75, 1.0)
MODE1E_KR_MATCH_RTOL = 1.0e-6
MODE1E_F0_F2_CONSISTENCY_MIN = 0.5
MODE1E_SHORTLIST_MIN_ANGULAR_CORRELATION = 0.40
MODE1E_Z_PERSISTENCE_MIN_FRACTION = 0.5
MODE1E_SCOPE_STATEMENT = (
    "MODE 1E actual redesigned downstream confirmation only: ideal canonical P2 input, "
    "redesigned pre-axicon k_r and consistent redesigned target/surface k_r, propagated "
    "through the actual inherited MODE 1 downstream machinery (vector axicon -> scalar "
    "per-component focus bridge -> vector ASM z-stack, F2 vectorial reference for "
    "shortlisted candidates). No patterned-HWP/QWP/dual-SLM physical routes, no 4F "
    "carrier/iris, no panel realism, no waveplate errors, no route ranking. "
    "Proxy-only, source-template, or P2 Jones passes cannot produce M1E-A."
)


@dataclass(frozen=True)
class Mode1EDesignCandidate:
    """One analytically designed MODE 1E redesigned-downstream candidate."""

    candidate_id: str
    target_ring_count: float
    p2_radius_m: float
    k_r_pre_m_inv: float
    k_r_surface_m_inv: float
    surface_na_required: float
    within_objective_na: bool
    pre_phase_period_m: float
    slm_phase_period_ok: bool
    axicon_base_angle_deg: float
    target_core_diameter_m: float
    config_status: str
    run_status: str
    symmetry_class: str | None
    template_gate_pass: bool | None
    failure_reason: tuple[str, ...]


@dataclass(frozen=True)
class Mode1ESourceTemplate:
    """Validated-source-style hexagonal Bessel template with ring-count control."""

    target_ring_count: float
    intensity_xy: np.ndarray
    grid: Mapping[str, Any]
    ring_radius_m: float
    dark_core_ratio: float
    symmetry_class: str
    angular_profile: np.ndarray
    metadata: Mapping[str, Any]
    z_reference_m: float = float("nan")
    x_profile: np.ndarray | None = field(default=None, repr=False, compare=False)
    y_profile: np.ndarray | None = field(default=None, repr=False, compare=False)
    symmetry: Mapping[str, Any] = field(default_factory=dict)
    accepted_hexagon: bool = False
    ring_count_actual: float = float("nan")

    def as_mode1b_template(self) -> Mode1BTargetTemplate:
        """Return the equivalent MODE 1B comparison template (no new physics)."""

        return Mode1BTargetTemplate(
            intensity_xy=self.intensity_xy,
            grid=self.grid,
            z_reference_m=float(self.z_reference_m),
            ring_radius_m=float(self.ring_radius_m),
            dark_core_ratio=float(self.dark_core_ratio),
            angular_profile=np.asarray(self.angular_profile, dtype=float),
            x_profile=np.asarray(self.x_profile, dtype=float),
            y_profile=np.asarray(self.y_profile, dtype=float),
            symmetry=dict(self.symmetry),
            classification=str(self.symmetry_class),
            metadata=dict(self.metadata),
        )


def mode1e_required_pre_kr(ring_count: float, p2_radius_m: float) -> float:
    """Pre-axicon k_r required for ``ring_count`` rings across ``p2_radius_m``."""

    return required_pre_kr_for_ring_count(float(ring_count), float(p2_radius_m))


def mode1e_surface_kr_from_mapping(k_pre: float, mapping_factor: float) -> float:
    """Surface k_r implied by the audited surface/pre mapping factor."""

    return required_surface_kr(float(k_pre), float(mapping_factor))


def mode1e_na_required(k_surface: float, wavelength_m: float) -> float:
    """Objective NA required to support ``k_surface`` at ``wavelength_m``."""

    return required_na(float(k_surface), float(wavelength_m))


def mode1e_base_angle_from_pre_kr(
    *,
    k_pre: float,
    wavelength_m: float,
    n_axicon: float,
    n_medium: float,
) -> float:
    """Physical-axicon base angle (deg) whose thin-element phase gives ``k_pre``."""

    k0 = 2.0 * np.pi / float(wavelength_m)
    denom = k0 * (float(n_axicon) - float(n_medium))
    return float(np.rad2deg(np.arctan(float(k_pre) / max(denom, EPS))))


def _mode1e_candidate_id(target_ring_count: float, p2_radius_m: float) -> str:
    text = f"mode1e_N{float(target_ring_count):04.1f}_R{float(p2_radius_m) / 1e-6:06.0f}um"
    return text.replace(".", "p")


def make_mode1e_redesigned_config(
    base: NathanHexagonConfig,
    *,
    p2_radius_m: float,
    target_ring_count: float,
    grid_n: int | None = None,
    z_planes: int | None = None,
    kr_match_rtol: float = MODE1E_KR_MATCH_RTOL,
) -> tuple[NathanHexagonConfig | None, Mode1EDesignCandidate]:
    """Build an actual redesigned ``NathanHexagonConfig``, not a proxy.

    Both the physical-axicon base angle (pre-axicon k_r) and the target design
    (surface k_r) are changed consistently under the audited MODE 1C mapping
    factor, and the P2 beam radius is moved onto the requested radius.  The
    resolved parameters are then re-audited through the same
    :func:`resolve_vector_axicon_parameters` path the downstream run uses; a
    mismatch is reported as ``surface_kr_fingerprint_mismatch`` and the
    candidate is blocked instead of silently continued.
    """

    mapping = audit_mode1c_kr_mapping(base)
    twin = base.twin
    r = float(p2_radius_m)
    n_ring = float(target_ring_count)
    k_pre = mode1e_required_pre_kr(n_ring, r)
    k_surface = mode1e_surface_kr_from_mapping(k_pre, mapping.k_scale_surface_over_pre)
    na_required = mode1e_na_required(k_surface, mapping.wavelength_m)
    within_na = bool(na_required <= float(mapping.objective_na) + 1e-15)
    pre_phase_period = float(2.0 * np.pi / max(abs(k_pre), EPS))
    min_slm_phase_period = 4.0 * float(twin.slm.pixel_pitch_m)
    slm_ok = bool(pre_phase_period >= min_slm_phase_period)
    n_axicon = float(twin.target.n_axicon if twin.physical_axicon.n_axicon is None else twin.physical_axicon.n_axicon)
    n_medium = float(twin.physical_axicon.axicon_medium_n)
    beta_deg = mode1e_base_angle_from_pre_kr(
        k_pre=k_pre,
        wavelength_m=float(mapping.wavelength_m),
        n_axicon=n_axicon,
        n_medium=n_medium,
    )
    target_core_diameter_m = float(2.0 * J0_FIRST_ZERO / max(k_surface, EPS))

    failures: list[str] = []
    if not within_na:
        failures.append("surface NA exceeds objective NA")
    if not slm_ok:
        failures.append("pre-axicon phase period too fine for 4-pixel SLM encoding")
    candidate = Mode1EDesignCandidate(
        candidate_id=_mode1e_candidate_id(n_ring, r),
        target_ring_count=n_ring,
        p2_radius_m=r,
        k_r_pre_m_inv=float(k_pre),
        k_r_surface_m_inv=float(k_surface),
        surface_na_required=float(na_required),
        within_objective_na=within_na,
        pre_phase_period_m=pre_phase_period,
        slm_phase_period_ok=slm_ok,
        axicon_base_angle_deg=float(beta_deg),
        target_core_diameter_m=target_core_diameter_m,
        config_status="infeasible_by_budget",
        run_status="infeasible_not_run",
        symmetry_class=None,
        template_gate_pass=False,
        failure_reason=tuple(failures),
    )
    if failures:
        return None, candidate

    twin2 = make_mode1c_twin_with_target_kr(
        make_mode1c_twin_with_axicon_base_angle(twin, beta_deg),
        k_surface,
    )
    # The P2 Gaussian envelope radius is controlled by LaserConfig.beam_radius_on_slm_m;
    # VectorArmConfig has no beam_radius_m field, its waist_m is re-derived from the twin.
    twin2 = replace(twin2, laser=replace(twin2.laser, beam_radius_on_slm_m=r))
    cfg2 = NathanHexagonConfig.from_existing_digital_twin_baseline(
        twin2,
        baseline_preset=f"{base.baseline_preset}_mode1e_redesigned",
        vector=base.vector,
        grid_n=int(base.grid_n if grid_n is None else grid_n),
        z_planes=int(base.z_planes if z_planes is None else z_planes),
        z_span_factor=float(base.z_span_factor),
        angular_samples=int(base.angular_samples),
        route_grid_side_factor=float(base.route_grid_side_factor),
    )
    params = resolve_vector_axicon_parameters(cfg2.twin)
    pre_rel = abs(float(params.k_r_pre_m_inv) - k_pre) / max(abs(k_pre), EPS)
    surface_rel = abs(float(params.k_r_surface_m_inv) - k_surface) / max(abs(k_surface), EPS)
    if pre_rel > float(kr_match_rtol) or surface_rel > float(kr_match_rtol):
        candidate = replace(
            candidate,
            config_status="surface_kr_fingerprint_mismatch",
            run_status="blocked_fingerprint_mismatch",
            template_gate_pass=False,
            failure_reason=(
                f"resolved k_r does not match requested redesign (pre rel err {pre_rel:.3e}, "
                f"surface rel err {surface_rel:.3e})",
            ),
        )
        return cfg2, candidate
    candidate = replace(
        candidate,
        config_status="redesigned_config_resolved",
        run_status="not_run",
        template_gate_pass=None,
        failure_reason=(),
    )
    return cfg2, candidate


def build_mode1e_source_template(
    target_ring_count: float,
    *,
    grid_n: int = 384,
    z_planes: int = 21,
    v0_template: Mode1BTargetTemplate | None = None,
    base_config: NathanSourceParityConfig | None = None,
    strict: bool = True,
) -> Mode1ESourceTemplate:
    """Build a validated-source-style template at a controlled ring count.

    This reuses the MODE 1D lower-ring source machinery (V0 conventions, literal
    segmented RA input, free-space vector axicon) and packages the declared
    reference plane as a scale-normalised comparison template.  ``strict=True``
    raises if the template itself does not classify as an accepted visual
    hexagon, so a broken template cannot silently pass candidates.
    """

    case = run_mode1d_source_ring_count_case(
        float(target_ring_count),
        v0_template,
        base_config=base_config,
        grid_n=int(grid_n),
        z_planes=int(z_planes),
    )
    plane = np.asarray(case.intensity_xy, dtype=float)
    full_grid = dict(case.grid)
    ring = float(case.template_score.candidate_ring_radius_m)
    window_m = float(full_grid["N"]) * float(full_grid["dx"])
    # The polar-signature comparison samples out to 2.2 x ring radius, so the
    # crop must keep at least that coverage around the centre.
    crop_fraction = float(np.clip(5.2 * ring / max(window_m, EPS), 0.10, 1.0))
    crop_xy, crop_grid = _mode1b_even_axis_crop(plane, full_grid, crop_fraction)
    dark = _mode1b_dark_core_ratio_for_ring(crop_xy, crop_grid, ring)
    sym = _mode1_symmetry(crop_xy, crop_grid, ring)
    cls = mode1_symmetry_class(sym, dark)
    if strict and (cls != "visual_hexagonal_field" or not bool(case.accepted_hexagon)):
        raise ValueError(
            f"MODE 1E source template N={float(target_ring_count):g} did not build as an accepted "
            f"visual hexagon (template class {cls!r}, sweep accepted={bool(case.accepted_hexagon)})."
        )
    _, angular = angular_profile_on_ring(crop_xy, crop_grid, ring)
    return Mode1ESourceTemplate(
        target_ring_count=float(target_ring_count),
        intensity_xy=np.asarray(crop_xy, dtype=np.float32),
        grid=crop_grid,
        ring_radius_m=ring,
        dark_core_ratio=float(dark),
        symmetry_class=cls,
        angular_profile=np.asarray(angular, dtype=float),
        metadata={
            "stage": MODE1E_STAGE,
            "source": "MODE 1D validated-source-style lower-ring machinery",
            "grid_n": int(grid_n),
            "z_planes": int(z_planes),
            "crop_fraction": crop_fraction,
            "k_r_m_inv": float(case.k_r_m_inv),
            "axicon_base_angle_deg": float(case.axicon_base_angle_deg),
            "z_reference_m": float(case.z_reference_m),
            "sweep_classification_full_plane": str(case.classification),
            "sweep_accepted_hexagon": bool(case.accepted_hexagon),
            "v0_template_angular_correlation": float(case.template_score.angular_profile_correlation),
        },
        z_reference_m=float(case.z_reference_m),
        x_profile=_normalised_axis_profile(crop_xy, crop_grid, ring, axis="x"),
        y_profile=_normalised_axis_profile(crop_xy, crop_grid, ring, axis="y"),
        symmetry={str(k): v for k, v in sym.items()},
        accepted_hexagon=bool(case.accepted_hexagon),
        ring_count_actual=float(case.ring_count_actual),
    )


@dataclass(frozen=True)
class Mode1ECandidateResult:
    """One MODE 1E candidate outcome (design + actual downstream run, if any)."""

    candidate: Mode1EDesignCandidate
    tier: str = "redesigned_candidate"
    config: NathanHexagonConfig | None = field(default=None, repr=False, compare=False)
    z_values_m: np.ndarray | None = field(default=None, repr=False, compare=False)
    reference_index: int = 0
    reference_z_m: float = float("nan")
    output_grid: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    p2_radius_measured_1e_m: float = float("nan")
    ring_count_actual_p2: float = float("nan")
    resolved_k_r_pre_m_inv: float = float("nan")
    resolved_k_r_surface_m_inv: float = float("nan")
    kr_pre_match_rel_error: float = float("nan")
    kr_surface_match_rel_error: float = float("nan")
    reference_plane: np.ndarray | None = field(default=None, repr=False, compare=False)
    xz_map: np.ndarray | None = field(default=None, repr=False, compare=False)
    yz_map: np.ndarray | None = field(default=None, repr=False, compare=False)
    ring_radius_m: float = float("nan")
    dark_core_ratio: float = float("nan")
    symmetry: Mapping[str, Any] = field(default_factory=dict)
    f0: DownstreamRouteResult | None = field(default=None, repr=False, compare=False)
    f0_survival: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    template_scores: Mapping[str, Mode1BTemplateScore] = field(default_factory=dict)
    fail_reasons: tuple[str, ...] = ()
    f2: DownstreamRouteResult | None = field(default=None, repr=False, compare=False)
    f2_survival: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    f0_vs_f2: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def run_status(self) -> str:
        return str(self.candidate.run_status)

    @property
    def symmetry_class(self) -> str | None:
        return self.candidate.symmetry_class

    @property
    def template_gate_pass(self) -> bool | None:
        return self.candidate.template_gate_pass

    @property
    def is_actual_downstream_run(self) -> bool:
        return self.run_status.startswith("actual_downstream")

    @property
    def f2_run(self) -> bool:
        return self.f0_vs_f2 is not None

    @property
    def f0_f2_consistent(self) -> bool | None:
        if self.f0_vs_f2 is None:
            return None
        return bool(float(self.f0_vs_f2.get("reference_full_field_correlation", 0.0)) >= MODE1E_F0_F2_CONSISTENCY_MIN)


def mode1e_template_gate(
    *,
    symmetry_class: str,
    dark_core_ratio: float,
    template_score: Mode1BTemplateScore,
    f0_survival: Mapping[str, Any],
    z_persistence_min_fraction: float = MODE1E_Z_PERSISTENCE_MIN_FRACTION,
) -> tuple[bool, tuple[str, ...]]:
    """MODE 1E pass gate against the N-ring source template (MODE 1D criteria).

    The reference plane must classify as a genuine visual hexagon (the C3 veto,
    six-sector balance, and ring-island veto live inside the classifier), match
    the source template, stay hollow, and persist across the adjacent z-stack.
    """

    reasons: list[str] = []
    if str(symmetry_class) != "visual_hexagonal_field":
        reasons.append(f"reference-plane class is {symmetry_class}, not visual_hexagonal_field")
    if float(dark_core_ratio) > MODE1D_ACCEPTABLE_DARK_CORE_RATIO:
        reasons.append("reference-plane core not sufficiently dark")
    if float(template_score.angular_profile_correlation) < MODE1D_ACCEPTABLE_TEMPLATE_CORRELATION:
        reasons.append("low angular-profile similarity to the source template")
    if template_score.xy_correlation is None or float(template_score.xy_correlation) < MODE1D_ACCEPTABLE_XY_CORRELATION:
        reasons.append("low scale/rotation-normalised XY similarity to the source template")
    sym_cls = dict(f0_survival.get("symmetry_classification", {}))
    frac_hex = float(sym_cls.get("fraction_planes_visual_hexagonal", 0.0))
    if frac_hex < float(z_persistence_min_fraction):
        reasons.append("visual hexagon does not persist across the adjacent z-stack")
    return (len(reasons) == 0), tuple(reasons)


def _mode1e_template_scores(
    plane: np.ndarray,
    grid: Mapping[str, Any],
    ring_radius_m: float,
    templates: Mapping[str, Mode1ESourceTemplate],
) -> dict[str, Mode1BTemplateScore]:
    return {
        str(key): compare_to_v0_template(plane, grid, tmpl.as_mode1b_template(), candidate_ring_radius_m=float(ring_radius_m))
        for key, tmpl in templates.items()
    }


def _mode1e_f0_vs_f2_metrics(f0: DownstreamRouteResult, f2: DownstreamRouteResult, reference_index: int) -> dict[str, Any]:
    ref = int(reference_index)
    full = _equal_power_shape_metrics(f0.intensity_stack[ref][None, ...], f2.intensity_stack[ref][None, ...], crop_fraction=1.0)
    crop = _equal_power_shape_metrics(f0.intensity_stack[ref][None, ...], f2.intensity_stack[ref][None, ...], crop_fraction=0.5)
    return {
        "reference_full_field_correlation": float(full["equal_power_intensity_correlation"]),
        "reference_full_field_shape_rms": float(full["equal_power_shape_rms"]),
        "reference_central_crop_correlation": float(crop["equal_power_intensity_correlation"]),
        "reference_central_crop_shape_rms": float(crop["equal_power_shape_rms"]),
        "f0_mean_ez_fraction": float(np.mean(f0.ez_energy_fraction)),
        "f2_mean_ez_fraction": float(np.mean(f2.ez_energy_fraction)),
    }


def run_mode1e_candidate(
    base_config: NathanHexagonConfig,
    templates: Mapping[str, Mode1ESourceTemplate],
    *,
    target_ring_count: float,
    p2_radius_m: float,
    primary_template_key: str = "N12",
    grid_n: int = 256,
    z_planes: int = 13,
    run_f2: bool = False,
    f2_solver: str = "fft",
    f2_chunk_size: int = 128,
    reference_z_index: int | None = None,
) -> Mode1ECandidateResult:
    """Run one MODE 1E candidate through the ACTUAL redesigned downstream path.

    The candidate is first built as a fully-resolved redesigned config; infeasible
    or fingerprint-mismatched candidates are returned unrun with an explicit
    status.  Feasible candidates propagate the ideal canonical P2 field through
    the inherited MODE 1 machinery (F0; optional F2 vectorial reference) and are
    scored against the source templates on the declared reference plane plus the
    adjacent z-stack.  No free-space proxy is used anywhere in this runner.
    """

    if primary_template_key not in templates:
        raise KeyError(f"primary template {primary_template_key!r} missing from templates {sorted(templates)}")
    cfg2, cand = make_mode1e_redesigned_config(
        base_config,
        p2_radius_m=float(p2_radius_m),
        target_ring_count=float(target_ring_count),
        grid_n=int(grid_n),
        z_planes=int(z_planes),
    )
    if cfg2 is None or cand.config_status != "redesigned_config_resolved":
        return Mode1ECandidateResult(candidate=cand, config=cfg2, fail_reasons=tuple(cand.failure_reason))

    twin2 = _twin_with_axial_points(cfg2.twin, cfg2.z_planes)
    z_values = air_z_values(twin2, planes=cfg2.z_planes, span_factor=cfg2.z_span_factor)
    ref = int(z_values.size // 2) if reference_z_index is None else int(np.clip(reference_z_index, 0, z_values.size - 1))

    grid = default_nathan_grid(cfg2)
    p2_field = canonical_target_field(cfg2, grid=grid)
    p2_radius_measured = one_over_e_field_radius_from_vector_field(p2_field, grid)
    params = resolve_vector_axicon_parameters(twin2)
    pre_rel = abs(float(params.k_r_pre_m_inv) - cand.k_r_pre_m_inv) / max(abs(cand.k_r_pre_m_inv), EPS)
    surface_rel = abs(float(params.k_r_surface_m_inv) - cand.k_r_surface_m_inv) / max(abs(cand.k_r_surface_m_inv), EPS)

    f0 = _vector_downstream_result(
        p2_field, twin2, z_values,
        control_id="nathan_six_sector",
        route_id="F0_redesigned_scalar_focus_bridge",
        route_role="mode1e_actual_redesigned_downstream",
        expected_surface_kr_m_inv=cand.k_r_surface_m_inv,
    )
    focal_grid = _mode1_focal_grid(f0)
    f0_survival = mode1_hexagonal_bessel_survival_metrics(f0.intensity_stack, z_values, focal_grid, reference_index=ref)
    plane = np.asarray(f0.intensity_stack[ref], dtype=float)
    diag = _v0_plane_diagnostics(plane, focal_grid)
    ring = float(diag["ring_radius_m"])
    dark = float(diag["central_core_darkness"])
    sym = _mode1_symmetry(plane, focal_grid, ring)
    cls = mode1_symmetry_class(sym, dark)
    scores = _mode1e_template_scores(plane, focal_grid, ring, templates)
    gate_pass, gate_reasons = mode1e_template_gate(
        symmetry_class=cls,
        dark_core_ratio=dark,
        template_score=scores[primary_template_key],
        f0_survival=f0_survival,
    )

    f2: DownstreamRouteResult | None = None
    f2_survival: Mapping[str, Any] | None = None
    f0_vs_f2: Mapping[str, Any] | None = None
    run_status = "actual_downstream_f0"
    if run_f2:
        f2 = _vectorial_pupil_spectrum_reference_result(
            p2_field, twin2, z_values,
            control_id="nathan_six_sector",
            solver=f2_solver,
            chunk_size=int(f2_chunk_size),
            expected_surface_kr_m_inv=cand.k_r_surface_m_inv,
        )
        f2_focal_grid = _mode1_focal_grid(f2)
        f2_survival = mode1_hexagonal_bessel_survival_metrics(f2.intensity_stack, z_values, f2_focal_grid, reference_index=ref)
        f0_vs_f2 = _mode1e_f0_vs_f2_metrics(f0, f2, ref)
        run_status = "actual_downstream_f0_f2"

    mid = plane.shape[0] // 2
    cand = replace(
        cand,
        run_status=run_status,
        symmetry_class=cls,
        template_gate_pass=bool(gate_pass),
        failure_reason=tuple(gate_reasons),
    )
    return Mode1ECandidateResult(
        candidate=cand,
        tier="redesigned_candidate",
        config=cfg2,
        z_values_m=z_values,
        reference_index=ref,
        reference_z_m=float(z_values[ref]),
        output_grid=focal_grid,
        p2_radius_measured_1e_m=float(p2_radius_measured),
        ring_count_actual_p2=effective_ring_count_for_plane(radius_1e_field_m=float(p2_radius_measured), k_r_m_inv=float(params.k_r_pre_m_inv)),
        resolved_k_r_pre_m_inv=float(params.k_r_pre_m_inv),
        resolved_k_r_surface_m_inv=float(params.k_r_surface_m_inv),
        kr_pre_match_rel_error=float(pre_rel),
        kr_surface_match_rel_error=float(surface_rel),
        reference_plane=plane.astype(np.float32),
        xz_map=np.asarray(f0.intensity_stack[:, mid, :], dtype=np.float32),
        yz_map=np.asarray(f0.intensity_stack[:, :, mid], dtype=np.float32),
        ring_radius_m=ring,
        dark_core_ratio=dark,
        symmetry={str(k): v for k, v in sym.items()},
        f0=f0,
        f0_survival=f0_survival,
        template_scores=scores,
        fail_reasons=tuple(gate_reasons),
        f2=f2,
        f2_survival=f2_survival,
        f0_vs_f2=f0_vs_f2,
        metadata={
            "model_family": "actual_inherited_downstream_machinery_with_redesigned_parameters",
            "p2_window_side_m": float(grid["N"]) * float(grid["dx"]),
            "p2_grid_n": int(grid["N"]),
            "p2_grid_dx_m": float(grid["dx"]),
            "output_grid_n": int(focal_grid["N"]),
            "output_grid_dx_m": float(focal_grid["dx"]),
            "primary_template_key": str(primary_template_key),
        },
    )


def add_mode1e_f2_reference(
    result: Mode1ECandidateResult,
    *,
    f2_solver: str = "fft",
    f2_chunk_size: int = 128,
) -> Mode1ECandidateResult:
    """Add the F2 vectorial pupil-spectrum reference to a shortlisted F0 result."""

    if not result.is_actual_downstream_run or result.config is None or result.f0 is None or result.z_values_m is None:
        raise ValueError("add_mode1e_f2_reference requires a completed actual F0 candidate run")
    if result.f0_vs_f2 is not None:
        return result
    cfg2 = result.config
    twin2 = _twin_with_axial_points(cfg2.twin, cfg2.z_planes)
    grid = default_nathan_grid(cfg2)
    p2_field = canonical_target_field(cfg2, grid=grid)
    f2 = _vectorial_pupil_spectrum_reference_result(
        p2_field, twin2, result.z_values_m,
        control_id="nathan_six_sector",
        solver=f2_solver,
        chunk_size=int(f2_chunk_size),
        expected_surface_kr_m_inv=result.candidate.k_r_surface_m_inv,
    )
    f2_focal_grid = _mode1_focal_grid(f2)
    f2_survival = mode1_hexagonal_bessel_survival_metrics(
        f2.intensity_stack, result.z_values_m, f2_focal_grid, reference_index=result.reference_index,
    )
    f0_vs_f2 = _mode1e_f0_vs_f2_metrics(result.f0, f2, result.reference_index)
    cand = replace(result.candidate, run_status="actual_downstream_f0_f2")
    return replace(result, candidate=cand, f2=f2, f2_survival=f2_survival, f0_vs_f2=f0_vs_f2)


def run_mode1e_current_inherited_control(
    base_config: NathanHexagonConfig | None = None,
    templates: Mapping[str, Mode1ESourceTemplate] | None = None,
    *,
    primary_template_key: str = "N12",
    grid_n: int = 256,
    z_planes: int = 13,
    reference_z_index: int | None = None,
) -> Mode1ECandidateResult:
    """Run the unmodified inherited configuration as the MODE 1E failure control."""

    base = base_config or NathanHexagonConfig.fast()
    cfg = replace(base, grid_n=int(grid_n), z_planes=int(z_planes))
    twin = _twin_with_axial_points(cfg.twin, cfg.z_planes)
    z_values = air_z_values(twin, planes=cfg.z_planes, span_factor=cfg.z_span_factor)
    ref = int(z_values.size // 2) if reference_z_index is None else int(np.clip(reference_z_index, 0, z_values.size - 1))
    grid = default_nathan_grid(cfg)
    p2_field = canonical_target_field(cfg, grid=grid)
    p2_radius_measured = one_over_e_field_radius_from_vector_field(p2_field, grid)
    params = resolve_vector_axicon_parameters(twin)

    f0 = _vector_downstream_result(
        p2_field, twin, z_values,
        control_id="nathan_six_sector",
        route_id="F0_current_scalar_focus_bridge",
        route_role="mode1e_current_inherited_control",
    )
    focal_grid = _mode1_focal_grid(f0)
    f0_survival = mode1_hexagonal_bessel_survival_metrics(f0.intensity_stack, z_values, focal_grid, reference_index=ref)
    plane = np.asarray(f0.intensity_stack[ref], dtype=float)
    diag = _v0_plane_diagnostics(plane, focal_grid)
    ring = float(diag["ring_radius_m"])
    dark = float(diag["central_core_darkness"])
    sym = _mode1_symmetry(plane, focal_grid, ring)
    cls = mode1_symmetry_class(sym, dark)
    scores = _mode1e_template_scores(plane, focal_grid, ring, dict(templates or {}))
    gate_pass: bool | None = None
    gate_reasons: tuple[str, ...] = ()
    if templates and primary_template_key in dict(templates):
        gate_pass, gate_reasons = mode1e_template_gate(
            symmetry_class=cls,
            dark_core_ratio=dark,
            template_score=scores[str(primary_template_key)],
            f0_survival=f0_survival,
        )
    mid = plane.shape[0] // 2
    cand = Mode1EDesignCandidate(
        candidate_id="mode1e_current_inherited_control",
        target_ring_count=effective_ring_count_for_plane(radius_1e_field_m=float(p2_radius_measured), k_r_m_inv=float(params.k_r_pre_m_inv)),
        p2_radius_m=float(p2_radius_measured),
        k_r_pre_m_inv=float(params.k_r_pre_m_inv),
        k_r_surface_m_inv=float(params.k_r_surface_m_inv),
        surface_na_required=mode1e_na_required(float(params.k_r_surface_m_inv), float(twin.laser.wavelength_m)),
        within_objective_na=True,
        pre_phase_period_m=float(2.0 * np.pi / max(abs(float(params.k_r_pre_m_inv)), EPS)),
        slm_phase_period_ok=True,
        axicon_base_angle_deg=float(np.rad2deg(params.base_angle_rad)),
        target_core_diameter_m=float(twin.target.target_core_diameter_m),
        config_status="inherited_reference",
        run_status="actual_downstream_f0",
        symmetry_class=cls,
        template_gate_pass=gate_pass,
        failure_reason=tuple(gate_reasons),
    )
    return Mode1ECandidateResult(
        candidate=cand,
        tier="current_inherited_control",
        config=cfg,
        z_values_m=z_values,
        reference_index=ref,
        reference_z_m=float(z_values[ref]),
        output_grid=focal_grid,
        p2_radius_measured_1e_m=float(p2_radius_measured),
        ring_count_actual_p2=float(cand.target_ring_count),
        resolved_k_r_pre_m_inv=float(params.k_r_pre_m_inv),
        resolved_k_r_surface_m_inv=float(params.k_r_surface_m_inv),
        kr_pre_match_rel_error=0.0,
        kr_surface_match_rel_error=0.0,
        reference_plane=plane.astype(np.float32),
        xz_map=np.asarray(f0.intensity_stack[:, mid, :], dtype=np.float32),
        yz_map=np.asarray(f0.intensity_stack[:, :, mid], dtype=np.float32),
        ring_radius_m=ring,
        dark_core_ratio=dark,
        symmetry={str(k): v for k, v in sym.items()},
        f0=f0,
        f0_survival=f0_survival,
        template_scores=scores,
        fail_reasons=tuple(gate_reasons),
        metadata={"model_family": "actual_inherited_downstream_machinery_unmodified_control"},
    )


def mode1e_candidate_row(result: Mode1ECandidateResult) -> dict[str, Any]:
    """Flatten one MODE 1E candidate result into a CSV/JSON-safe row."""

    cand = result.candidate
    score12 = result.template_scores.get("N12")
    score31 = result.template_scores.get("N31")
    sym = dict(result.symmetry)
    sym_cls = dict(result.f0_survival.get("symmetry_classification", {})) if result.f0_survival else {}
    f0f2 = dict(result.f0_vs_f2) if result.f0_vs_f2 is not None else {}
    return {
        "candidate_id": str(cand.candidate_id),
        "tier": str(result.tier),
        "target_ring_count": float(cand.target_ring_count),
        "p2_radius_um": float(cand.p2_radius_m / 1e-6),
        "p2_radius_measured_1e_um": float(result.p2_radius_measured_1e_m / 1e-6),
        "ring_count_actual_p2": float(result.ring_count_actual_p2),
        "requested_k_r_pre_m_inv": float(cand.k_r_pre_m_inv),
        "resolved_k_r_pre_m_inv": float(result.resolved_k_r_pre_m_inv),
        "kr_pre_match_rel_error": float(result.kr_pre_match_rel_error),
        "requested_k_r_surface_m_inv": float(cand.k_r_surface_m_inv),
        "resolved_k_r_surface_m_inv": float(result.resolved_k_r_surface_m_inv),
        "kr_surface_match_rel_error": float(result.kr_surface_match_rel_error),
        "surface_na_required": float(cand.surface_na_required),
        "within_objective_na": bool(cand.within_objective_na),
        "pre_phase_period_um": float(cand.pre_phase_period_m / 1e-6),
        "slm_phase_period_ok": bool(cand.slm_phase_period_ok),
        "axicon_base_angle_deg": float(cand.axicon_base_angle_deg),
        "target_core_diameter_um": float(cand.target_core_diameter_m / 1e-6),
        "config_status": str(cand.config_status),
        "run_status": str(cand.run_status),
        "is_actual_downstream_run": bool(result.is_actual_downstream_run),
        "reference_z_um": float(result.reference_z_m / 1e-6),
        "ring_radius_um": float(result.ring_radius_m / 1e-6),
        "dark_core_ratio": float(result.dark_core_ratio),
        "symmetry_class": "" if cand.symmetry_class is None else str(cand.symmetry_class),
        "c60": float(sym.get("rot_corr_60", np.nan)),
        "c120": float(sym.get("rot_corr_120", np.nan)),
        "c120_minus_c60": float(sym.get("c120_minus_c60", np.nan)),
        "order3_over_order6": float(sym.get("order3_over_order6", np.nan)),
        "sector_balance_max_over_min": float(sym.get("six_sector_max_over_min", np.nan)),
        "ring_island_count": int(sym.get("ring_island_count", -1)),
        "fraction_planes_visual_hexagonal": float(sym_cls.get("fraction_planes_visual_hexagonal", np.nan)),
        "fraction_planes_triangular_lobed": float(sym_cls.get("fraction_planes_triangular_lobed", np.nan)),
        "template_n12_angular_correlation": float(score12.angular_profile_correlation) if score12 else np.nan,
        "template_n12_xy_correlation": float(score12.xy_correlation) if score12 and score12.xy_correlation is not None else np.nan,
        "template_n12_x_profile_correlation": float(score12.x_profile_correlation) if score12 else np.nan,
        "template_n12_y_profile_correlation": float(score12.y_profile_correlation) if score12 else np.nan,
        "template_n31_angular_correlation": float(score31.angular_profile_correlation) if score31 else np.nan,
        "template_n31_xy_correlation": float(score31.xy_correlation) if score31 and score31.xy_correlation is not None else np.nan,
        "template_gate_pass": "" if cand.template_gate_pass is None else bool(cand.template_gate_pass),
        "f2_run": bool(result.f2_run),
        "f0_f2_full_field_correlation": float(f0f2.get("reference_full_field_correlation", np.nan)),
        "f0_f2_consistent": "" if result.f0_f2_consistent is None else bool(result.f0_f2_consistent),
        "failure_reason": "; ".join(cand.failure_reason),
    }


def mode1e_outcome_report(
    *,
    mapping: Mode1CKMapping,
    aperture: Mode1CApertureRingLimit,
    results: Sequence[Mode1ECandidateResult],
    control: Mode1ECandidateResult | None = None,
    template_n12: Mode1ESourceTemplate | None = None,
    f0_f2_min_correlation: float = MODE1E_F0_F2_CONSISTENCY_MIN,
) -> dict[str, Any]:
    """Choose exactly one M1E-A/B/C/D outcome from actual-run evidence only.

    Only ``tier == "redesigned_candidate"`` results whose ``run_status`` starts
    with ``actual_downstream`` can contribute to M1E-A, and a passing candidate
    must additionally have a consistent F2 reference.  Proxy-only, source-template,
    and P2 Jones passes are structurally excluded.
    """

    cands = [r for r in results if str(r.tier) == "redesigned_candidate"]
    actual = [r for r in cands if r.is_actual_downstream_run]
    non_actual_passes = [r for r in cands if not r.is_actual_downstream_run and r.template_gate_pass is True]
    gate_passes = [r for r in actual if r.template_gate_pass is True]

    def _f2_corr(r: Mode1ECandidateResult) -> float:
        return float((r.f0_vs_f2 or {}).get("reference_full_field_correlation", np.nan))

    confirmed = [r for r in gate_passes if r.f0_vs_f2 is not None and _f2_corr(r) >= float(f0_f2_min_correlation)]
    f2_inconsistent = [r for r in gate_passes if r.f0_vs_f2 is not None and _f2_corr(r) < float(f0_f2_min_correlation)]
    f2_missing = [r for r in gate_passes if r.f0_vs_f2 is None]
    fingerprint_blocked = [r for r in cands if r.candidate.config_status == "surface_kr_fingerprint_mismatch"]
    feasible = [r for r in cands if r.candidate.within_objective_na and r.candidate.slm_phase_period_ok]

    if confirmed:
        outcome = "M1E-A"
        statement = (
            "A legitimate actual redesigned downstream configuration produces a visually acceptable "
            "micro-scale version of the N=12 source-template hexagonal Bessel target, with a consistent "
            "F2 vectorial reference. MODE 2A/2B may begin ONLY for this redesigned configuration, not for "
            "the old inherited geometry."
        )
    elif f2_inconsistent or f2_missing:
        outcome = "M1E-D"
        statement = (
            "A redesigned candidate is feasible by budget and passes the F0 source-template gate, but the "
            "F0/F2 downstream models are not mutually confirmed (inconsistent or missing F2 reference). "
            "Resolve the downstream propagation model before any physical route work."
        )
    elif feasible and not actual and fingerprint_blocked:
        outcome = "M1E-C"
        statement = (
            "The code/model cannot run a legitimate actual redesigned downstream candidate: the resolved "
            "k_r/P2/target-plane semantics disagree with the requested redesign (locked fingerprint). "
            "Refactor model semantics before any physical route work."
        )
    else:
        outcome = "M1E-B"
        statement = (
            "Actual redesigned downstream configurations run legitimately but the sample-plane output "
            "remains triangular/lobed or otherwise non-hexagonal against the N=12 source template. "
            "MODE 2A/2B remains blocked."
        )

    def _summary(r: Mode1ECandidateResult) -> dict[str, Any]:
        score12 = r.template_scores.get("N12")
        return {
            "candidate_id": str(r.candidate.candidate_id),
            "run_status": str(r.run_status),
            "symmetry_class": r.candidate.symmetry_class,
            "template_gate_pass": r.candidate.template_gate_pass,
            "template_n12_angular_correlation": float(score12.angular_profile_correlation) if score12 else None,
            "f0_f2_full_field_correlation": None if r.f0_vs_f2 is None else _f2_corr(r),
            "failure_reason": list(r.candidate.failure_reason),
        }

    best_actual = max(
        actual,
        key=lambda r: float(r.template_scores["N12"].angular_profile_correlation) if "N12" in r.template_scores else -np.inf,
        default=None,
    )
    return {
        "stage": MODE1E_STAGE,
        "suggested_outcome": outcome,
        "allowed_outcomes": MODE1E_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "scope": MODE1E_SCOPE_STATEMENT,
        "mode2a_2b_realisation_allowed": bool(outcome == "M1E-A"),
        "mode2a_2b_gate": (
            "open_only_for_confirmed_redesigned_configuration" if outcome == "M1E-A" else "blocked"
        ),
        "confirmed_candidate_ids": [str(r.candidate.candidate_id) for r in confirmed],
        "f0_gate_pass_candidate_ids": [str(r.candidate.candidate_id) for r in gate_passes],
        "n_candidates": int(len(cands)),
        "n_feasible_by_budget": int(len(feasible)),
        "n_actual_downstream_runs": int(len(actual)),
        "n_fingerprint_blocked": int(len(fingerprint_blocked)),
        "n_f0_gate_passes": int(len(gate_passes)),
        "n_confirmed_with_f2": int(len(confirmed)),
        "n_f2_inconsistent_passes": int(len(f2_inconsistent)),
        "n_f2_missing_passes": int(len(f2_missing)),
        "n_non_actual_gate_passes_excluded": int(len(non_actual_passes)),
        "proxy_pass_policy": "non-actual (proxy/source-template/Jones) passes are structurally excluded from M1E-A",
        "f0_f2_min_correlation": float(f0_f2_min_correlation),
        "best_actual_candidate": None if best_actual is None else _summary(best_actual),
        "control": None if control is None else _summary(control),
        "control_remains_non_hexagonal": (
            None if control is None else bool(control.candidate.symmetry_class != "visual_hexagonal_field")
        ),
        "primary_template_ring_count": None if template_n12 is None else float(template_n12.target_ring_count),
        "primary_template_accepted": None if template_n12 is None else bool(template_n12.accepted_hexagon),
        "current_mapping_factor": float(mapping.k_scale_surface_over_pre),
        "objective_na": float(mapping.objective_na),
        "current_p2_radius_m": float(aperture.p2_radius_current_m),
        "slm_safe_radius_m": float(aperture.p2_radius_max_with_safety_m),
        "mode1c_ring_budget_slm_safe_na_limited": float(aperture.ring_count_max_slm_radius_na_limited),
        "candidates": [_summary(r) for r in cands],
    }


def mode1e_scope_manifest(outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Machine-readable MODE 1E simulation-scope manifest."""

    allowed = bool(outcome.get("mode2a_2b_realisation_allowed", False)) if outcome else False
    return {
        "stage": MODE1E_STAGE,
        "scope": MODE1E_SCOPE_STATEMENT,
        "git_commit": _mode1_git_commit_short(),
        "ideal_p2_input_only": True,
        "actual_inherited_downstream_machinery": True,
        "free_space_proxy_runs_count_as_actual": False,
        "hwp_qwp_slm_panel_realism": False,
        "carrier_iris_realism": False,
        "waveplate_errors": False,
        "route_ranking": False,
        "physical_route_approval": False,
        "mode2a_2b_realisation_allowed": allowed,
        "mode2a_2b_scope_if_allowed": (
            "only the confirmed redesigned configuration; the inherited geometry remains blocked"
        ),
        "suggested_outcome": None if outcome is None else outcome.get("suggested_outcome"),
        "sample_plane_hexagon_claim": allowed,
        "blocking_dependency": (
            "MODE 2A/2B remains blocked unless the MODE 1E outcome is M1E-A from an actual redesigned "
            "downstream run confirmed by F2; proxy-only, source-template, and P2 Jones passes do not count."
        ),
        "claim_boundary": {
            "model_status": MODEL_STATUS,
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "material_model": False,
            "camera_model": False,
            "physical_generation_modelled": False,
        },
    }


def run_mode1e_redesigned_downstream(
    config: NathanHexagonConfig | None = None,
    *,
    target_ring_counts: Sequence[float] = MODE1E_DEFAULT_RING_COUNTS,
    p2_radius_factors: Sequence[float] = MODE1E_DEFAULT_P2_RADIUS_FACTORS,
    grid_n: int = 256,
    z_planes: int = 13,
    template_grid_n: int = 384,
    template_z_planes: int = 21,
    slm_safety: float = 0.90,
    run_f2_for_shortlist: bool = True,
    max_f2_candidates: int = 2,
    shortlist_min_angular_correlation: float = MODE1E_SHORTLIST_MIN_ANGULAR_CORRELATION,
    include_reference_template: bool = True,
    f2_solver: str = "fft",
    f2_chunk_size: int = 128,
) -> dict[str, Any]:
    """Run the full MODE 1E study: templates, control, candidates, F2 shortlist, outcome."""

    base = config or NathanHexagonConfig.fast()
    mapping = audit_mode1c_kr_mapping(base)
    aperture = audit_mode1c_aperture_ring_limit(base, safety=float(slm_safety))

    v0_template = build_mode1b_target_template(grid_n=int(template_grid_n), z_planes=max(21, int(template_z_planes)))
    templates: dict[str, Mode1ESourceTemplate] = {
        "N12": build_mode1e_source_template(
            MODE1E_PRIMARY_TEMPLATE_RING_COUNT,
            grid_n=int(template_grid_n),
            z_planes=int(template_z_planes),
            v0_template=v0_template,
        ),
    }
    if include_reference_template:
        templates["N31"] = build_mode1e_source_template(
            MODE1E_REFERENCE_TEMPLATE_RING_COUNT,
            grid_n=int(template_grid_n),
            z_planes=int(template_z_planes),
            v0_template=v0_template,
        )

    control = run_mode1e_current_inherited_control(
        base, templates, grid_n=int(grid_n), z_planes=int(z_planes),
    )

    results: list[Mode1ECandidateResult] = []
    for factor in p2_radius_factors:
        radius = float(factor) * float(aperture.p2_radius_max_with_safety_m)
        for n_ring in target_ring_counts:
            results.append(
                run_mode1e_candidate(
                    base,
                    templates,
                    target_ring_count=float(n_ring),
                    p2_radius_m=radius,
                    grid_n=int(grid_n),
                    z_planes=int(z_planes),
                    run_f2=False,
                )
            )

    shortlist_ids: list[str] = []
    if run_f2_for_shortlist:
        def _shortlist_key(item: tuple[int, Mode1ECandidateResult]) -> tuple[int, float]:
            score = item[1].template_scores.get("N12")
            corr = float(score.angular_profile_correlation) if score else -np.inf
            return (1 if item[1].template_gate_pass else 0, corr)
        indexed_actual = [(idx, r) for idx, r in enumerate(results) if r.is_actual_downstream_run]
        ranked = sorted(indexed_actual, key=_shortlist_key, reverse=True)
        shortlist = [
            (idx, r) for idx, r in ranked
            if r.template_gate_pass
            or float(r.template_scores["N12"].angular_profile_correlation) >= float(shortlist_min_angular_correlation)
        ][: max(0, int(max_f2_candidates))]
        if not shortlist and ranked and int(max_f2_candidates) > 0:
            # No pass / near-pass: still run F2 on the single best actual candidate so the
            # report carries one honest F0-vs-F2 consistency check for the redesigned geometry.
            shortlist = ranked[:1]
        for idx, entry in shortlist:
            upgraded = add_mode1e_f2_reference(entry, f2_solver=f2_solver, f2_chunk_size=int(f2_chunk_size))
            results[idx] = upgraded
            shortlist_ids.append(str(upgraded.candidate.candidate_id))

    outcome = mode1e_outcome_report(
        mapping=mapping,
        aperture=aperture,
        results=results,
        control=control,
        template_n12=templates["N12"],
    )
    manifest = mode1e_scope_manifest(outcome)
    return {
        "config": base,
        "mapping": mapping,
        "aperture": aperture,
        "templates": templates,
        "control": control,
        "results": results,
        "candidate_rows": tuple(mode1e_candidate_row(r) for r in [*results, control]),
        "shortlist_candidate_ids": tuple(shortlist_ids),
        "outcome": outcome,
        "manifest": manifest,
    }


# ---------------------------------------------------------------------------
# MODE 1E figures (package functions; the notebook calls these, no duplicated physics)
# ---------------------------------------------------------------------------


def plot_mode1e_source_template(
    template: Mode1ESourceTemplate,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    x_mm = np.asarray(template.grid["x"], dtype=float) / 1e-3
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)
    axes[0].imshow(_normalise_image(np.asarray(template.intensity_xy, dtype=float), local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(
        f"source template N={template.target_ring_count:g}\n"
        f"{template.symmetry_class}, dark core {template.dark_core_ratio:.2f}"
    )
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    theta = np.linspace(0.0, 360.0, int(np.asarray(template.angular_profile).size), endpoint=False)
    prof = np.asarray(template.angular_profile, dtype=float)
    axes[1].plot(theta, prof / max(float(prof.max()), EPS), color="tab:blue", lw=1.0)
    axes[1].set_title("annular angular profile at ring radius")
    axes[1].set_xlabel("theta (deg)")
    axes[1].set_ylabel("normalised intensity")
    axes[1].set_ylim(0.0, 1.05)
    fig.suptitle(f"MODE 1E source template (accepted={template.accepted_hexagon})")
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1e_candidate_planes(
    result: Mode1ECandidateResult,
    *,
    route: str = "F0",
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    if route.upper() == "F2":
        if result.f2 is None:
            raise ValueError("candidate has no F2 stack; run add_mode1e_f2_reference first")
        stack = np.asarray(result.f2.intensity_stack, dtype=float)
    else:
        if result.f0 is None:
            raise ValueError("candidate has no F0 stack (not an actual downstream run)")
        stack = np.asarray(result.f0.intensity_stack, dtype=float)
    grid = result.output_grid
    z_um = np.asarray(result.z_values_m, dtype=float) / 1e-6
    x_um = _mode1_um_axis(grid)
    ref = int(result.reference_index)
    plane = stack[ref]
    mid = plane.shape[0] // 2
    ext_xy = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    ext_z = [float(x_um[0]), float(x_um[-1]), float(z_um[0]), float(z_um[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    axes[0].imshow(_normalise_image(plane, local=True), origin="lower", extent=ext_xy, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"{route} xy at declared reference z={z_um[ref]:.1f} um\n{result.candidate.symmetry_class}")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[1].imshow(_normalise_image(stack[:, mid, :], local=True), origin="lower", aspect="auto", extent=ext_z, cmap="inferno", vmin=0, vmax=1)
    axes[1].axhline(z_um[ref], color="white", lw=0.8, alpha=0.8)
    axes[1].set_title("x-z persistence")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("z (um)")
    axes[2].imshow(_normalise_image(stack[:, :, mid], local=True), origin="lower", aspect="auto", extent=ext_z, cmap="inferno", vmin=0, vmax=1)
    axes[2].axhline(z_um[ref], color="white", lw=0.8, alpha=0.8)
    axes[2].set_title("y-z persistence")
    axes[2].set_xlabel("y (um)")
    axes[2].set_ylabel("z (um)")
    score12 = result.template_scores.get("N12")
    corr_text = "" if score12 is None else f", N12 angular corr {score12.angular_profile_correlation:.2f}"
    fig.suptitle(
        f"MODE 1E {result.candidate.candidate_id} [{result.run_status}] "
        f"gate_pass={result.template_gate_pass}{corr_text}"
    )
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode1e_template_comparison(
    result: Mode1ECandidateResult,
    template: Mode1ESourceTemplate,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    if result.reference_plane is None:
        raise ValueError("candidate has no reference plane (not an actual downstream run)")
    plane = np.asarray(result.reference_plane, dtype=float)
    x_um = _mode1_um_axis(result.output_grid)
    ext_c = [float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])]
    x_mm = np.asarray(template.grid["x"], dtype=float) / 1e-3
    ext_t = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    axes[0].imshow(_normalise_image(plane, local=True), origin="lower", extent=ext_c, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"candidate sample plane (um)\n{result.candidate.candidate_id}")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[1].imshow(_normalise_image(np.asarray(template.intensity_xy, dtype=float), local=True), origin="lower", extent=ext_t, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title(f"source template N={template.target_ring_count:g} (mm)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    _, cand_prof = angular_profile_on_ring(plane, result.output_grid, float(result.ring_radius_m), angular_bins=int(np.asarray(template.angular_profile).size))
    tmpl_prof = np.asarray(template.angular_profile, dtype=float)
    theta = np.linspace(0.0, 360.0, tmpl_prof.size, endpoint=False)
    axes[2].plot(theta, cand_prof / max(float(np.max(cand_prof)), EPS), color="tab:red", lw=1.0, label="candidate")
    axes[2].plot(theta, tmpl_prof / max(float(np.max(tmpl_prof)), EPS), color="0.35", lw=1.0, ls="--", label="template")
    key = f"N{template.target_ring_count:g}"
    score = result.template_scores.get(key) or result.template_scores.get("N12")
    corr = float(score.angular_profile_correlation) if score else float("nan")
    axes[2].set_title(f"angular profiles (corr {corr:.2f})")
    axes[2].set_xlabel("theta (deg)")
    axes[2].set_ylabel("normalised intensity")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].legend(fontsize=8)
    fig.suptitle("MODE 1E candidate vs source template (ring-radius scale-normalised metrics)")
    _save_fig(fig, output_path)
    return fig, axes


def write_mode1e_outputs(
    config: NathanHexagonConfig | None = None,
    *,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode1e_redesigned_downstream",
    report: Mapping[str, Any] | None = None,
    **run_kwargs: Any,
) -> dict[str, Path]:
    """Run (or reuse) the MODE 1E study and write all required artefacts."""

    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = dict(report) if report is not None else run_mode1e_redesigned_downstream(config, **run_kwargs)
    results: list[Mode1ECandidateResult] = list(study["results"])
    control: Mode1ECandidateResult = study["control"]
    templates: Mapping[str, Mode1ESourceTemplate] = study["templates"]

    paths: dict[str, Path] = {
        "design_candidates_csv": out / "mode1e_design_candidates.csv",
        "design_candidates_json": out / "mode1e_design_candidates.json",
        "outcome_report": out / "mode1e_outcome_report.json",
        "scope_manifest": out / "simulation_scope_manifest.json",
    }
    rows = study["candidate_rows"]
    _write_rows(paths["design_candidates_csv"], rows)
    paths["design_candidates_json"].write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")

    for key, template in templates.items():
        name = f"source_template_{key}"
        path = out / f"mode1e_source_template_{key}.png"
        fig, _ = plot_mode1e_source_template(template, output_path=path)
        plt.close(fig)
        paths[name] = path

    control_path = out / "mode1e_current_inherited_control.png"
    fig, _ = plot_mode1e_candidate_planes(control, route="F0", output_path=control_path)
    plt.close(fig)
    paths["current_inherited_control"] = control_path

    for entry in results:
        if not entry.is_actual_downstream_run:
            continue
        cid = str(entry.candidate.candidate_id)
        f0_path = out / f"mode1e_candidate_{cid}_f0.png"
        fig, _ = plot_mode1e_candidate_planes(entry, route="F0", output_path=f0_path)
        plt.close(fig)
        paths[f"candidate_{cid}_f0"] = f0_path
        cmp_path = out / f"mode1e_template_comparison_{cid}.png"
        fig, _ = plot_mode1e_template_comparison(entry, templates["N12"], output_path=cmp_path)
        plt.close(fig)
        paths[f"template_comparison_{cid}"] = cmp_path
        if entry.f2 is not None:
            f2_path = out / f"mode1e_candidate_{cid}_f2.png"
            fig, _ = plot_mode1e_candidate_planes(entry, route="F2", output_path=f2_path)
            plt.close(fig)
            paths[f"candidate_{cid}_f2"] = f2_path

    paths["outcome_report"].write_text(json.dumps(_json_ready(dict(study["outcome"])), indent=2), encoding="utf-8")
    paths["scope_manifest"].write_text(json.dumps(_json_ready(dict(study["manifest"])), indent=2), encoding="utf-8")
    return paths


# ===========================================================================
# MODE 2N - Nathan source-scale physical bench replication
# ===========================================================================
#
# MODE 2P proved the component-level Jones synthesis at P2 but stopped before
# the axicon; MODE 1C/1E studied the inherited microfabrication downstream and
# are explicitly out of scope here.  MODE 2N asks the source-scale question:
# can a physically realistic bench (patterned HWP, or dual SLM + QWP, with the
# realistic route adding carrier + 4F first-order filtering) generate Nathan's
# segmented radial/azimuthal field and, after the SAME Nathan source-scale
# physical axicon and free-space vector propagation used by V0, reproduce the
# Fig. 4 hexagonal Bessel beam around z = 60 mm?  A route only succeeds on the
# propagated intensity, never on the pre-axicon overlap alone.

MODE2N_STAGE = "nathan_mode2n_source_scale_physical_replication"
MODE2N_ALLOWED_OUTCOMES = ("M2N-A", "M2N-B", "M2N-C", "M2N-D")
MODE2N_PRE_AXICON_OVERLAP_PASS = 0.999
MODE2N_IDEAL_PROPAGATED_CORRELATION_PASS = 0.999
MODE2N_REALISTIC_PROPAGATED_CORRELATION_PASS = 0.90
MODE2N_DEFAULT_CARRIER_LPMM = 6.25
MODE2N_DEFAULT_IRIS_RADIUS_FRAC = 0.4
MODE2N_SCOPE_STATEMENT = (
    "MODE 2N source-scale physical bench replication only: Gaussian input, HWP/dual-SLM/QWP "
    "generation (ideal routes plus a carrier + 4F first-order filtered dual-SLM route), Nathan "
    "source-scale physical axicon (n = 1.458, base angle 2 deg), and free-space vector "
    "angular-spectrum propagation observed around z = 60 mm on Nathan's axis-sampled 10 mm grid. "
    "The inherited objective/sample microfabrication geometry is NOT used and no micro-scale "
    "sample-plane claim is made or judged by MODE 1C/M1E constraints."
)


@dataclass(frozen=True)
class Mode2NRouteResult:
    """One MODE 2N generation route propagated through the Nathan source axicon."""

    route_id: str
    pre_axicon_metrics: Mapping[str, Any]
    slm_4f_report: Mapping[str, Any] | None
    z_values_m: np.ndarray = field(repr=False, compare=False, default=None)
    reference_index: int = 0
    reference_z_m: float = float("nan")
    grid: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    pre_axicon_field: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False, compare=False)
    reference_plane: np.ndarray | None = field(default=None, repr=False, compare=False)
    xz_map: np.ndarray | None = field(default=None, repr=False, compare=False)
    yz_map: np.ndarray | None = field(default=None, repr=False, compare=False)
    on_axis_intensity: np.ndarray | None = field(default=None, repr=False, compare=False)
    ring_radius_m: float = float("nan")
    dark_core_ratio: float = float("nan")
    symmetry: Mapping[str, Any] = field(default_factory=dict)
    symmetry_class: str = ""
    v0_comparison: Mapping[str, Any] = field(default_factory=dict)
    passes_v0_match: bool = False
    fail_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def mode2n_source_target(
    config: NathanSourceParityConfig | None = None,
    *,
    grid_n: int = 512,
    z_planes: int = 61,
) -> dict[str, Any]:
    """Build the Nathan source-scale target field on the V0 axis-sampled grid.

    Everything downstream of generation (grid convention, axicon constants,
    z-stack, observable) is inherited from the validated V0 source machinery so
    that route-vs-V0 differences measure only the generation hardware.
    """

    cfg = replace(config or NathanSourceParityConfig(), grid_n=int(grid_n), z_planes=int(z_planes))
    grid = source_parity_grid(cfg)
    target_field, radial_mask = nathan_literal_segmented_ra_input(
        grid,
        wavelength_m=float(cfg.wavelength_m),
        beam_radius_m=float(cfg.beam_radius_m),
        n_pairs=int(cfg.n_pairs),
        sector_theta_rad=float(cfg.sector_theta_rad),
        sector_rotation_rad=float(cfg.sector_rotation_rad),
    )
    theta = np.asarray(grid["PHI"], dtype=float)
    alpha, alpha_radial_mask = nathan_alpha_map(
        theta,
        sector_num_pairs=int(cfg.n_pairs),
        sector_theta=float(cfg.sector_theta_rad),
        sector_rotation=float(cfg.sector_rotation_rad),
    )
    R = np.asarray(grid["R"], dtype=float)
    A = np.exp(-(R**2) / max(float(cfg.beam_radius_m), EPS) ** 2)
    target = (np.asarray(target_field.ex, dtype=np.complex128), np.asarray(target_field.ey, dtype=np.complex128))
    if not np.array_equal(radial_mask, alpha_radial_mask):
        raise ValueError("MODE 2N alpha map disagrees with the V0 literal sector convention")
    if not (np.allclose(target[0], A * np.cos(alpha)) and np.allclose(target[1], A * np.sin(alpha))):
        raise ValueError("MODE 2N target arrays disagree with the V0 literal segmented RA input")
    metric_mask = A > 1.0e-6 * max(float(np.max(A)), EPS)
    return {
        "config": cfg,
        "grid": grid,
        "A": A,
        "alpha": alpha,
        "radial_mask": radial_mask,
        "target": target,
        "target_field": target_field,
        "metric_mask": metric_mask,
        "axis_sampled": bool(np.any(R <= EPS)),
    }


def _mode2n_vector_field(Ex: Any, Ey: Any, data: Mapping[str, Any]) -> VectorField:
    cfg: NathanSourceParityConfig = data["config"]
    grid = data["grid"]
    ex = np.asarray(Ex, dtype=np.complex128)
    return VectorField(
        ex=ex,
        ey=np.asarray(Ey, dtype=np.complex128),
        ez=np.zeros_like(ex),
        grid=grid,
        wavelength_m=float(cfg.wavelength_m),
        medium_index=1.0,
        metadata={"stage": MODE2N_STAGE},
    )


def mode2n_propagate_through_source_axicon(
    Ex: Any,
    Ey: Any,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the Nathan source-scale physical axicon and vector ASM z-stack (V0 path)."""

    cfg: NathanSourceParityConfig = data["config"]
    field = _mode2n_vector_field(Ex, Ey, data)
    after, axicon_meta = _apply_free_space_vector_axicon(
        field,
        n_axicon=float(cfg.axicon_n),
        n_medium=float(cfg.medium_n),
        base_angle_rad=float(cfg.axicon_base_angle_rad),
    )
    z_values = _v0_z_values(cfg)
    stack, _ = _free_space_intensity_stack(after, z_values)
    ref = _nearest_z_index(z_values, float(cfg.z_reference_m))
    return {
        "z_values_m": z_values,
        "reference_index": int(ref),
        "intensity_stack": stack,
        "axicon_meta": axicon_meta,
    }


def _mode2n_reference_plane_metrics(plane: np.ndarray, grid: Mapping[str, Any]) -> dict[str, Any]:
    diag = _v0_plane_diagnostics(plane, grid)
    ring = float(diag["ring_radius_m"])
    dark = float(diag["central_core_darkness"])
    sym = _mode1_symmetry(plane, grid, ring)
    return {
        "ring_radius_m": ring,
        "dark_core_ratio": dark,
        "symmetry": {str(k): v for k, v in sym.items()},
        "symmetry_class": mode1_symmetry_class(sym, dark),
    }


def mode2n_compare_stacks_to_v0(
    stack: np.ndarray,
    z_values_m: np.ndarray,
    reference_index: int,
    grid: Mapping[str, Any],
    v0: "Mode2NRouteResult",
) -> dict[str, Any]:
    """Compare a propagated route stack against the V0 reference on the same grid."""

    ref = int(reference_index)
    plane = np.asarray(stack[ref], dtype=float)
    v0_plane = np.asarray(v0.reference_plane, dtype=float)
    full = _equal_power_shape_metrics(plane[None, ...], v0_plane[None, ...], crop_fraction=1.0)
    crop = _equal_power_shape_metrics(plane[None, ...], v0_plane[None, ...], crop_fraction=0.5)
    own = _mode2n_reference_plane_metrics(plane, grid)
    _, prof_route = angular_profile_on_ring(plane, grid, float(own["ring_radius_m"]))
    _, prof_v0 = angular_profile_on_ring(v0_plane, grid, float(v0.ring_radius_m))
    ang_corr, ang_shift = circular_profile_correlation(prof_route, prof_v0)
    mid = plane.shape[0] // 2
    xz = np.asarray(stack[:, mid, :], dtype=float)
    xz_corr = _safe_corr(xz, np.asarray(v0.xz_map, dtype=float))
    on_axis = np.asarray(stack[:, mid, mid], dtype=float)
    on_axis_corr = _safe_corr(on_axis, np.asarray(v0.on_axis_intensity, dtype=float))
    return {
        **own,
        "z60_full_field_correlation": float(full["equal_power_intensity_correlation"]),
        "z60_full_field_shape_rms": float(full["equal_power_shape_rms"]),
        "z60_central_crop_correlation": float(crop["equal_power_intensity_correlation"]),
        "z60_central_crop_shape_rms": float(crop["equal_power_shape_rms"]),
        "angular_profile_correlation_to_v0": float(ang_corr),
        "angular_profile_best_shift_bins": int(ang_shift),
        "xz_map_correlation_to_v0": float(xz_corr),
        "on_axis_intensity_correlation_to_v0": float(on_axis_corr),
        "reference_z_m": float(np.asarray(z_values_m, dtype=float)[ref]),
    }


def mode2n_route_passes_v0_match(
    pre_axicon_overlap: float,
    comparison: Mapping[str, Any],
    *,
    ideal: bool,
) -> tuple[bool, tuple[str, ...]]:
    """A route succeeds only on the propagated intensity, never pre-axicon alone."""

    threshold = MODE2N_IDEAL_PROPAGATED_CORRELATION_PASS if ideal else MODE2N_REALISTIC_PROPAGATED_CORRELATION_PASS
    reasons: list[str] = []
    if ideal and float(pre_axicon_overlap) < MODE2N_PRE_AXICON_OVERLAP_PASS:
        reasons.append("pre-axicon complex vector overlap below 0.999")
    if float(comparison["z60_full_field_correlation"]) < float(threshold):
        reasons.append(f"z=60 mm equal-power correlation to V0 below {threshold:g}")
    if str(comparison["symmetry_class"]) != "visual_hexagonal_field":
        reasons.append(f"propagated class is {comparison['symmetry_class']}, not visual_hexagonal_field")
    return (len(reasons) == 0), tuple(reasons)


def _mode2n_route_result(
    route_id: str,
    Ex: np.ndarray,
    Ey: np.ndarray,
    data: Mapping[str, Any],
    v0: "Mode2NRouteResult",
    *,
    ideal: bool,
    slm_4f_report: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Mode2NRouteResult:
    pre = jones_metric_row(route_id, (Ex, Ey), data["target"], mask=data["metric_mask"])
    prop = mode2n_propagate_through_source_axicon(Ex, Ey, data)
    stack = prop["intensity_stack"]
    ref = int(prop["reference_index"])
    z_values = prop["z_values_m"]
    comparison = mode2n_compare_stacks_to_v0(stack, z_values, ref, data["grid"], v0)
    passes, reasons = mode2n_route_passes_v0_match(float(pre["complex_vector_overlap"]), comparison, ideal=ideal)
    mid = stack.shape[1] // 2
    return Mode2NRouteResult(
        route_id=route_id,
        pre_axicon_metrics=pre,
        slm_4f_report=None if slm_4f_report is None else dict(slm_4f_report),
        z_values_m=np.asarray(z_values, dtype=float),
        reference_index=ref,
        reference_z_m=float(z_values[ref]),
        grid=data["grid"],
        pre_axicon_field=(np.asarray(Ex, dtype=np.complex128), np.asarray(Ey, dtype=np.complex128)),
        reference_plane=np.asarray(stack[ref], dtype=np.float32),
        xz_map=np.asarray(stack[:, mid, :], dtype=np.float32),
        yz_map=np.asarray(stack[:, :, mid], dtype=np.float32),
        on_axis_intensity=np.asarray(stack[:, mid, mid], dtype=np.float32),
        ring_radius_m=float(comparison["ring_radius_m"]),
        dark_core_ratio=float(comparison["dark_core_ratio"]),
        symmetry=dict(comparison["symmetry"]),
        symmetry_class=str(comparison["symmetry_class"]),
        v0_comparison=comparison,
        passes_v0_match=bool(passes),
        fail_reasons=tuple(reasons),
        metadata={
            "ideal_route": bool(ideal),
            "axicon_meta": dict(prop["axicon_meta"]),
            **dict(extra_metadata or {}),
        },
    )


def run_mode2n_v0_reference(data: Mapping[str, Any]) -> Mode2NRouteResult:
    """Propagate the literal V0 target field itself as the replication reference."""

    Ex, Ey = data["target"]
    pre = jones_metric_row("v0_reference", (Ex, Ey), data["target"], mask=data["metric_mask"])
    prop = mode2n_propagate_through_source_axicon(Ex, Ey, data)
    stack = prop["intensity_stack"]
    ref = int(prop["reference_index"])
    z_values = prop["z_values_m"]
    plane = np.asarray(stack[ref], dtype=float)
    own = _mode2n_reference_plane_metrics(plane, data["grid"])
    mid = stack.shape[1] // 2
    comparison = {
        **own,
        "z60_full_field_correlation": 1.0,
        "z60_central_crop_correlation": 1.0,
        "angular_profile_correlation_to_v0": 1.0,
        "xz_map_correlation_to_v0": 1.0,
        "on_axis_intensity_correlation_to_v0": 1.0,
        "reference_z_m": float(z_values[ref]),
    }
    return Mode2NRouteResult(
        route_id="v0_reference",
        pre_axicon_metrics=pre,
        slm_4f_report=None,
        z_values_m=np.asarray(z_values, dtype=float),
        reference_index=ref,
        reference_z_m=float(z_values[ref]),
        grid=data["grid"],
        pre_axicon_field=(np.asarray(Ex, dtype=np.complex128), np.asarray(Ey, dtype=np.complex128)),
        reference_plane=plane.astype(np.float32),
        xz_map=np.asarray(stack[:, mid, :], dtype=np.float32),
        yz_map=np.asarray(stack[:, :, mid], dtype=np.float32),
        on_axis_intensity=np.asarray(stack[:, mid, mid], dtype=np.float32),
        ring_radius_m=float(own["ring_radius_m"]),
        dark_core_ratio=float(own["dark_core_ratio"]),
        symmetry=dict(own["symmetry"]),
        symmetry_class=str(own["symmetry_class"]),
        v0_comparison=comparison,
        passes_v0_match=True,
        fail_reasons=(),
        metadata={"ideal_route": True, "axicon_meta": dict(prop["axicon_meta"]), "role": "validated_v0_reference"},
    )


def run_mode2n_patterned_hwp_route(data: Mapping[str, Any], v0: Mode2NRouteResult) -> Mode2NRouteResult:
    """Route 2N-A: horizontal Gaussian through an ideal patterned HWP (beta = alpha/2)."""

    Ex, Ey, meta = synthesize_with_patterned_hwp(data["A"], data["alpha"])
    meta = {k: v for k, v in meta.items() if k != "beta_rad"}
    return _mode2n_route_result(
        "route_2na_patterned_hwp",
        Ex, Ey, data, v0,
        ideal=True,
        extra_metadata={"route_family": "patterned_hwp", **meta},
    )


def run_mode2n_dual_slm_qwp_route(data: Mapping[str, Any], v0: Mode2NRouteResult) -> Mode2NRouteResult:
    """Route 2N-B: ideal dual-SLM linear channels + uniform QWP (M2P selected convention)."""

    jones = route_dual_slm_linear_then_qwp_ideal(data["A"], data["alpha"], data["target"], mask=data["metric_mask"])
    return _mode2n_route_result(
        "route_2nb_dual_slm_qwp",
        jones.Ex, jones.Ey, data, v0,
        ideal=True,
        extra_metadata={
            "route_family": "dual_slm_linear_channels_then_uniform_qwp",
            "selected_h_phase_sign": jones.metadata["selected_h_phase_sign"],
            "selected_v_phase_sign": jones.metadata["selected_v_phase_sign"],
            "selected_v_piston_rad": jones.metadata["selected_v_piston_rad"],
            "selected_qwp_angle_rad": jones.metadata["selected_qwp_angle_rad"],
        },
    )


def run_mode2n_dual_slm_4f_route(
    data: Mapping[str, Any],
    v0: Mode2NRouteResult,
    *,
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM,
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    tilt_tolerance_rad: float = 1.0e-4,
) -> Mode2NRouteResult:
    """Route 2N-C: phase-only dual SLM with carrier blaze + hard 4F first-order iris + QWP.

    Each polarisation channel displays a wrapped phase-only hologram
    (``H: +alpha``, ``V: -alpha + pi/2``, both plus a linear carrier), a hard
    circular iris in the shared Fourier plane keeps the first order, the carrier
    is removed analytically, the channels recombine losslessly, and the M2P
    selected uniform QWP (-pi/4) closes the chain.  No pixel fill factor,
    quantisation, or aberrations yet (deliberately: basic carrier-filtered case
    first).
    """

    cfg: NathanSourceParityConfig = data["config"]
    grid = data["grid"]
    A = np.asarray(data["A"], dtype=float)
    alpha = np.asarray(data["alpha"], dtype=float)
    X = np.asarray(grid["X"], dtype=float)
    carrier_cpm = float(carrier_lpmm) * 1.0e3
    nyquist_cpm = 0.5 / float(grid["dx"])
    if (1.0 + float(iris_radius_frac)) * carrier_cpm >= nyquist_cpm:
        raise ValueError("carrier plus iris exceeds the grid spectral band; increase grid_n or lower the carrier")
    carrier = np.exp(1j * TWOPI * carrier_cpm * X)
    h_phase = wrap_2pi(alpha)
    v_phase = wrap_2pi(-alpha + 0.5 * np.pi)
    eh_slm = (A / np.sqrt(2.0)) * np.exp(1j * h_phase) * carrier
    ev_slm = (A / np.sqrt(2.0)) * np.exp(1j * v_phase) * carrier

    iris = apply_fourier_iris(
        (eh_slm, ev_slm),
        grid,
        signal_fx_cpm=carrier_cpm,
        iris_radius_frac=float(iris_radius_frac),
        wavelength_m=float(cfg.wavelength_m),
        tilt_tolerance_rad=float(tilt_tolerance_rad),
    )
    eh_f, ev_f = iris.signal
    Ex, Ey = apply_uniform_jones(qwp(-0.25 * np.pi), eh_f, ev_f)

    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    iris_radius_cpm = float(iris_radius_frac) * carrier_cpm
    dc_disk = (FX**2 + FY**2) <= iris_radius_cpm**2
    spec_h = fft2c(eh_slm)
    spec_v = fft2c(ev_slm)
    spec_power = np.abs(spec_h) ** 2 + np.abs(spec_v) ** 2
    total_spec_power = float(np.sum(spec_power))
    zero_order_content = float(np.sum(spec_power[dc_disk]) / max(total_spec_power, EPS))
    passed_mask = np.asarray(iris.mask, dtype=bool)
    zero_order_leakage = float(np.sum(spec_power[dc_disk & passed_mask]) / max(total_spec_power, EPS))

    ledger = iris.ledger
    slm_4f_report = {
        "carrier_lpmm": float(carrier_lpmm),
        "iris_radius_lpmm": float(iris_radius_frac) * float(carrier_lpmm),
        "iris_radius_frac_of_carrier": float(iris_radius_frac),
        "incident_power": float(ledger.incident_power),
        "signal_power": float(ledger.signal_power),
        "blocked_power": float(ledger.blocked_power),
        "power_ledger_relative_error": float(ledger.relative_error),
        "first_order_efficiency": float(ledger.signal_power / max(ledger.incident_power, EPS)),
        "blocked_power_fraction": float(ledger.blocked_power / max(ledger.incident_power, EPS)),
        "zero_order_content_before_iris": zero_order_content,
        "zero_order_leakage_after_iris": zero_order_leakage,
        "residual_tilt_rad": float(np.hypot(*iris.residual_tilt_rad)),
        "phase_only_slm": True,
        "fill_factor_quantisation_aberrations": "not modelled in this basic carrier-filtered case",
        "recombination": "ideal lossless H/V recombination",
    }
    return _mode2n_route_result(
        "route_2nc_dual_slm_4f",
        Ex, Ey, data, v0,
        ideal=False,
        slm_4f_report=slm_4f_report,
        extra_metadata={
            "route_family": "dual_slm_carrier_4f_filtered_then_uniform_qwp",
            "h_channel_phase": "wrap(+alpha + carrier)",
            "v_channel_phase": "wrap(-alpha + pi/2 + carrier)",
            "qwp_angle_rad": float(-0.25 * np.pi),
        },
    )


def mode2n_route_metric_row(result: Mode2NRouteResult) -> dict[str, Any]:
    """Flatten one MODE 2N route into a CSV/JSON-safe row."""

    pre = dict(result.pre_axicon_metrics)
    cmp_ = dict(result.v0_comparison)
    f4 = dict(result.slm_4f_report or {})
    sym = dict(result.symmetry)
    return {
        "route_id": str(result.route_id),
        "ideal_route": bool(result.metadata.get("ideal_route", False)),
        "pre_axicon_overlap": float(pre.get("complex_vector_overlap", np.nan)),
        "pre_axicon_phase_aligned_rms": float(pre.get("phase_aligned_rms", np.nan)),
        "pre_axicon_stokes_rms": float(pre.get("stokes_rms", np.nan)),
        "pre_axicon_alpha_rms_mod_pi": float(pre.get("alpha_angle_rms_mod_pi", np.nan)),
        "pre_axicon_power_ratio": float(pre.get("power_ratio", np.nan)),
        "first_order_efficiency": float(f4.get("first_order_efficiency", np.nan)),
        "zero_order_content_before_iris": float(f4.get("zero_order_content_before_iris", np.nan)),
        "zero_order_leakage_after_iris": float(f4.get("zero_order_leakage_after_iris", np.nan)),
        "power_ledger_relative_error": float(f4.get("power_ledger_relative_error", np.nan)),
        "reference_z_mm": float(result.reference_z_m / 1e-3),
        "z60_full_field_correlation": float(cmp_.get("z60_full_field_correlation", np.nan)),
        "z60_central_crop_correlation": float(cmp_.get("z60_central_crop_correlation", np.nan)),
        "angular_profile_correlation_to_v0": float(cmp_.get("angular_profile_correlation_to_v0", np.nan)),
        "xz_map_correlation_to_v0": float(cmp_.get("xz_map_correlation_to_v0", np.nan)),
        "on_axis_intensity_correlation_to_v0": float(cmp_.get("on_axis_intensity_correlation_to_v0", np.nan)),
        "ring_radius_mm": float(result.ring_radius_m / 1e-3),
        "dark_core_ratio": float(result.dark_core_ratio),
        "c120_minus_c60": float(sym.get("c120_minus_c60", np.nan)),
        "order3_over_order6": float(sym.get("order3_over_order6", np.nan)),
        "symmetry_class": str(result.symmetry_class),
        "passes_v0_match": bool(result.passes_v0_match),
        "fail_reasons": "; ".join(result.fail_reasons),
    }


def mode2n_outcome_report(
    *,
    v0: Mode2NRouteResult,
    patterned_hwp: Mode2NRouteResult,
    dual_slm_qwp: Mode2NRouteResult,
    dual_slm_4f: Mode2NRouteResult | None,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Choose exactly one M2N-A/B/C/D outcome from source-scale evidence only."""

    cfg: NathanSourceParityConfig = data["config"]
    ideal_routes = (patterned_hwp, dual_slm_qwp)
    ideal_pre_ok = all(
        float(r.pre_axicon_metrics["complex_vector_overlap"]) >= MODE2N_PRE_AXICON_OVERLAP_PASS for r in ideal_routes
    )
    ideal_prop_ok = all(r.passes_v0_match for r in ideal_routes)
    realistic_ok = dual_slm_4f is not None and dual_slm_4f.passes_v0_match
    semantics_ok = bool(
        v0.symmetry_class == "visual_hexagonal_field"
        and all(np.isfinite(float(r.v0_comparison["z60_full_field_correlation"])) for r in ideal_routes)
        and (dual_slm_4f is None or float(dual_slm_4f.slm_4f_report["power_ledger_relative_error"]) < 1.0e-9)
    )
    if not semantics_ok:
        outcome = "M2N-D"
        statement = (
            "The model cannot yet simulate a legitimate source-scale component bench: the V0 reference or "
            "the grid/FFT/4F power semantics are inconsistent. Fix the bench model before interpreting routes."
        )
    elif ideal_prop_ok and realistic_ok:
        outcome = "M2N-A"
        statement = (
            "Ideal patterned-HWP and ideal dual-SLM/QWP routes reproduce V0 after axicon propagation, and the "
            "carrier/4F dual-SLM route also reproduces it with acceptable fidelity. Experimental source-scale "
            "replication of Nathan's Fig. 4 beam is plausible."
        )
    elif ideal_prop_ok:
        outcome = "M2N-B"
        statement = (
            "Ideal component routes reproduce V0 after axicon propagation, but carrier/4F SLM realism degrades "
            "the beam below the acceptance threshold. SLM/4F optimisation is needed before a lab attempt."
        )
    elif ideal_pre_ok:
        outcome = "M2N-C"
        statement = (
            "Ideal component routes reproduce the pre-axicon target but do not reproduce V0 after axicon "
            "propagation: the M2P target convention and the V0 propagation convention are mismatched."
        )
    else:
        outcome = "M2N-D"
        statement = (
            "The generation routes do not even reproduce the pre-axicon target on the source grid; the bench "
            "model semantics are inconsistent."
        )

    def _route_summary(r: Mode2NRouteResult | None) -> dict[str, Any] | None:
        if r is None:
            return None
        return {
            "route_id": r.route_id,
            "pre_axicon_overlap": float(r.pre_axicon_metrics["complex_vector_overlap"]),
            "z60_full_field_correlation": float(r.v0_comparison["z60_full_field_correlation"]),
            "symmetry_class": r.symmetry_class,
            "passes_v0_match": bool(r.passes_v0_match),
            "fail_reasons": list(r.fail_reasons),
            "first_order_efficiency": None if r.slm_4f_report is None else float(r.slm_4f_report["first_order_efficiency"]),
        }

    return {
        "stage": MODE2N_STAGE,
        "suggested_outcome": outcome,
        "allowed_outcomes": MODE2N_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "scope": MODE2N_SCOPE_STATEMENT,
        "observable": "total intensity |Ex|^2 + |Ey|^2 + |Ez|^2 on Nathan's axis-sampled source grid",
        "source_scale_parameters": {
            "wavelength_m": float(cfg.wavelength_m),
            "beam_radius_m": float(cfg.beam_radius_m),
            "axicon_n": float(cfg.axicon_n),
            "medium_n": float(cfg.medium_n),
            "axicon_apex_angle_deg": float(cfg.axicon_apex_angle_deg),
            "axicon_base_angle_deg": float(np.rad2deg(cfg.axicon_base_angle_rad)),
            "window_m": float(cfg.window_m),
            "grid_n": int(cfg.grid_n),
            "z_reference_m": float(cfg.z_reference_m),
        },
        "pre_axicon_overlap_pass": MODE2N_PRE_AXICON_OVERLAP_PASS,
        "ideal_propagated_correlation_pass": MODE2N_IDEAL_PROPAGATED_CORRELATION_PASS,
        "realistic_propagated_correlation_pass": MODE2N_REALISTIC_PROPAGATED_CORRELATION_PASS,
        "v0_reference": _route_summary(v0),
        "route_patterned_hwp": _route_summary(patterned_hwp),
        "route_dual_slm_qwp": _route_summary(dual_slm_qwp),
        "route_dual_slm_4f": _route_summary(dual_slm_4f),
        "inherited_objective_sample_geometry_used": False,
        "microfabrication_sample_plane_claim": False,
        "micro_scale_note": (
            "MODE 2N is a source-scale replication result only; it makes no statement about the inherited "
            "objective/sample microfabrication architecture (see MODE 1C/M1E for that separate question)."
        ),
    }


def mode2n_scope_manifest(outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Machine-readable MODE 2N simulation-scope manifest."""

    return {
        "stage": MODE2N_STAGE,
        "scope": MODE2N_SCOPE_STATEMENT,
        "git_commit": _mode1_git_commit_short(),
        "source_scale_bench_replication": True,
        "grid_convention": "Nathan/V0 axis-sampled source grid (samples x = 0 exactly)",
        "inherited_objective_sample_geometry": False,
        "micro_scale_sample_plane_simulated": False,
        "microfabrication_sample_plane_claim": False,
        "hwp_qwp_slm_generation_modelled": True,
        "carrier_4f_first_order_filtering_modelled": True,
        "pixel_fill_factor_quantisation_aberrations": False,
        "physical_axicon_modelled": "thin Nathan source-scale axicon with p/s Fresnel split",
        "propagation": "free-space vector angular-spectrum z-stack, reference plane z = 60 mm",
        "suggested_outcome": None if outcome is None else outcome.get("suggested_outcome"),
        "success_criterion": "propagated z = 60 mm intensity must match V0/Fig. 4, never pre-axicon overlap alone",
        "claim_boundary": {
            "model_status": MODEL_STATUS,
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "material_model": False,
            "camera_model": False,
            "judged_by_mode1c_m1e_microfabrication_constraints": False,
        },
    }


def run_mode2n_source_replication(
    config: NathanSourceParityConfig | None = None,
    *,
    grid_n: int = 512,
    z_planes: int = 61,
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM,
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    include_4f_route: bool = True,
) -> dict[str, Any]:
    """Run the full MODE 2N study: target, V0 reference, three routes, outcome."""

    data = mode2n_source_target(config, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    patterned = run_mode2n_patterned_hwp_route(data, v0)
    dual_qwp = run_mode2n_dual_slm_qwp_route(data, v0)
    dual_4f = (
        run_mode2n_dual_slm_4f_route(data, v0, carrier_lpmm=float(carrier_lpmm), iris_radius_frac=float(iris_radius_frac))
        if include_4f_route
        else None
    )
    routes = [v0, patterned, dual_qwp] + ([dual_4f] if dual_4f is not None else [])
    outcome = mode2n_outcome_report(
        v0=v0, patterned_hwp=patterned, dual_slm_qwp=dual_qwp, dual_slm_4f=dual_4f, data=data,
    )
    return {
        "data": data,
        "v0": v0,
        "patterned_hwp": patterned,
        "dual_slm_qwp": dual_qwp,
        "dual_slm_4f": dual_4f,
        "route_rows": tuple(mode2n_route_metric_row(r) for r in routes),
        "outcome": outcome,
        "manifest": mode2n_scope_manifest(outcome),
    }


# ---------------------------------------------------------------------------
# MODE 2N figures (package functions; the notebook calls these, no duplicated physics)
# ---------------------------------------------------------------------------


def _mode2n_mm_axis(grid: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(grid["x"], dtype=float) / 1e-3


def plot_mode2n_pre_axicon(
    Ex: Any,
    Ey: Any,
    data: Mapping[str, Any],
    *,
    title: str,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = data["grid"]
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    ex = np.asarray(Ex, dtype=np.complex128)
    ey = np.asarray(Ey, dtype=np.complex128)
    intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2
    st = stokes_from_linear_components(ex, ey)
    pol_angle = 0.5 * np.arctan2(np.asarray(st["S2"], dtype=float), np.asarray(st["S1"], dtype=float))
    angle_err = 0.5 * np.angle(np.exp(2j * (pol_angle - np.asarray(data["alpha"], dtype=float))))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    axes[0].imshow(_normalise_image(intensity, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("total transverse intensity (Gaussian expected)")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    im1 = axes[1].imshow(pol_angle, origin="lower", extent=ext, cmap="twilight", vmin=-0.5 * np.pi, vmax=0.5 * np.pi)
    axes[1].set_title("local polarisation angle (mod pi)")
    axes[1].set_xlabel("x (mm)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)
    im2 = axes[2].imshow(np.abs(angle_err), origin="lower", extent=ext, cmap="magma", vmin=0.0, vmax=0.2)
    axes[2].set_title("|angle error| to segmented alpha (rad)")
    axes[2].set_xlabel("x (mm)")
    fig.colorbar(im2, ax=axes[2], shrink=0.8)
    fig.suptitle(title)
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2n_z60(
    result: Mode2NRouteResult,
    *,
    crop_fraction: float = 0.35,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    plane = np.asarray(result.reference_plane, dtype=float)
    x_mm = _mode2n_mm_axis(result.grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    crop, crop_grid = _mode1b_even_axis_crop(plane, result.grid, float(crop_fraction))
    xc_mm = np.asarray(crop_grid["x"], dtype=float) / 1e-3
    ext_c = [float(xc_mm[0]), float(xc_mm[-1]), float(xc_mm[0]), float(xc_mm[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    axes[0].imshow(_normalise_image(plane, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"xy at z = {result.reference_z_m / 1e-3:.1f} mm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[1].imshow(_normalise_image(crop, local=True), origin="lower", extent=ext_c, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("central crop")
    axes[1].set_xlabel("x (mm)")
    corr = float(result.v0_comparison.get("z60_full_field_correlation", np.nan))
    fig.suptitle(
        f"MODE 2N {result.route_id}: {result.symmetry_class}, "
        f"V0 correlation {corr:.4f}, passes={result.passes_v0_match}"
    )
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2n_xz_comparison(
    results: Sequence[Mode2NRouteResult],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(9.6, 2.6 * n), constrained_layout=True, squeeze=False)
    for idx, result in enumerate(results):
        ax = axes[idx, 0]
        x_mm = _mode2n_mm_axis(result.grid)
        z_mm = np.asarray(result.z_values_m, dtype=float) / 1e-3
        ext = [float(x_mm[0]), float(x_mm[-1]), float(z_mm[0]), float(z_mm[-1])]
        ax.imshow(
            _normalise_image(np.asarray(result.xz_map, dtype=float), local=True),
            origin="lower", aspect="auto", extent=ext, cmap="inferno", vmin=0, vmax=1,
        )
        ax.axhline(result.reference_z_m / 1e-3, color="white", lw=0.8, alpha=0.8)
        corr = float(result.v0_comparison.get("xz_map_correlation_to_v0", np.nan))
        ax.set_title(f"{result.route_id} (x-z, corr to V0 {corr:.4f})", fontsize=9)
        ax.set_ylabel("z (mm)")
    axes[-1, 0].set_xlabel("x (mm)")
    fig.suptitle("MODE 2N x-z propagation comparison")
    _save_fig(fig, output_path)
    return fig, axes


def write_mode2n_outputs(
    config: NathanSourceParityConfig | None = None,
    *,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode2n_source_replication",
    report: Mapping[str, Any] | None = None,
    **run_kwargs: Any,
) -> dict[str, Path]:
    """Run (or reuse) the MODE 2N study and write all required artefacts."""

    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = dict(report) if report is not None else run_mode2n_source_replication(config, **run_kwargs)
    data = study["data"]
    v0: Mode2NRouteResult = study["v0"]
    routes: dict[str, Mode2NRouteResult | None] = {
        "patterned_hwp": study["patterned_hwp"],
        "dual_slm_qwp": study["dual_slm_qwp"],
        "dual_slm_4f": study["dual_slm_4f"],
    }
    paths: dict[str, Path] = {
        "route_metrics_csv": out / "mode2n_route_metrics.csv",
        "route_metrics_json": out / "mode2n_route_metrics.json",
        "outcome_report": out / "mode2n_outcome_report.json",
        "scope_manifest": out / "simulation_scope_manifest.json",
    }
    _write_rows(paths["route_metrics_csv"], study["route_rows"])
    paths["route_metrics_json"].write_text(json.dumps(_json_ready(study["route_rows"]), indent=2), encoding="utf-8")
    paths["outcome_report"].write_text(json.dumps(_json_ready(dict(study["outcome"])), indent=2), encoding="utf-8")
    paths["scope_manifest"].write_text(json.dumps(_json_ready(dict(study["manifest"])), indent=2), encoding="utf-8")

    fig, _ = plot_mode2n_pre_axicon(
        data["target"][0], data["target"][1], data,
        title="MODE 2N target pre-axicon field (V0 segmented RA convention)",
        output_path=out / "mode2n_target_pre_axicon.png",
    )
    plt.close(fig)
    paths["target_pre_axicon"] = out / "mode2n_target_pre_axicon.png"

    pre_names = {
        "patterned_hwp": "mode2n_route_patterned_hwp_pre_axicon.png",
        "dual_slm_qwp": "mode2n_route_dual_slm_qwp_pre_axicon.png",
        "dual_slm_4f": "mode2n_route_dual_slm_4f_pre_axicon.png",
    }
    z60_names = {
        "patterned_hwp": "mode2n_patterned_hwp_z60.png",
        "dual_slm_qwp": "mode2n_dual_slm_qwp_z60.png",
        "dual_slm_4f": "mode2n_dual_slm_4f_z60.png",
    }
    for key, route in routes.items():
        if route is None:
            continue
        overlap = float(route.pre_axicon_metrics["complex_vector_overlap"])
        fig, _ = plot_mode2n_pre_axicon(
            route.pre_axicon_field[0], route.pre_axicon_field[1], data,
            title=f"MODE 2N {route.route_id} pre-axicon (overlap {overlap:.6f})",
            output_path=out / pre_names[key],
        )
        plt.close(fig)
        paths[f"route_{key}_pre_axicon"] = out / pre_names[key]
        fig, _ = plot_mode2n_z60(route, output_path=out / z60_names[key])
        plt.close(fig)
        paths[f"{key}_z60"] = out / z60_names[key]

    fig, _ = plot_mode2n_z60(v0, output_path=out / "mode2n_v0_reference_z60.png")
    plt.close(fig)
    paths["v0_reference_z60"] = out / "mode2n_v0_reference_z60.png"

    xz_routes = [v0] + [r for r in routes.values() if r is not None]
    fig, _ = plot_mode2n_xz_comparison(xz_routes, output_path=out / "mode2n_xz_comparison.png")
    plt.close(fig)
    paths["xz_comparison"] = out / "mode2n_xz_comparison.png"
    return paths


# ===========================================================================
# MODE 2Q - Nathan source-scale backward / adjoint mask synthesis
# ===========================================================================
#
# MODE 2N replicated Nathan's Fig. 4 beam by forward-displaying the known masks.
# MODE 2Q is the inverse-design correction: the propagated beam is the target,
# the masks are whatever the inverse optical system says they must be.
#   V0 complex target at z_ref -> inverse propagation -> inverse axicon ->
#   inverse QWP -> required H/V complex fields -> phase-only/constrained masks
#   -> full forward verification.
# The 4F iris is handled as an adjoint projection (it discards frequencies and
# has no exact inverse).  No inherited objective/sample geometry, no MODE
# 1C/M1E sample-plane constraints, no panel realism.  Six bright lobes are not
# called a hexagon unless the strict classifier gate passes.

MODE2Q_STAGE = "nathan_mode2q_backward_mask_synthesis"
MODE2Q_ALLOWED_OUTCOMES = ("M2Q-A", "M2Q-B", "M2Q-C", "M2Q-D", "M2Q-E")
MODE2Q_BACKWARD_RECOVERY_OVERLAP_PASS = 0.999
MODE2Q_FORWARD_DIRECT_CORRELATION_PASS = 0.999
MODE2Q_FORWARD_4F_CORRELATION_PASS = MODE2N_REALISTIC_PROPAGATED_CORRELATION_PASS
MODE2Q_C120_MINUS_C60_TOL = 0.04
MODE2Q_SCOPE_STATEMENT = (
    "MODE 2Q source-scale backward/adjoint mask synthesis only: the validated V0 complex vector "
    "field at z_ref = 60 mm is the target; inverse angular-spectrum propagation, inverse thin "
    "axicon Jones, and inverse (adjoint) QWP algebra determine the required H/V fields and masks, "
    "which are then verified by full forward propagation. The 4F iris is treated as an adjoint "
    "projection, never a fake exact inverse. No inherited objective/sample microfabrication "
    "geometry, no MODE 1C/M1E sample-plane constraints, no panel realism, and no hexagon claim "
    "unless the strict classifier gate passes."
)


def mode2q_forward_propagate_vector(field: VectorField, z_m: float) -> VectorField:
    """Forward vector angular-spectrum propagation to one plane (V0 conventions)."""

    _, planes = _free_space_intensity_stack(field, [float(z_m)], return_fields=True)
    return planes[0]


def mode2q_backpropagate_vector(field: VectorField, z_m: float) -> tuple[VectorField, dict[str, Any]]:
    """Inverse angular-spectrum propagation over ``z_m`` using propagating modes only.

    Backward transfer is ``exp(-i kz z)``; evanescent modes are clipped (a
    backward ``exp(+|kz| z)`` amplification would be unphysical noise gain) and
    the clipped spectral energy fraction is reported.
    """

    grid = field.grid
    z = float(z_m)
    k = TWOPI * float(field.medium_index) / float(field.wavelength_m)
    kx = TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = TWOPI * np.asarray(grid["FY"], dtype=float)
    kz_sq = k * k - kx * kx - ky * ky
    propagating = kz_sq > 0.0
    kz = np.sqrt(np.maximum(kz_sq, 0.0))
    ax = fft2c(field.ex)
    ay = fft2c(field.ey)
    az = fft2c(field.ez)
    total = float(np.sum(np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2))
    clipped = float(np.sum((np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2)[~propagating]))
    transfer = np.where(propagating, np.exp(-1j * kz * z), 0.0)
    out = VectorField(
        ex=ifft2c(ax * transfer),
        ey=ifft2c(ay * transfer),
        ez=ifft2c(az * transfer),
        grid=grid,
        wavelength_m=field.wavelength_m,
        medium_index=field.medium_index,
        metadata={**dict(field.metadata), "stage": "mode2q_backpropagated", "backpropagated_z_m": z},
    )
    return out, {
        "backpropagated_z_m": z,
        "evanescent_clipped_energy_fraction": float(clipped / max(total, EPS)),
        "propagating_mode_fraction_of_grid": float(np.mean(propagating)),
        "transfer": "exp(-i kz z), propagating modes only",
    }


def mode2q_inverse_axicon(
    field: VectorField,
    config: NathanSourceParityConfig,
) -> tuple[VectorField, dict[str, Any]]:
    """Invert the thin source-scale vector axicon (conical phase + p/s Fresnel).

    The forward element is diagonal in the radial/azimuthal basis
    (``t_entry t_p`` and ``t_entry t_s`` times the conical phase), so the local
    Jones inverse is the reciprocal diagonal.  The condition number and any
    near-singular transmission are reported instead of silently inverted.
    """

    grid = field.grid
    R = np.asarray(grid["R"], dtype=float)
    phi = np.asarray(grid["PHI"], dtype=float)
    t_entry, t_p, t_s = fresnel_sp_amplitudes(float(config.axicon_n), float(config.medium_n), float(config.axicon_base_angle_rad))
    k_r = float(TWOPI / field.wavelength_m * (float(config.axicon_n) - float(config.medium_n)) * np.tan(float(config.axicon_base_angle_rad)))
    eigen = (abs(t_entry * t_p), abs(t_entry * t_s))
    condition = float(max(eigen) / max(min(eigen), EPS))
    singular = bool(min(eigen) < 1.0e-6)
    if singular:
        raise ValueError("inverse axicon is singular (near-zero p/s transmission); refusing to invert silently")
    er_x = np.cos(phi)
    er_y = np.sin(phi)
    ep_x = -np.sin(phi)
    ep_y = np.cos(phi)
    er = field.ex * er_x + field.ey * er_y
    ephi = field.ex * ep_x + field.ey * ep_y
    inv_phase = np.exp(+1j * abs(k_r) * R)
    er_in = er * inv_phase / (t_entry * t_p)
    ephi_in = ephi * inv_phase / (t_entry * t_s)
    ez_scale = t_entry * 0.5 * (t_p + t_s)
    out = VectorField(
        ex=er_in * er_x + ephi_in * ep_x,
        ey=er_in * er_y + ephi_in * ep_y,
        ez=field.ez * inv_phase / ez_scale,
        grid=grid,
        wavelength_m=field.wavelength_m,
        medium_index=1.0,
        metadata={**dict(field.metadata), "stage": "mode2q_inverse_axicon"},
    )
    return out, {
        "k_r_m_inv": k_r,
        "t_entry_abs": float(abs(t_entry)),
        "t_p_abs": float(abs(t_p)),
        "t_s_abs": float(abs(t_s)),
        "jones_condition_number": condition,
        "near_singular_transmission": singular,
        "inverse": "reciprocal diagonal Jones in radial/azimuthal basis, conjugate conical phase",
    }


def mode2q_inverse_retarder(J: Any) -> np.ndarray:
    """Inverse of an ideal (unitary) retarder Jones matrix: ``J^-1 = J^dagger``."""

    arr = np.asarray(J, dtype=np.complex128)
    if arr.shape != (2, 2):
        raise ValueError("mode2q_inverse_retarder expects a uniform 2x2 Jones matrix")
    inv = arr.conj().T
    residual = float(np.max(np.abs(inv @ arr - np.eye(2))))
    if residual > 1.0e-10:
        raise ValueError(f"retarder is not unitary (|J^dagger J - I| = {residual:.3e}); adjoint is not its inverse")
    return inv


def mode2q_4f_passband_report(
    Ex: Any,
    Ey: Any,
    grid: Mapping[str, Any],
    *,
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM,
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
) -> dict[str, Any]:
    """Adjoint/backprojection bookkeeping for the 4F iris on a required baseband field.

    The 4F first-order iris discards spatial frequencies, so it has no exact
    inverse; the correct backward map is the adjoint projection (the same
    passband).  This reports how much of the required field's spectral energy
    lies outside the baseband-equivalent passband, i.e. the part no mask behind
    this 4F aperture can ever supply.
    """

    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    radius_cpm = float(iris_radius_frac) * float(carrier_lpmm) * 1.0e3
    passband = (FX**2 + FY**2) <= radius_cpm**2
    spec = np.abs(fft2c(np.asarray(Ex, dtype=np.complex128))) ** 2 + np.abs(fft2c(np.asarray(Ey, dtype=np.complex128))) ** 2
    total = float(np.sum(spec))
    outside = float(np.sum(spec[~passband]))
    return {
        "kind": "adjoint_projection_not_exact_inverse",
        "carrier_lpmm": float(carrier_lpmm),
        "iris_radius_lpmm": float(iris_radius_frac) * float(carrier_lpmm),
        "required_field_energy_outside_passband_fraction": float(outside / max(total, EPS)),
        "exactly_realizable_through_this_4f": bool(outside / max(total, EPS) < 1.0e-6),
    }


@dataclass(frozen=True)
class Mode2QBackwardField:
    """Analytic backward initialisation: required fields and initial masks."""

    Ex_required_pre_axicon: np.ndarray
    Ey_required_pre_axicon: np.ndarray
    Ex_required_pre_qwp: np.ndarray
    Ey_required_pre_qwp: np.ndarray
    phi_H_initial: np.ndarray
    phi_V_initial: np.ndarray
    amp_H_required: np.ndarray
    amp_V_required: np.ndarray
    diagnostics: Mapping[str, Any]


def mode2q_v0_complex_target(data: Mapping[str, Any]) -> tuple[VectorField, VectorField, dict[str, Any]]:
    """Return (field after axicon at z=0, complex V0 vector field at z_ref).

    The V0 propagation machinery already retains complex vector fields
    (``return_fields=True``), so the full complex target is available and no
    intensity-only phase-retrieval fallback is needed.
    """

    cfg: NathanSourceParityConfig = data["config"]
    after, axicon_meta = _apply_free_space_vector_axicon(
        data["target_field"],
        n_axicon=float(cfg.axicon_n),
        n_medium=float(cfg.medium_n),
        base_angle_rad=float(cfg.axicon_base_angle_rad),
    )
    target_z = mode2q_forward_propagate_vector(after, float(cfg.z_reference_m))
    return after, target_z, {
        "complex_vector_target_available": True,
        "z_reference_m": float(cfg.z_reference_m),
        "axicon_meta": dict(axicon_meta),
    }


def run_mode2q_backward_initialisation(
    data: Mapping[str, Any],
    *,
    qwp_angle_rad: float = -0.25 * np.pi,
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM,
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
) -> Mode2QBackwardField:
    """Backward pass: V0 complex target -> required pre-axicon field -> H/V masks."""

    cfg: NathanSourceParityConfig = data["config"]
    _, target_z, target_meta = mode2q_v0_complex_target(data)
    back_to_axicon, back_meta = mode2q_backpropagate_vector(target_z, float(cfg.z_reference_m))
    required_pre_axicon, inv_axicon_meta = mode2q_inverse_axicon(back_to_axicon, cfg)

    raw = data["target"]
    recovered = (required_pre_axicon.ex, required_pre_axicon.ey)
    mask = data["metric_mask"]
    recovery = {
        "overlap_to_raw_nathan_input": complex_vector_overlap(recovered, raw, mask),
        "phase_aligned_rms_to_raw": phase_aligned_rms(recovered, raw, mask),
        "stokes_rms_to_raw": jones_stokes_rms(recovered, raw, mask),
        "alpha_angle_rms_mod_pi_to_raw": alpha_angle_rms_mod_pi(recovered, raw, mask),
        "power_ratio_to_raw": jones_power_ratio(recovered, raw, mask),
        "ez_power_fraction_pre_axicon": float(
            np.sum(np.abs(required_pre_axicon.ez) ** 2)
            / max(float(np.sum(np.abs(required_pre_axicon.ex) ** 2 + np.abs(required_pre_axicon.ey) ** 2 + np.abs(required_pre_axicon.ez) ** 2)), EPS)
        ),
    }

    inv_qwp = mode2q_inverse_retarder(qwp(float(qwp_angle_rad)))
    EH_required, EV_required = apply_uniform_jones(inv_qwp, required_pre_axicon.ex, required_pre_axicon.ey)
    phi_H = np.angle(np.asarray(EH_required, dtype=np.complex128))
    phi_V = np.angle(np.asarray(EV_required, dtype=np.complex128))
    amp_H = np.abs(np.asarray(EH_required, dtype=np.complex128))
    amp_V = np.abs(np.asarray(EV_required, dtype=np.complex128))

    supply = np.asarray(data["A"], dtype=float) / np.sqrt(2.0)
    supply_power = float(np.sum(supply**2))
    amp_mismatch = {
        "phase_only_supply": "A(r)/sqrt(2) per channel from the input Gaussian",
        "amp_H_over_supply_rms": float(np.sqrt(np.sum((amp_H - supply) ** 2) / max(supply_power, EPS))),
        "amp_V_over_supply_rms": float(np.sqrt(np.sum((amp_V - supply) ** 2) / max(supply_power, EPS))),
        "amp_H_max": float(np.max(amp_H)),
        "amp_V_max": float(np.max(amp_V)),
        "supply_max": float(np.max(supply)),
    }
    passband = mode2q_4f_passband_report(
        EH_required, EV_required, data["grid"],
        carrier_lpmm=float(carrier_lpmm), iris_radius_frac=float(iris_radius_frac),
    )
    return Mode2QBackwardField(
        Ex_required_pre_axicon=np.asarray(required_pre_axicon.ex, dtype=np.complex128),
        Ey_required_pre_axicon=np.asarray(required_pre_axicon.ey, dtype=np.complex128),
        Ex_required_pre_qwp=np.asarray(EH_required, dtype=np.complex128),
        Ey_required_pre_qwp=np.asarray(EV_required, dtype=np.complex128),
        phi_H_initial=phi_H,
        phi_V_initial=phi_V,
        amp_H_required=amp_H,
        amp_V_required=amp_V,
        diagnostics={
            "target": dict(target_meta),
            "backpropagation": dict(back_meta),
            "inverse_axicon": dict(inv_axicon_meta),
            "recovery_vs_raw_nathan_input": recovery,
            "inverse_qwp_angle_rad": float(qwp_angle_rad),
            "amplitude_vs_phase_only_supply": amp_mismatch,
            "four_f_adjoint": passband,
            "backward_recovery_pass": bool(
                float(recovery["overlap_to_raw_nathan_input"]) >= MODE2Q_BACKWARD_RECOVERY_OVERLAP_PASS
            ),
        },
    )


def mode2q_strict_hexagon_gate(plane: np.ndarray, grid: Mapping[str, Any]) -> dict[str, Any]:
    """Strict visual-hexagon gate: six bright lobes alone are NOT a hexagon.

    Wraps the calibrated C3-vs-C6 classifier and renames the failure classes so
    a six-island non-hexagon is reported as ``six_lobed_structured`` rather
    than being visually mistaken for a hexagon.
    """

    metrics = _mode2n_reference_plane_metrics(np.asarray(plane, dtype=float), grid)
    sym = dict(metrics["symmetry"])
    cls = str(metrics["symmetry_class"])
    c_delta = float(sym.get("c120_minus_c60", np.nan))
    islands = int(sym.get("ring_island_count", -1))
    if cls == "visual_hexagonal_field":
        strict = "visual_hexagonal_field"
    elif cls == "triangular_lobed_field":
        strict = "triangular_lobed_field"
    elif islands == 6:
        strict = "six_lobed_structured"
    else:
        strict = "dark_core_structured_field"
    return {
        **metrics,
        "strict_class": strict,
        "c120_minus_c60": c_delta,
        "ring_island_count": islands,
        "c120_c60_within_tolerance": bool(np.isfinite(c_delta) and c_delta <= MODE2Q_C120_MINUS_C60_TOL),
        "passes_true_hexagon_gate": bool(strict == "visual_hexagonal_field"),
        "note": "order-6 lobe count alone never passes; the C3-vs-C6 rotational discriminator is decisive",
    }


def run_mode2q_forward_candidate(
    candidate_id: str,
    phi_H: np.ndarray,
    phi_V: np.ndarray,
    data: Mapping[str, Any],
    v0: Mode2NRouteResult,
    target_z: VectorField,
    backward: Mode2QBackwardField,
    *,
    use_4f: bool,
    qwp_angle_rad: float = -0.25 * np.pi,
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM,
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
) -> dict[str, Any]:
    """Run the full source-scale forward operator for one phase-mask candidate."""

    cfg: NathanSourceParityConfig = data["config"]
    grid = data["grid"]
    A = np.asarray(data["A"], dtype=float)
    supply = A / np.sqrt(2.0)
    eh = supply * np.exp(1j * np.asarray(phi_H, dtype=float))
    ev = supply * np.exp(1j * np.asarray(phi_V, dtype=float))
    slm_4f_report: dict[str, Any] | None = None
    if use_4f:
        carrier_cpm = float(carrier_lpmm) * 1.0e3
        carrier = np.exp(1j * TWOPI * carrier_cpm * np.asarray(grid["X"], dtype=float))
        iris = apply_fourier_iris(
            (eh * carrier, ev * carrier),
            grid,
            signal_fx_cpm=carrier_cpm,
            iris_radius_frac=float(iris_radius_frac),
            wavelength_m=float(cfg.wavelength_m),
            tilt_tolerance_rad=1.0e-4,
        )
        eh, ev = iris.signal
        slm_4f_report = {
            "first_order_efficiency": float(iris.ledger.signal_power / max(iris.ledger.incident_power, EPS)),
            "power_ledger_relative_error": float(iris.ledger.relative_error),
            "carrier_lpmm": float(carrier_lpmm),
            "iris_radius_lpmm": float(iris_radius_frac) * float(carrier_lpmm),
            "kind": "forward 4F; backward direction is adjoint projection only",
        }
    Ex, Ey = apply_uniform_jones(qwp(float(qwp_angle_rad)), eh, ev)

    pre_required = jones_metric_row(
        candidate_id, (Ex, Ey), (backward.Ex_required_pre_axicon, backward.Ey_required_pre_axicon), mask=data["metric_mask"],
    )
    pre_raw = jones_metric_row(candidate_id, (Ex, Ey), data["target"], mask=data["metric_mask"])

    prop = mode2n_propagate_through_source_axicon(Ex, Ey, data)
    stack = prop["intensity_stack"]
    ref = int(prop["reference_index"])
    z_values = prop["z_values_m"]
    comparison = mode2n_compare_stacks_to_v0(stack, z_values, ref, grid, v0)
    strict = mode2q_strict_hexagon_gate(np.asarray(stack[ref], dtype=float), grid)

    field = _mode2n_vector_field(Ex, Ey, data)
    after, _ = _apply_free_space_vector_axicon(
        field, n_axicon=float(cfg.axicon_n), n_medium=float(cfg.medium_n), base_angle_rad=float(cfg.axicon_base_angle_rad),
    )
    cand_z = mode2q_forward_propagate_vector(after, float(cfg.z_reference_m))
    num = np.vdot(cand_z.ex, target_z.ex) + np.vdot(cand_z.ey, target_z.ey) + np.vdot(cand_z.ez, target_z.ez)
    den = float(
        np.sum(np.abs(cand_z.ex) ** 2 + np.abs(cand_z.ey) ** 2 + np.abs(cand_z.ez) ** 2)
        * np.sum(np.abs(target_z.ex) ** 2 + np.abs(target_z.ey) ** 2 + np.abs(target_z.ez) ** 2)
    )
    complex_overlap_z = float(min(1.0, abs(num) ** 2 / max(den, EPS)))

    threshold = MODE2Q_FORWARD_4F_CORRELATION_PASS if use_4f else MODE2Q_FORWARD_DIRECT_CORRELATION_PASS
    reasons: list[str] = []
    if float(comparison["z60_full_field_correlation"]) < float(threshold):
        reasons.append(f"z=60 mm equal-power correlation to V0 below {threshold:g}")
    if not bool(strict["passes_true_hexagon_gate"]):
        reasons.append(f"strict class is {strict['strict_class']}, not visual_hexagonal_field")
    mid = stack.shape[1] // 2
    per_plane_hex = [
        bool(mode2q_strict_hexagon_gate(np.asarray(stack[i], dtype=float), grid)["passes_true_hexagon_gate"])
        for i in range(0, stack.shape[0], max(1, stack.shape[0] // 8))
    ]
    return {
        "candidate_id": str(candidate_id),
        "use_4f": bool(use_4f),
        "pre_axicon_vs_required": pre_required,
        "pre_axicon_vs_raw_nathan": pre_raw,
        "slm_4f_report": slm_4f_report,
        "z_values_m": np.asarray(z_values, dtype=float),
        "reference_index": ref,
        "reference_z_m": float(z_values[ref]),
        "reference_plane": np.asarray(stack[ref], dtype=np.float32),
        "xz_map": np.asarray(stack[:, mid, :], dtype=np.float32),
        "yz_map": np.asarray(stack[:, :, mid], dtype=np.float32),
        "comparison": comparison,
        "strict_gate": strict,
        "complex_vector_overlap_to_v0_target_z": complex_overlap_z,
        "sampled_plane_true_hexagon_fraction": float(np.mean(per_plane_hex)),
        "passes": bool(len(reasons) == 0),
        "fail_reasons": tuple(reasons),
    }


def mode2q_candidate_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one MODE 2Q forward candidate into a CSV/JSON-safe row."""

    pre_req = dict(candidate["pre_axicon_vs_required"])
    pre_raw = dict(candidate["pre_axicon_vs_raw_nathan"])
    cmp_ = dict(candidate["comparison"])
    strict = dict(candidate["strict_gate"])
    f4 = dict(candidate.get("slm_4f_report") or {})
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "use_4f": bool(candidate["use_4f"]),
        "pre_axicon_overlap_to_required": float(pre_req["complex_vector_overlap"]),
        "pre_axicon_overlap_to_raw_nathan": float(pre_raw["complex_vector_overlap"]),
        "pre_axicon_stokes_rms_to_raw": float(pre_raw["stokes_rms"]),
        "pre_axicon_alpha_rms_to_raw": float(pre_raw["alpha_angle_rms_mod_pi"]),
        "pre_axicon_power_ratio_to_raw": float(pre_raw["power_ratio"]),
        "first_order_efficiency": float(f4.get("first_order_efficiency", np.nan)),
        "z60_full_field_correlation": float(cmp_["z60_full_field_correlation"]),
        "z60_central_crop_correlation": float(cmp_["z60_central_crop_correlation"]),
        "complex_vector_overlap_to_v0_target_z": float(candidate["complex_vector_overlap_to_v0_target_z"]),
        "angular_profile_correlation_to_v0": float(cmp_["angular_profile_correlation_to_v0"]),
        "dark_core_ratio": float(strict["dark_core_ratio"]),
        "c120_minus_c60": float(strict["c120_minus_c60"]),
        "ring_island_count": int(strict["ring_island_count"]),
        "strict_class": str(strict["strict_class"]),
        "passes_true_hexagon_gate": bool(strict["passes_true_hexagon_gate"]),
        "sampled_plane_true_hexagon_fraction": float(candidate["sampled_plane_true_hexagon_fraction"]),
        "passes": bool(candidate["passes"]),
        "fail_reasons": "; ".join(candidate["fail_reasons"]),
    }


def mode2q_outcome_report(
    *,
    data: Mapping[str, Any],
    backward: Mode2QBackwardField,
    target_gate: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    complex_target_available: bool,
    optimisation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose exactly one M2Q-A/B/C/D/E outcome."""

    diag = dict(backward.diagnostics)
    recovery = dict(diag["recovery_vs_raw_nathan_input"])
    backward_ok = bool(diag["backward_recovery_pass"])
    target_is_hexagon = bool(target_gate["passes_true_hexagon_gate"])
    direct = [c for c in candidates if not c["use_4f"]]
    filtered = [c for c in candidates if c["use_4f"]]
    direct_pass = any(c["passes"] for c in direct)
    filtered_pass = any(c["passes"] for c in filtered)
    any_pass = direct_pass or filtered_pass

    if not complex_target_available:
        outcome = "M2Q-D"
        statement = (
            "No full complex V0 vector target is available and an intensity-only target is non-unique; "
            "complex V0 propagation output or phase retrieval is required before mask design."
        )
    elif not target_is_hexagon:
        outcome = "M2Q-E"
        statement = (
            "The V0 target itself does not pass the strict visual-hexagon classifier at this resolution "
            f"(strict class {target_gate['strict_class']}); it is only six-lobed/triangular-structured, so "
            "mask design against it would chase a non-hexagon."
        )
    elif not backward_ok:
        outcome = "M2Q-C"
        statement = (
            "Backward inversion does not recover Nathan's raw pre-axicon field from the V0 target: the "
            "inverse/forward operators are inconsistent and must be fixed before mask design."
        )
    elif any_pass:
        outcome = "M2Q-A"
        statement = (
            "Backward inversion recovers Nathan's raw pre-axicon field, and physically constrained "
            "phase-only SLM/QWP masks reproduce the V0 output after full forward propagation "
            "(strict hexagon gate included). Source-scale IRL replication via inverse-designed masks is plausible."
        )
    else:
        outcome = "M2Q-B"
        statement = (
            "Backward inversion recovers the required field, but the phase-only SLM / 4F constraints prevent "
            "accurate reproduction after forward propagation. Complex-amplitude modulation, different 4F "
            "filtering, or richer modulation is needed."
        )

    return {
        "stage": MODE2Q_STAGE,
        "suggested_outcome": outcome,
        "allowed_outcomes": MODE2Q_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "scope": MODE2Q_SCOPE_STATEMENT,
        "complex_vector_target_available": bool(complex_target_available),
        "target_strict_class": str(target_gate["strict_class"]),
        "target_passes_true_hexagon_gate": target_is_hexagon,
        "backward_recovery_overlap_to_raw_nathan": float(recovery["overlap_to_raw_nathan_input"]),
        "backward_recovery_pass": backward_ok,
        "evanescent_clipped_energy_fraction": float(diag["backpropagation"]["evanescent_clipped_energy_fraction"]),
        "inverse_axicon_condition_number": float(diag["inverse_axicon"]["jones_condition_number"]),
        "amplitude_vs_phase_only_supply": dict(diag["amplitude_vs_phase_only_supply"]),
        "four_f_adjoint": dict(diag["four_f_adjoint"]),
        "phase_only_direct_pass": bool(direct_pass),
        "phase_only_4f_pass": bool(filtered_pass),
        "four_f_is_the_blocker": bool(direct_pass and filtered and not filtered_pass),
        "optimisation_run": optimisation is not None,
        "optimisation": None if optimisation is None else dict(optimisation),
        "candidates": [mode2q_candidate_row(c) for c in candidates],
        "inherited_objective_sample_geometry_used": False,
        "microfabrication_sample_plane_claim": False,
        "micro_scale_note": (
            "MODE 2Q is source-scale inverse mask design only; it makes no statement about the inherited "
            "objective/sample microfabrication architecture."
        ),
    }


def mode2q_scope_manifest(outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Machine-readable MODE 2Q simulation-scope manifest."""

    return {
        "stage": MODE2Q_STAGE,
        "scope": MODE2Q_SCOPE_STATEMENT,
        "git_commit": _mode1_git_commit_short(),
        "target": "validated V0 complex vector field at z_ref = 60 mm (never the PNG/intensity alone)",
        "inverse_blocks": {
            "propagation": "exp(-i kz z), propagating modes only, evanescent clipping reported",
            "axicon": "reciprocal diagonal Jones in radial/azimuthal basis with condition-number report",
            "qwp_hwp": "unitary adjoint J^dagger with unitarity check",
            "four_f": "adjoint projection only; lost spectral energy reported; never an exact inverse",
        },
        "inherited_objective_sample_geometry": False,
        "micro_scale_sample_plane_simulated": False,
        "microfabrication_sample_plane_claim": False,
        "panel_realism": False,
        "strict_hexagon_gate": "six bright lobes are not a hexagon unless the C3-vs-C6 classifier passes",
        "suggested_outcome": None if outcome is None else outcome.get("suggested_outcome"),
        "claim_boundary": {
            "model_status": MODEL_STATUS,
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "material_model": False,
            "camera_model": False,
            "judged_by_mode1c_m1e_microfabrication_constraints": False,
        },
    }


def run_mode2q_lowdim_optimisation(
    data: Mapping[str, Any],
    v0: Mode2NRouteResult,
    target_z: VectorField,
    backward: Mode2QBackwardField,
    *,
    use_4f: bool = False,
    maxiter: int = 60,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Bounded low-dimensional sector precompensation via Nelder-Mead.

    Variables: six per-sector pistons, sector rotation, sector duty scale, and a
    global V-channel piston (9 parameters).  Pixel-level optimisation is out of
    scope until this low-dimensional model demonstrably fails.
    """

    from scipy.optimize import minimize

    cfg: NathanSourceParityConfig = data["config"]
    grid = data["grid"]
    theta = np.asarray(grid["PHI"], dtype=float)
    w = {"corr_xy": 1.0, "corr_angular": 0.5, "c3_excess": 1.0, "sector_imbalance": 0.25, "dark_core": 0.5, "regularisation": 0.05}
    w.update(dict(weights or {}))

    def _masks(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pistons = np.asarray(params[:6], dtype=float)
        rotation = float(params[6])
        duty_scale = float(np.clip(params[7], 0.5, 1.5))
        v_piston = float(params[8])
        alpha, _ = nathan_alpha_map(
            theta,
            sector_num_pairs=int(cfg.n_pairs),
            sector_theta=float(np.clip(cfg.sector_theta_rad * duty_scale, 0.1, TWOPI / cfg.n_pairs - 0.05)),
            sector_rotation=float(cfg.sector_rotation_rad + rotation),
        )
        cell = TWOPI / (2.0 * float(cfg.n_pairs))
        sector_idx = np.mod(np.floor((theta - cfg.sector_rotation_rad) / cell).astype(int), 6)
        piston_map = pistons[sector_idx]
        return wrap_2pi(alpha + piston_map), wrap_2pi(-alpha + 0.5 * np.pi + piston_map + v_piston)

    def _loss(params: np.ndarray) -> float:
        phi_h, phi_v = _masks(np.asarray(params, dtype=float))
        cand = run_mode2q_forward_candidate(
            "opt_eval", phi_h, phi_v, data, v0, target_z, backward, use_4f=use_4f,
        )
        cmp_ = cand["comparison"]
        strict = cand["strict_gate"]
        sym = dict(strict["symmetry"])
        c3_excess = max(0.0, float(strict["c120_minus_c60"]) - MODE2Q_C120_MINUS_C60_TOL)
        imbalance = max(0.0, float(sym.get("six_sector_max_over_min", 1.0)) - 1.0)
        dark_err = max(0.0, float(strict["dark_core_ratio"]) - MODE1_DARK_CORE_HOLLOW_THRESHOLD)
        reg = float(np.mean(np.square(np.asarray(params, dtype=float))))
        return (
            w["corr_xy"] * (1.0 - float(cmp_["z60_full_field_correlation"]))
            + w["corr_angular"] * (1.0 - float(cmp_["angular_profile_correlation_to_v0"]))
            + w["c3_excess"] * c3_excess
            + w["sector_imbalance"] * imbalance
            + w["dark_core"] * dark_err
            + w["regularisation"] * reg
        )

    x0 = np.zeros(9, dtype=float)
    result = minimize(_loss, x0, method="Nelder-Mead", options={"maxiter": int(maxiter), "xatol": 1e-3, "fatol": 1e-5})
    phi_h, phi_v = _masks(np.asarray(result.x, dtype=float))
    return {
        "variables": "six sector pistons, sector rotation, sector duty scale, global V piston",
        "initial_loss": float(_loss(x0)),
        "final_loss": float(result.fun),
        "n_evaluations": int(result.nfev),
        "converged": bool(result.success),
        "parameters": [float(v) for v in np.asarray(result.x, dtype=float)],
        "phi_H": phi_h,
        "phi_V": phi_v,
    }


def run_mode2q_backward_mask_synthesis(
    config: NathanSourceParityConfig | None = None,
    *,
    grid_n: int = 512,
    z_planes: int = 61,
    run_optimisation: str | bool = "auto",
    optimisation_maxiter: int = 60,
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM,
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
) -> dict[str, Any]:
    """Run the full MODE 2Q study: backward init, projections, forward verification."""

    data = mode2n_source_target(config, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    _, target_z, target_meta = mode2q_v0_complex_target(data)
    target_gate = mode2q_strict_hexagon_gate(np.asarray(v0.reference_plane, dtype=float), data["grid"])
    backward = run_mode2q_backward_initialisation(
        data, carrier_lpmm=float(carrier_lpmm), iris_radius_frac=float(iris_radius_frac),
    )
    candidates = [
        run_mode2q_forward_candidate(
            "phase_only_direct", backward.phi_H_initial, backward.phi_V_initial,
            data, v0, target_z, backward, use_4f=False,
        ),
        run_mode2q_forward_candidate(
            "phase_only_4f", backward.phi_H_initial, backward.phi_V_initial,
            data, v0, target_z, backward, use_4f=True,
            carrier_lpmm=float(carrier_lpmm), iris_radius_frac=float(iris_radius_frac),
        ),
    ]
    optimisation: dict[str, Any] | None = None
    should_optimise = bool(run_optimisation) if isinstance(run_optimisation, bool) else (
        run_optimisation == "always" or (run_optimisation == "auto" and not any(c["passes"] for c in candidates))
    )
    if should_optimise:
        optimisation = run_mode2q_lowdim_optimisation(
            data, v0, target_z, backward, use_4f=False, maxiter=int(optimisation_maxiter),
        )
        candidates.append(
            run_mode2q_forward_candidate(
                "optimised_lowdim", optimisation["phi_H"], optimisation["phi_V"],
                data, v0, target_z, backward, use_4f=False,
            )
        )
    outcome = mode2q_outcome_report(
        data=data,
        backward=backward,
        target_gate=target_gate,
        candidates=candidates,
        complex_target_available=bool(target_meta["complex_vector_target_available"]),
        optimisation=None if optimisation is None else {k: v for k, v in optimisation.items() if k not in {"phi_H", "phi_V"}},
    )
    return {
        "data": data,
        "v0": v0,
        "target_z": target_z,
        "target_gate": target_gate,
        "backward": backward,
        "candidates": candidates,
        "candidate_rows": tuple(mode2q_candidate_row(c) for c in candidates),
        "optimisation": optimisation,
        "outcome": outcome,
        "manifest": mode2q_scope_manifest(outcome),
    }


# ---------------------------------------------------------------------------
# MODE 2Q figures (package functions; the notebook calls these, no duplicated physics)
# ---------------------------------------------------------------------------


def plot_mode2q_backward_vs_raw(
    backward: Mode2QBackwardField,
    data: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = data["grid"]
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    raw_i = np.abs(np.asarray(data["target"][0])) ** 2 + np.abs(np.asarray(data["target"][1])) ** 2
    rec_i = np.abs(backward.Ex_required_pre_axicon) ** 2 + np.abs(backward.Ey_required_pre_axicon) ** 2
    st = stokes_from_linear_components(backward.Ex_required_pre_axicon, backward.Ey_required_pre_axicon)
    rec_angle = 0.5 * np.arctan2(np.asarray(st["S2"], dtype=float), np.asarray(st["S1"], dtype=float))
    err = np.abs(0.5 * np.angle(np.exp(2j * (rec_angle - np.asarray(data["alpha"], dtype=float)))))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    axes[0].imshow(_normalise_image(raw_i, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("raw Nathan pre-axicon intensity")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[1].imshow(_normalise_image(rec_i, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("backward-recovered required intensity")
    axes[1].set_xlabel("x (mm)")
    im = axes[2].imshow(err, origin="lower", extent=ext, cmap="magma", vmin=0.0, vmax=0.2)
    axes[2].set_title("|pol-angle error| to raw alpha (rad)")
    axes[2].set_xlabel("x (mm)")
    fig.colorbar(im, ax=axes[2], shrink=0.8)
    rec = dict(backward.diagnostics["recovery_vs_raw_nathan_input"])
    fig.suptitle(
        f"MODE 2Q backward recovery vs raw Nathan input (overlap {rec['overlap_to_raw_nathan_input']:.6f})"
    )
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2q_required_hv(
    backward: Mode2QBackwardField,
    data: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = data["grid"]
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    panels = (
        (backward.amp_H_required, "required |E_H|", "inferno", None),
        (backward.phi_H_initial, "required arg(E_H) (rad)", "twilight", (-np.pi, np.pi)),
        (backward.amp_V_required, "required |E_V|", "inferno", None),
        (backward.phi_V_initial, "required arg(E_V) (rad)", "twilight", (-np.pi, np.pi)),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.6), constrained_layout=True)
    for ax, (arr, title, cmap, clim) in zip(axes.ravel(), panels, strict=True):
        arr = np.asarray(arr, dtype=float)
        if clim is None:
            im = ax.imshow(_normalise_image(arr, local=True), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=1)
        else:
            im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=clim[0], vmax=clim[1])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.75)
    amp = dict(backward.diagnostics["amplitude_vs_phase_only_supply"])
    fig.suptitle(
        "MODE 2Q required H/V fields (amplitude mismatch to phase-only supply: "
        f"H {amp['amp_H_over_supply_rms']:.2e}, V {amp['amp_V_over_supply_rms']:.2e})"
    )
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2q_masks(
    phi_H: np.ndarray,
    phi_V: np.ndarray,
    data: Mapping[str, Any],
    *,
    title: str = "MODE 2Q phase-only projected masks",
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = data["grid"]
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), constrained_layout=True)
    for ax, (mask, label) in zip(axes, ((phi_H, "phi_H (rad, wrapped)"), (phi_V, "phi_V (rad, wrapped)")), strict=True):
        im = ax.imshow(wrap_2pi(np.asarray(mask, dtype=float)), origin="lower", extent=ext, cmap="twilight", vmin=0.0, vmax=TWOPI)
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(title)
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2q_candidate_z60(
    candidate: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    crop_fraction: float = 0.35,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = data["grid"]
    plane = np.asarray(candidate["reference_plane"], dtype=float)
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    crop, crop_grid = _mode1b_even_axis_crop(plane, grid, float(crop_fraction))
    xc = np.asarray(crop_grid["x"], dtype=float) / 1e-3
    ext_c = [float(xc[0]), float(xc[-1]), float(xc[0]), float(xc[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    axes[0].imshow(_normalise_image(plane, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"xy at z = {candidate['reference_z_m'] / 1e-3:.1f} mm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[1].imshow(_normalise_image(crop, local=True), origin="lower", extent=ext_c, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("central crop")
    axes[1].set_xlabel("x (mm)")
    strict = dict(candidate["strict_gate"])
    corr = float(candidate["comparison"]["z60_full_field_correlation"])
    fig.suptitle(
        f"MODE 2Q {candidate['candidate_id']}: {strict['strict_class']}, "
        f"V0 correlation {corr:.4f}, passes={candidate['passes']}"
    )
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2q_zstack_summary(
    candidate: Mapping[str, Any],
    v0: Mode2NRouteResult,
    data: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = data["grid"]
    x_mm = _mode2n_mm_axis(grid)
    z_mm = np.asarray(candidate["z_values_m"], dtype=float) / 1e-3
    ext = [float(x_mm[0]), float(x_mm[-1]), float(z_mm[0]), float(z_mm[-1])]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 6.2), constrained_layout=True)
    axes[0].imshow(_normalise_image(np.asarray(v0.xz_map, dtype=float), local=True), origin="lower", aspect="auto", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title("V0 reference (x-z)")
    axes[0].set_ylabel("z (mm)")
    axes[1].imshow(_normalise_image(np.asarray(candidate["xz_map"], dtype=float), local=True), origin="lower", aspect="auto", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[1].axhline(candidate["reference_z_m"] / 1e-3, color="white", lw=0.8, alpha=0.8)
    axes[1].set_title(f"{candidate['candidate_id']} (x-z)")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")
    frac = float(candidate["sampled_plane_true_hexagon_fraction"])
    fig.suptitle(f"MODE 2Q z-stack summary (sampled-plane true-hexagon fraction {frac:.2f})")
    _save_fig(fig, output_path)
    return fig, axes


def write_mode2q_outputs(
    config: NathanSourceParityConfig | None = None,
    *,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode2q_backward_mask_synthesis",
    report: Mapping[str, Any] | None = None,
    **run_kwargs: Any,
) -> dict[str, Path]:
    """Run (or reuse) the MODE 2Q study and write all required artefacts."""

    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = dict(report) if report is not None else run_mode2q_backward_mask_synthesis(config, **run_kwargs)
    data = study["data"]
    v0: Mode2NRouteResult = study["v0"]
    backward: Mode2QBackwardField = study["backward"]
    candidates = list(study["candidates"])
    by_id = {str(c["candidate_id"]): c for c in candidates}

    paths: dict[str, Path] = {
        "backward_diagnostics": out / "mode2q_backward_diagnostics.json",
        "mask_candidates_csv": out / "mode2q_mask_candidates.csv",
        "mask_candidates_json": out / "mode2q_mask_candidates.json",
        "outcome_report": out / "mode2q_outcome_report.json",
        "scope_manifest": out / "simulation_scope_manifest.json",
    }
    paths["backward_diagnostics"].write_text(json.dumps(_json_ready(dict(backward.diagnostics)), indent=2), encoding="utf-8")
    _write_rows(paths["mask_candidates_csv"], study["candidate_rows"])
    paths["mask_candidates_json"].write_text(json.dumps(_json_ready(study["candidate_rows"]), indent=2), encoding="utf-8")
    paths["outcome_report"].write_text(json.dumps(_json_ready(dict(study["outcome"])), indent=2), encoding="utf-8")
    paths["scope_manifest"].write_text(json.dumps(_json_ready(dict(study["manifest"])), indent=2), encoding="utf-8")

    fig, _ = plot_mode2n_z60(v0, output_path=out / "mode2q_v0_target_z60.png")
    plt.close(fig)
    paths["v0_target_z60"] = out / "mode2q_v0_target_z60.png"

    fig, _ = plot_mode2n_pre_axicon(
        backward.Ex_required_pre_axicon, backward.Ey_required_pre_axicon, data,
        title="MODE 2Q backward-required pre-axicon field",
        output_path=out / "mode2q_backward_required_pre_axicon.png",
    )
    plt.close(fig)
    paths["backward_required_pre_axicon"] = out / "mode2q_backward_required_pre_axicon.png"

    fig, _ = plot_mode2q_backward_vs_raw(backward, data, output_path=out / "mode2q_backward_vs_raw_nathan_input.png")
    plt.close(fig)
    paths["backward_vs_raw_nathan_input"] = out / "mode2q_backward_vs_raw_nathan_input.png"

    fig, _ = plot_mode2q_required_hv(backward, data, output_path=out / "mode2q_required_hv_amplitude_phase.png")
    plt.close(fig)
    paths["required_hv_amplitude_phase"] = out / "mode2q_required_hv_amplitude_phase.png"

    fig, _ = plot_mode2q_masks(
        backward.phi_H_initial, backward.phi_V_initial, data,
        output_path=out / "mode2q_phase_only_projected_masks.png",
    )
    plt.close(fig)
    paths["phase_only_projected_masks"] = out / "mode2q_phase_only_projected_masks.png"

    if "phase_only_direct" in by_id:
        fig, _ = plot_mode2q_candidate_z60(by_id["phase_only_direct"], data, output_path=out / "mode2q_forward_phase_only_z60.png")
        plt.close(fig)
        paths["forward_phase_only_z60"] = out / "mode2q_forward_phase_only_z60.png"
    if "optimised_lowdim" in by_id:
        fig, _ = plot_mode2q_candidate_z60(by_id["optimised_lowdim"], data, output_path=out / "mode2q_forward_optimised_z60.png")
        plt.close(fig)
        paths["forward_optimised_z60"] = out / "mode2q_forward_optimised_z60.png"

    summary_candidate = by_id.get("optimised_lowdim") or by_id.get("phase_only_direct") or candidates[0]
    fig, _ = plot_mode2q_zstack_summary(summary_candidate, v0, data, output_path=out / "mode2q_zstack_summary.png")
    plt.close(fig)
    paths["zstack_summary"] = out / "mode2q_zstack_summary.png"
    return paths


# ===========================================================================
# MODE 2S - degraded source-scale bench realism, tolerance audit, precompensation
# ===========================================================================
#
# MODE 2N/2Q solved the clean source-scale bench (forward and inverse).  MODE 2S
# asks the lab question: when realistic imperfections are included - SLM
# discretisation/quantisation/fill-factor/stroke, H/V registration, channel
# imbalance and piston drift, QWP angle/retardance errors, 4F iris errors,
# axicon alignment errors, input wavefront aberrations - does the source-scale
# hexagon survive, and can the bounded M2Q precompensator recover it with
# physically interpretable corrections?  Still source-scale only: no inherited
# objective/sample geometry and no microfabrication sample-plane claim.

MODE2S_STAGE = "nathan_mode2s_source_scale_lab_realism_tolerance"
MODE2S_ALLOWED_OUTCOMES = ("M2S-A", "M2S-B", "M2S-C", "M2S-D", "M2S-E")
MODE2S_PASS_CORRELATION = MODE2N_REALISTIC_PROPAGATED_CORRELATION_PASS
MODE2S_NEAR_RECOVERABLE_CORRELATION = 0.75
MODE2S_FAILURE_MODES = (
    "none",
    "triangular_dark_core",
    "six_lobed_structured",
    "blurred_hexagon",
    "power_loss",
    "zero_order_contamination",
    "asymmetric_lobes",
    "z_shifted_bessel_zone",
)
# Planning estimates of typical lab setting errors (NOT calibrated lab data);
# used only to suggest whether a measured tolerance is comfortable.
MODE2S_TYPICAL_LAB_ERRORS = {
    "phase_levels_min": 256,
    "hv_piston_rad": 0.2,
    "hv_amplitude_ratio_dev": 0.10,
    "qwp_angle_error_deg": 0.5,
    "qwp_retardance_error_deg": 2.0,
    "iris_radius_frac_dev": 0.10,
    "iris_decentre_lpmm": 0.5,
    "hv_shift_um": 16.0,
    "axicon_decentre_mm": 0.2,
    "z_offset_mm": 5.0,
}
MODE2S_SCOPE_STATEMENT = (
    "MODE 2S source-scale lab-realism tolerance audit and bounded inverse precompensation only: "
    "the clean M2N/M2Q dual-SLM + carrier + 4F + QWP + axicon bench is degraded by explicit, "
    "physically labelled imperfections (SLM discretisation/quantisation/fill-factor/stroke, H/V "
    "registration/imbalance/piston, QWP angle/retardance, 4F iris radius/decentre/shape, axicon "
    "alignment, input Zernike aberrations), swept one at a time and in representative combined "
    "cases, with the bounded M2Q-style precompensator as the recovery tool. No inherited "
    "objective/sample geometry, no microfabrication sample-plane claim, and no unconstrained "
    "pixel-level hologram optimisation."
)


@dataclass(frozen=True)
class Mode2SPerturbation:
    """One explicit, physically labelled degradation of the source-scale bench."""

    label: str = "clean"
    # A-D: SLM discretisation / quantisation / fill factor / stroke
    slm_aperture_clip: bool = False
    slm_pixel_pitch_m: float = 8.0e-6
    slm_resolution_x: int = 1920
    slm_resolution_y: int = 1080
    slm_pixelate: bool = False
    phase_levels: int | None = None
    lut_gamma: float = 1.0
    fill_factor: float = 1.0
    phase_stroke_rad: float | None = None
    # E-G: H/V registration, imbalance, piston
    hv_shift_x_m: float = 0.0
    hv_shift_y_m: float = 0.0
    hv_rotation_rad: float = 0.0
    hv_magnification: float = 1.0
    hv_amplitude_ratio: float = 1.0
    hv_piston_rad: float = 0.0
    # H: QWP errors
    qwp_angle_error_rad: float = 0.0
    qwp_retardance_error_rad: float = 0.0
    # I: 4F errors
    carrier_lpmm: float = MODE2N_DEFAULT_CARRIER_LPMM
    iris_radius_frac: float = MODE2N_DEFAULT_IRIS_RADIUS_FRAC
    iris_decentre_fx_lpmm: float = 0.0
    iris_decentre_fy_lpmm: float = 0.0
    iris_shape: str = "circular"
    # J: axicon errors and observation-plane offset
    axicon_base_angle_error_deg: float = 0.0
    axicon_decentre_x_m: float = 0.0
    axicon_decentre_y_m: float = 0.0
    axicon_tilt_x_rad: float = 0.0
    axicon_tilt_y_rad: float = 0.0
    axicon_n_error: float = 0.0
    z_offset_m: float = 0.0
    # K: input wavefront aberrations (rad at the normalisation radius)
    zernike_common: Mapping[str, float] = field(default_factory=dict)
    zernike_differential_v: Mapping[str, float] = field(default_factory=dict)


def mode2s_slm_aperture_fit_report(
    data: Mapping[str, Any],
    perturbation: Mode2SPerturbation | None = None,
) -> dict[str, Any]:
    """Report (never assume) how the 10 mm source window fits the real SLM.

    The HOLOEYE-class panel is 1920 x 1080 at 8 um (15.36 mm x 8.64 mm), so the
    10 mm source window does NOT fit vertically; the beam itself (1/e radius
    2 mm) does, and the clipped power fraction is reported.  Pixel pitch is also
    audited against the simulation grid: at source-scale grids the 8 um pitch is
    finer than the grid step, so per-pixel structure is not resolvable and the
    honest pixel-level statements are the sampling ratios reported here.
    """

    pert = perturbation or Mode2SPerturbation()
    grid = data["grid"]
    cfg: NathanSourceParityConfig = data["config"]
    dx = float(grid["dx"])
    window_m = float(grid["N"]) * dx
    active_w = float(pert.slm_resolution_x) * float(pert.slm_pixel_pitch_m)
    active_h = float(pert.slm_resolution_y) * float(pert.slm_pixel_pitch_m)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    inside = (np.abs(X) <= 0.5 * active_w) & (np.abs(Y) <= 0.5 * active_h)
    intensity = np.asarray(data["A"], dtype=float) ** 2
    clipped_power_fraction = float(np.sum(intensity[~inside]) / max(float(np.sum(intensity)), EPS))
    carrier_period_m = 1.0e-3 / max(float(pert.carrier_lpmm), EPS)
    return {
        "window_m": window_m,
        "slm_active_width_m": active_w,
        "slm_active_height_m": active_h,
        "window_fits_horizontally": bool(window_m <= active_w),
        "window_fits_vertically": bool(window_m <= active_h),
        "largest_valid_square_window_m": float(min(active_w, active_h)),
        "beam_1e_radius_m": float(cfg.beam_radius_m),
        "beam_power_clipped_by_active_area_fraction": clipped_power_fraction,
        "grid_dx_m": dx,
        "slm_pixel_pitch_m": float(pert.slm_pixel_pitch_m),
        "grid_step_over_pixel_pitch": float(dx / max(float(pert.slm_pixel_pitch_m), EPS)),
        "pixelation_resolvable_on_this_grid": bool(dx < float(pert.slm_pixel_pitch_m)),
        "carrier_period_in_slm_pixels": float(carrier_period_m / max(float(pert.slm_pixel_pitch_m), EPS)),
        "note": (
            "10 mm window exceeds the SLM short axis; the physical model clips the field by the active "
            "area rather than silently rescaling, and the Gaussian beam power lost to that clip is reported."
        ),
    }


def mode2s_quantise_phase(
    phase_rad: np.ndarray,
    *,
    phase_levels: int | None,
    lut_gamma: float = 1.0,
    phase_stroke_rad: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply LUT gamma, discrete phase levels, and finite stroke to a wrapped phase."""

    wrapped = wrap_2pi(phase_rad)
    out = wrapped
    if float(lut_gamma) != 1.0:
        out = TWOPI * np.power(out / TWOPI, float(lut_gamma))
    if phase_levels is not None:
        levels = max(2, int(phase_levels))
        out = np.round(out / TWOPI * levels) / levels * TWOPI
        out = np.mod(out, TWOPI)
    stroke_clipped_fraction = 0.0
    if phase_stroke_rad is not None and float(phase_stroke_rad) < TWOPI:
        stroke = float(phase_stroke_rad)
        stroke_clipped_fraction = float(np.mean(out > stroke))
        out = np.minimum(out, stroke)
    rms = float(np.sqrt(np.mean((np.angle(np.exp(1j * (out - wrapped)))) ** 2)))
    return out, {
        "phase_levels": None if phase_levels is None else int(phase_levels),
        "lut_gamma": float(lut_gamma),
        "phase_stroke_rad": None if phase_stroke_rad is None else float(phase_stroke_rad),
        "stroke_clipped_fraction": stroke_clipped_fraction,
        "phase_rms_error_rad": rms,
    }


def _mode2s_pixelate_phase(phase_rad: np.ndarray, grid: Mapping[str, Any], pitch_m: float) -> np.ndarray:
    """Piecewise-constant SLM pixel phase (no-op when the grid is coarser than the pitch)."""

    dx = float(grid["dx"])
    block = int(round(float(pitch_m) / max(dx, EPS)))
    if block <= 1:
        return np.asarray(phase_rad, dtype=float)
    arr = np.asarray(phase_rad, dtype=float)
    n = arr.shape[0]
    trimmed = arr[: (n // block) * block, : (n // block) * block]
    coarse = trimmed.reshape(n // block, block, n // block, block)[:, 0, :, 0]
    out = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)
    full = np.array(arr)
    full[: out.shape[0], : out.shape[1]] = out
    return full


def mode2s_zernike_phase(
    grid: Mapping[str, Any],
    coefficients: Mapping[str, float],
    *,
    normalisation_radius_m: float,
) -> np.ndarray:
    """Low-order Zernike-style phase (rad) with coefficients given at rho = 1."""

    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    rho = np.hypot(X, Y) / max(float(normalisation_radius_m), EPS)
    theta = np.arctan2(Y, X)
    terms = {
        "defocus": 2.0 * rho**2 - 1.0,
        "astig0": rho**2 * np.cos(2.0 * theta),
        "astig45": rho**2 * np.sin(2.0 * theta),
        "coma_x": (3.0 * rho**3 - 2.0 * rho) * np.cos(theta),
        "coma_y": (3.0 * rho**3 - 2.0 * rho) * np.sin(theta),
        "spherical": 6.0 * rho**4 - 6.0 * rho**2 + 1.0,
    }
    phase = np.zeros_like(rho)
    for key, value in dict(coefficients).items():
        if key not in terms:
            raise KeyError(f"unsupported Zernike term {key!r}; allowed: {sorted(terms)}")
        phase = phase + float(value) * terms[key]
    return phase


def _mode2s_transform_channel(
    field2d: np.ndarray,
    grid: Mapping[str, Any],
    *,
    shift_x_m: float = 0.0,
    shift_y_m: float = 0.0,
    rotation_rad: float = 0.0,
    magnification: float = 1.0,
) -> np.ndarray:
    """Sub-pixel shift / rotation / magnification of one channel (misregistration)."""

    if shift_x_m == 0.0 and shift_y_m == 0.0 and rotation_rad == 0.0 and magnification == 1.0:
        return np.asarray(field2d, dtype=np.complex128)
    arr = np.asarray(field2d, dtype=np.complex128)
    x = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    x0 = float(x[0])
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    c = np.cos(float(rotation_rad))
    s = np.sin(float(rotation_rad))
    m = max(float(magnification), EPS)
    # Inverse map: source coordinates that land on each output grid point.
    xs = (c * (X - float(shift_x_m)) + s * (Y - float(shift_y_m))) / m
    ys = (-s * (X - float(shift_x_m)) + c * (Y - float(shift_y_m))) / m
    cols = (xs - x0) / dx
    rows = (ys - x0) / dx
    real = map_coordinates(arr.real, [rows.ravel(), cols.ravel()], order=1, mode="constant", cval=0.0)
    imag = map_coordinates(arr.imag, [rows.ravel(), cols.ravel()], order=1, mode="constant", cval=0.0)
    return (real + 1j * imag).reshape(arr.shape)


def mode2s_apply_4f(
    eh: np.ndarray,
    ev: np.ndarray,
    grid: Mapping[str, Any],
    *,
    carrier_lpmm: float,
    iris_radius_frac: float,
    iris_decentre_fx_lpmm: float = 0.0,
    iris_decentre_fy_lpmm: float = 0.0,
    iris_shape: str = "circular",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Shared 4F first-order filter with iris radius/decentre/shape errors.

    Unlike the clean-path helper this supports a decentred and non-circular
    iris, so the residual tilt is reported (not asserted) and the power ledger
    plus zero-order leakage are returned for tolerance bookkeeping.
    """

    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    carrier_cpm = float(carrier_lpmm) * 1.0e3
    centre_fx = carrier_cpm + float(iris_decentre_fx_lpmm) * 1.0e3
    centre_fy = float(iris_decentre_fy_lpmm) * 1.0e3
    radius_cpm = float(iris_radius_frac) * carrier_cpm
    if str(iris_shape) == "circular":
        mask = (FX - centre_fx) ** 2 + (FY - centre_fy) ** 2 <= radius_cpm**2
    elif str(iris_shape) == "square":
        mask = (np.abs(FX - centre_fx) <= radius_cpm) & (np.abs(FY - centre_fy) <= radius_cpm)
    else:
        raise ValueError(f"unsupported iris shape {iris_shape!r}")
    X = np.asarray(grid["X"], dtype=float)
    demod = np.exp(-1j * TWOPI * carrier_cpm * X)
    dc_disk = FX**2 + FY**2 <= radius_cpm**2
    spec_h = fft2c(np.asarray(eh, dtype=np.complex128))
    spec_v = fft2c(np.asarray(ev, dtype=np.complex128))
    spec_power = np.abs(spec_h) ** 2 + np.abs(spec_v) ** 2
    total = float(np.sum(spec_power))
    passed = float(np.sum(spec_power[mask]))
    leakage = float(np.sum(spec_power[mask & dc_disk]) / max(total, EPS))
    eh_out = ifft2c(spec_h * mask) * demod
    ev_out = ifft2c(spec_v * mask) * demod
    centroid_fx = float(np.sum(FX[mask] * spec_power[mask]) / max(passed, EPS))
    return eh_out, ev_out, {
        "carrier_lpmm": float(carrier_lpmm),
        "iris_radius_lpmm": float(iris_radius_frac) * float(carrier_lpmm),
        "iris_decentre_fx_lpmm": float(iris_decentre_fx_lpmm),
        "iris_decentre_fy_lpmm": float(iris_decentre_fy_lpmm),
        "iris_shape": str(iris_shape),
        "first_order_efficiency": float(passed / max(total, EPS)),
        "rejected_power_fraction": float(1.0 - passed / max(total, EPS)),
        "zero_order_leakage_after_iris": leakage,
        "signal_spectral_centroid_offset_lpmm": float((centroid_fx - carrier_cpm) / 1.0e3),
    }


def _mode2s_apply_axicon_with_errors(
    field: VectorField,
    cfg: NathanSourceParityConfig,
    pert: Mode2SPerturbation,
) -> tuple[VectorField, dict[str, Any]]:
    """Nathan source axicon with base-angle / index / decentre / tilt errors."""

    grid = dict(field.grid)
    if pert.axicon_decentre_x_m != 0.0 or pert.axicon_decentre_y_m != 0.0:
        X = np.asarray(grid["X"], dtype=float) - float(pert.axicon_decentre_x_m)
        Y = np.asarray(grid["Y"], dtype=float) - float(pert.axicon_decentre_y_m)
        shifted = {**grid, "R": np.hypot(X, Y), "PHI": np.arctan2(Y, X)}
        work = VectorField(
            ex=field.ex, ey=field.ey, ez=field.ez, grid=shifted,
            wavelength_m=field.wavelength_m, medium_index=field.medium_index, metadata=dict(field.metadata),
        )
    else:
        work = field
    base_angle = float(cfg.axicon_base_angle_rad) + float(np.deg2rad(pert.axicon_base_angle_error_deg))
    after, meta = _apply_free_space_vector_axicon(
        work,
        n_axicon=float(cfg.axicon_n) + float(pert.axicon_n_error),
        n_medium=float(cfg.medium_n),
        base_angle_rad=base_angle,
    )
    if pert.axicon_tilt_x_rad != 0.0 or pert.axicon_tilt_y_rad != 0.0:
        # Small-angle tilt model: the deviated exit beam carries a linear phase ramp.
        k0 = TWOPI / float(field.wavelength_m)
        ramp = np.exp(
            1j * k0 * (
                float(pert.axicon_tilt_x_rad) * (float(cfg.axicon_n) - 1.0) * np.asarray(grid["X"], dtype=float)
                + float(pert.axicon_tilt_y_rad) * (float(cfg.axicon_n) - 1.0) * np.asarray(grid["Y"], dtype=float)
            )
        )
        after = VectorField(
            ex=after.ex * ramp, ey=after.ey * ramp, ez=after.ez * ramp, grid=field.grid,
            wavelength_m=after.wavelength_m, medium_index=after.medium_index, metadata=dict(after.metadata),
        )
    else:
        after = VectorField(
            ex=after.ex, ey=after.ey, ez=after.ez, grid=field.grid,
            wavelength_m=after.wavelength_m, medium_index=after.medium_index, metadata=dict(after.metadata),
        )
    meta = {**meta, "base_angle_error_deg": float(pert.axicon_base_angle_error_deg),
            "decentre_m": (float(pert.axicon_decentre_x_m), float(pert.axicon_decentre_y_m)),
            "tilt_rad": (float(pert.axicon_tilt_x_rad), float(pert.axicon_tilt_y_rad)),
            "n_error": float(pert.axicon_n_error)}
    return after, meta


@dataclass(frozen=True)
class Mode2SCorrection:
    """Bounded, physically interpretable precompensation applied to the clean masks.

    ``mask_recentre_*`` is the standard software hologram recentre: the sector
    pattern is redrawn about the measured axicon/beam axis.  The vector
    singularity must sit on the cone axis to a fraction of the ~64 um radial
    fringe period, so this is the first calibration any real bench performs.
    """

    sector_pistons_rad: tuple[float, float, float, float, float, float] = (0.0,) * 6
    global_v_piston_rad: float = 0.0
    sector_rotation_rad: float = 0.0
    sector_duty_scale: float = 1.0
    qwp_angle_correction_rad: float = 0.0
    defocus_rad: float = 0.0
    astig0_rad: float = 0.0
    astig45_rad: float = 0.0
    iris_recentre_fx_lpmm: float = 0.0
    iris_recentre_fy_lpmm: float = 0.0
    mask_recentre_x_m: float = 0.0
    mask_recentre_y_m: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {
            "sector_pistons_rad": tuple(float(v) for v in self.sector_pistons_rad),
            "global_v_piston_rad": float(self.global_v_piston_rad),
            "sector_rotation_deg": float(np.rad2deg(self.sector_rotation_rad)),
            "sector_duty_scale": float(self.sector_duty_scale),
            "qwp_angle_correction_deg": float(np.rad2deg(self.qwp_angle_correction_rad)),
            "defocus_rad": float(self.defocus_rad),
            "astig0_rad": float(self.astig0_rad),
            "astig45_rad": float(self.astig45_rad),
            "iris_recentre_fx_lpmm": float(self.iris_recentre_fx_lpmm),
            "iris_recentre_fy_lpmm": float(self.iris_recentre_fy_lpmm),
            "mask_recentre_x_um": float(self.mask_recentre_x_m / 1e-6),
            "mask_recentre_y_um": float(self.mask_recentre_y_m / 1e-6),
        }


def _mode2s_masks_with_correction(
    data: Mapping[str, Any],
    correction: Mode2SCorrection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean Nathan inverse masks plus the interpretable correction terms."""

    cfg: NathanSourceParityConfig = data["config"]
    grid = data["grid"]
    # The hologram pattern is drawn about the (possibly recentred) mask origin.
    Xc = np.asarray(grid["X"], dtype=float) - float(correction.mask_recentre_x_m)
    Yc = np.asarray(grid["Y"], dtype=float) - float(correction.mask_recentre_y_m)
    theta = np.arctan2(Yc, Xc)
    alpha, _ = nathan_alpha_map(
        theta,
        sector_num_pairs=int(cfg.n_pairs),
        sector_theta=float(np.clip(cfg.sector_theta_rad * float(correction.sector_duty_scale), 0.1, TWOPI / cfg.n_pairs - 0.05)),
        sector_rotation=float(cfg.sector_rotation_rad + correction.sector_rotation_rad),
    )
    cell = TWOPI / (2.0 * float(cfg.n_pairs))
    sector_idx = np.mod(np.floor((theta - float(cfg.sector_rotation_rad)) / cell).astype(int), 6)
    piston_map = np.asarray(correction.sector_pistons_rad, dtype=float)[sector_idx]
    zern = mode2s_zernike_phase(
        grid,
        {"defocus": correction.defocus_rad, "astig0": correction.astig0_rad, "astig45": correction.astig45_rad},
        normalisation_radius_m=2.0 * float(cfg.beam_radius_m),
    )
    phi_h = alpha + piston_map + zern
    phi_v = -alpha + 0.5 * np.pi + piston_map + float(correction.global_v_piston_rad) + zern
    return phi_h, phi_v, zern


def run_mode2s_degraded_forward(
    data: Mapping[str, Any],
    v0: Mode2NRouteResult,
    backward: Mode2QBackwardField,
    perturbation: Mode2SPerturbation,
    *,
    correction: Mode2SCorrection | None = None,
    sweep_parameter: str = "",
    sweep_value: float = float("nan"),
    fast_single_plane: bool = False,
) -> dict[str, Any]:
    """Full degraded source-scale forward run for one perturbation (+ optional correction).

    ``fast_single_plane=True`` propagates only the (offset) reference plane; it is
    used inside the precompensation loss where the x-z/z-shift metrics are not needed.
    """

    cfg: NathanSourceParityConfig = data["config"]
    grid = data["grid"]
    A = np.asarray(data["A"], dtype=float)
    supply = A / np.sqrt(2.0)
    corr = correction or Mode2SCorrection()
    if correction is None:
        phi_h_mask = np.asarray(backward.phi_H_initial, dtype=float)
        phi_v_mask = np.asarray(backward.phi_V_initial, dtype=float)
    else:
        phi_h_mask, phi_v_mask, _ = _mode2s_masks_with_correction(data, corr)

    norm_radius = 2.0 * float(cfg.beam_radius_m)
    common = mode2s_zernike_phase(grid, perturbation.zernike_common, normalisation_radius_m=norm_radius) if perturbation.zernike_common else 0.0
    diff_v = mode2s_zernike_phase(grid, perturbation.zernike_differential_v, normalisation_radius_m=norm_radius) if perturbation.zernike_differential_v else 0.0

    carrier_cpm = float(perturbation.carrier_lpmm) * 1.0e3
    carrier_phase = TWOPI * carrier_cpm * np.asarray(grid["X"], dtype=float)
    quant_meta: dict[str, Any] = {}
    channels: dict[str, np.ndarray] = {}
    for key, mask_phase, extra in (("H", phi_h_mask, 0.0), ("V", phi_v_mask, diff_v)):
        hologram = mask_phase + carrier_phase + common + extra
        if perturbation.slm_pixelate:
            hologram = _mode2s_pixelate_phase(hologram, grid, float(perturbation.slm_pixel_pitch_m))
        displayed, qmeta = mode2s_quantise_phase(
            hologram,
            phase_levels=perturbation.phase_levels,
            lut_gamma=float(perturbation.lut_gamma),
            phase_stroke_rad=perturbation.phase_stroke_rad,
        )
        quant_meta[key] = qmeta
        ff = float(np.clip(perturbation.fill_factor, 0.0, 1.0))
        # Dead-space light is unmodulated (no carrier), so the 4F iris rejects it downstream.
        channel = supply * (ff * np.exp(1j * displayed) + (1.0 - ff))
        if perturbation.slm_aperture_clip:
            X = np.asarray(grid["X"], dtype=float)
            Y = np.asarray(grid["Y"], dtype=float)
            active = (
                (np.abs(X) <= 0.5 * float(perturbation.slm_resolution_x) * float(perturbation.slm_pixel_pitch_m))
                & (np.abs(Y) <= 0.5 * float(perturbation.slm_resolution_y) * float(perturbation.slm_pixel_pitch_m))
            )
            channel = channel * active
        channels[key] = channel

    channels["V"] = _mode2s_transform_channel(
        channels["V"], grid,
        shift_x_m=float(perturbation.hv_shift_x_m),
        shift_y_m=float(perturbation.hv_shift_y_m),
        rotation_rad=float(perturbation.hv_rotation_rad),
        magnification=float(perturbation.hv_magnification),
    )
    channels["V"] = channels["V"] * float(perturbation.hv_amplitude_ratio) * np.exp(1j * float(perturbation.hv_piston_rad))

    eh_f, ev_f, iris_meta = mode2s_apply_4f(
        channels["H"], channels["V"], grid,
        carrier_lpmm=float(perturbation.carrier_lpmm),
        iris_radius_frac=float(perturbation.iris_radius_frac),
        iris_decentre_fx_lpmm=float(perturbation.iris_decentre_fx_lpmm) + float(corr.iris_recentre_fx_lpmm),
        iris_decentre_fy_lpmm=float(perturbation.iris_decentre_fy_lpmm) + float(corr.iris_recentre_fy_lpmm),
        iris_shape=str(perturbation.iris_shape),
    )
    qwp_angle = -0.25 * np.pi + float(perturbation.qwp_angle_error_rad) + float(corr.qwp_angle_correction_rad)
    qwp_jones = linear_retarder(0.5 * np.pi + float(perturbation.qwp_retardance_error_rad), qwp_angle)
    Ex, Ey = apply_uniform_jones(qwp_jones, eh_f, ev_f)

    # Pre-axicon fidelity vs the clean required fields.
    mask = data["metric_mask"]
    h_overlap = complex_vector_overlap((eh_f, np.zeros_like(eh_f)), (backward.Ex_required_pre_qwp, np.zeros_like(eh_f)), mask)
    v_overlap = complex_vector_overlap((ev_f, np.zeros_like(ev_f)), (backward.Ey_required_pre_qwp, np.zeros_like(ev_f)), mask)
    pre = jones_metric_row("pre_axicon", (Ex, Ey), (backward.Ex_required_pre_axicon, backward.Ey_required_pre_axicon), mask=mask)

    field = _mode2n_vector_field(Ex, Ey, data)
    after, axicon_meta = _mode2s_apply_axicon_with_errors(field, cfg, perturbation)
    # Camera frame: a decentred axicon forms its pattern about the cone axis, and a real
    # bench aligns the camera to the beam axis, so evaluation happens in that frame.  The
    # translation is applied to the field with the exact spectral shift theorem (it
    # commutes with propagation); the mask/axicon structural mismatch stays fully modelled.
    if perturbation.axicon_decentre_x_m != 0.0 or perturbation.axicon_decentre_y_m != 0.0:
        kx = TWOPI * np.asarray(grid["FX"], dtype=float)
        ky = TWOPI * np.asarray(grid["FY"], dtype=float)
        ramp = np.exp(1j * (kx * float(perturbation.axicon_decentre_x_m) + ky * float(perturbation.axicon_decentre_y_m)))
        after = VectorField(
            ex=ifft2c(fft2c(after.ex) * ramp),
            ey=ifft2c(fft2c(after.ey) * ramp),
            ez=ifft2c(fft2c(after.ez) * ramp),
            grid=after.grid,
            wavelength_m=after.wavelength_m,
            medium_index=after.medium_index,
            metadata=dict(after.metadata),
        )
        evaluation_frame = "beam_axis_camera_aligned"
    else:
        evaluation_frame = "grid_origin"
    if fast_single_plane:
        z_values = np.asarray([float(cfg.z_reference_m) + float(perturbation.z_offset_m)], dtype=float)
    else:
        z_values = _v0_z_values(cfg)
    stack, _ = _free_space_intensity_stack(after, z_values)
    ref_nominal = _nearest_z_index(z_values, float(cfg.z_reference_m))
    ref = _nearest_z_index(z_values, float(cfg.z_reference_m) + float(perturbation.z_offset_m))
    if fast_single_plane:
        plane = np.asarray(stack[ref], dtype=float)
        v0_plane_arr = np.asarray(v0.reference_plane, dtype=float)
        full = _equal_power_shape_metrics(plane[None, ...], v0_plane_arr[None, ...], crop_fraction=1.0)
        crop = _equal_power_shape_metrics(plane[None, ...], v0_plane_arr[None, ...], crop_fraction=0.5)
        own = _mode2n_reference_plane_metrics(plane, grid)
        _, prof_route = angular_profile_on_ring(plane, grid, float(own["ring_radius_m"]))
        _, prof_v0 = angular_profile_on_ring(v0_plane_arr, grid, float(v0.ring_radius_m))
        ang_corr, ang_shift = circular_profile_correlation(prof_route, prof_v0)
        comparison = {
            **own,
            "z60_full_field_correlation": float(full["equal_power_intensity_correlation"]),
            "z60_full_field_shape_rms": float(full["equal_power_shape_rms"]),
            "z60_central_crop_correlation": float(crop["equal_power_intensity_correlation"]),
            "z60_central_crop_shape_rms": float(crop["equal_power_shape_rms"]),
            "angular_profile_correlation_to_v0": float(ang_corr),
            "angular_profile_best_shift_bins": int(ang_shift),
            "xz_map_correlation_to_v0": float("nan"),
            "on_axis_intensity_correlation_to_v0": float("nan"),
            "reference_z_m": float(z_values[ref]),
        }
    else:
        comparison = mode2n_compare_stacks_to_v0(stack, z_values, ref, grid, v0)
    strict = mode2q_strict_hexagon_gate(np.asarray(stack[ref], dtype=float), grid)

    passes = bool(
        float(comparison["z60_full_field_correlation"]) >= MODE2S_PASS_CORRELATION
        and bool(strict["passes_true_hexagon_gate"])
    )
    # Best plane over the stack: distinguishes a z-shifted zone from a destroyed beam.
    best_corr = -np.inf
    best_idx = ref
    v0_plane = np.asarray(v0.reference_plane, dtype=float)
    for idx in range(stack.shape[0]):
        c = _equal_power_shape_metrics(np.asarray(stack[idx], dtype=float)[None, ...], v0_plane[None, ...], crop_fraction=1.0)
        val = float(c["equal_power_intensity_correlation"])
        if val > best_corr:
            best_corr = val
            best_idx = idx
    failure = mode2s_failure_mode(
        passes=passes,
        strict=strict,
        comparison=comparison,
        iris_meta=iris_meta,
        best_plane_correlation=float(best_corr),
        best_plane_index=int(best_idx),
        reference_index=int(ref_nominal),
    )
    mid = stack.shape[1] // 2
    return {
        "label": str(perturbation.label),
        "sweep_parameter": str(sweep_parameter),
        "sweep_value": float(sweep_value),
        "perturbation": perturbation,
        "correction": None if correction is None else corr,
        "compensated": bool(correction is not None),
        "evaluation_frame": evaluation_frame,
        "h_channel_overlap_to_required": float(h_overlap),
        "v_channel_overlap_to_required": float(v_overlap),
        "pre_axicon": pre,
        "pre_axicon_field": (
            np.asarray(Ex, dtype=np.complex128),
            np.asarray(Ey, dtype=np.complex128),
        ),
        "quantisation": quant_meta,
        "iris": iris_meta,
        "axicon": axicon_meta,
        "z_values_m": np.asarray(z_values, dtype=float),
        "reference_index": int(ref),
        "reference_z_m": float(z_values[ref]),
        "reference_plane": np.asarray(stack[ref], dtype=np.float32),
        "xz_map": np.asarray(stack[:, mid, :], dtype=np.float32),
        "yz_map": np.asarray(stack[:, :, mid], dtype=np.float32),
        "comparison": comparison,
        "strict_gate": strict,
        "best_plane_correlation": float(best_corr),
        "best_plane_z_m": float(z_values[best_idx]),
        "passes": passes,
        "failure_mode": failure,
    }


def mode2s_failure_mode(
    *,
    passes: bool,
    strict: Mapping[str, Any],
    comparison: Mapping[str, Any],
    iris_meta: Mapping[str, Any],
    best_plane_correlation: float,
    best_plane_index: int,
    reference_index: int,
) -> str:
    """Classify the dominant failure mode of a degraded run (heuristic, documented)."""

    if passes:
        return "none"
    cls = str(strict["strict_class"])
    if cls == "triangular_lobed_field":
        return "triangular_dark_core"
    if cls == "six_lobed_structured":
        return "six_lobed_structured"
    if float(best_plane_correlation) >= MODE2S_PASS_CORRELATION and int(best_plane_index) != int(reference_index):
        return "z_shifted_bessel_zone"
    if float(iris_meta.get("zero_order_leakage_after_iris", 0.0)) > 1.0e-3:
        return "zero_order_contamination"
    if float(iris_meta.get("first_order_efficiency", 1.0)) < 0.5:
        return "power_loss"
    sym = dict(strict.get("symmetry", {}))
    if float(sym.get("six_sector_max_over_min", 1.0)) >= 1.6:
        return "asymmetric_lobes"
    return "blurred_hexagon"


def mode2s_case_row(case: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one MODE 2S degraded run into a CSV/JSON-safe row."""

    pre = dict(case["pre_axicon"])
    cmp_ = dict(case["comparison"])
    strict = dict(case["strict_gate"])
    iris = dict(case["iris"])
    quant = dict(case["quantisation"])
    row = {
        "label": str(case["label"]),
        "sweep_parameter": str(case["sweep_parameter"]),
        "sweep_value": float(case["sweep_value"]),
        "compensated": bool(case["compensated"]),
        "h_channel_overlap_to_required": float(case["h_channel_overlap_to_required"]),
        "v_channel_overlap_to_required": float(case["v_channel_overlap_to_required"]),
        "pre_axicon_overlap_to_required": float(pre["complex_vector_overlap"]),
        "pre_axicon_stokes_rms": float(pre["stokes_rms"]),
        "pre_axicon_alpha_rms": float(pre["alpha_angle_rms_mod_pi"]),
        "pre_axicon_power_ratio": float(pre["power_ratio"]),
        "phase_rms_error_rad_h": float(quant.get("H", {}).get("phase_rms_error_rad", np.nan)),
        "first_order_efficiency": float(iris["first_order_efficiency"]),
        "zero_order_leakage_after_iris": float(iris["zero_order_leakage_after_iris"]),
        "rejected_power_fraction": float(iris["rejected_power_fraction"]),
        "reference_z_mm": float(case["reference_z_m"] / 1e-3),
        "z60_full_field_correlation": float(cmp_["z60_full_field_correlation"]),
        "z60_central_crop_correlation": float(cmp_["z60_central_crop_correlation"]),
        "xz_map_correlation_to_v0": float(cmp_["xz_map_correlation_to_v0"]),
        "angular_profile_correlation_to_v0": float(cmp_["angular_profile_correlation_to_v0"]),
        "c60": float(dict(strict["symmetry"]).get("rot_corr_60", np.nan)),
        "c120": float(dict(strict["symmetry"]).get("rot_corr_120", np.nan)),
        "c120_minus_c60": float(strict["c120_minus_c60"]),
        "dark_core_ratio": float(strict["dark_core_ratio"]),
        "ring_island_count": int(strict["ring_island_count"]),
        "strict_class": str(strict["strict_class"]),
        "best_plane_correlation": float(case["best_plane_correlation"]),
        "best_plane_z_mm": float(case["best_plane_z_m"] / 1e-3),
        "passes": bool(case["passes"]),
        "failure_mode": str(case["failure_mode"]),
    }
    if case["correction"] is not None:
        row["correction"] = json.dumps(_json_ready(case["correction"].as_row()))
    else:
        row["correction"] = ""
    return row


def mode2s_tier1_sweeps() -> dict[str, dict[str, Any]]:
    """Tier-1 single-parameter sweep definitions (nominal value included in each)."""

    base = Mode2SPerturbation(slm_aperture_clip=True)

    def _cases(param: str, values: Sequence[Any], **fields_for_value: Any) -> dict[str, Any]:
        cases = []
        for value in values:
            overrides = {key: (value if template == "value" else template(value)) for key, template in fields_for_value.items()}
            label = f"{param}_{value}"
            cases.append((float(value if isinstance(value, (int, float)) and value is not None else np.nan), replace(base, label=label, **overrides)))
        return {"parameter": param, "cases": cases}

    sweeps: dict[str, dict[str, Any]] = {}
    quant_cases = []
    for levels in (None, 1024, 256, 64, 16):
        quant_cases.append((
            float(np.inf if levels is None else levels),
            replace(base, label=f"phase_levels_{levels}", phase_levels=levels),
        ))
    sweeps["phase_quantisation"] = {"parameter": "phase_levels", "cases": quant_cases}
    sweeps["hv_piston"] = _cases(
        "hv_piston_rad", [0.0, 0.1, 0.2, 0.4, 0.8, 1.6, np.pi, 4.7, 2.0 * np.pi - 0.4], hv_piston_rad="value",
    )
    sweeps["hv_amplitude_ratio"] = _cases(
        "hv_amplitude_ratio", [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2], hv_amplitude_ratio="value",
    )
    sweeps["qwp_angle"] = _cases(
        "qwp_angle_error_deg", [-2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
        qwp_angle_error_rad=lambda v: float(np.deg2rad(v)),
    )
    sweeps["qwp_retardance"] = _cases(
        "qwp_retardance_error_deg", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
        qwp_retardance_error_rad=lambda v: float(np.deg2rad(v)),
    )
    sweeps["iris_radius"] = _cases(
        "iris_radius_frac", [0.24, 0.32, 0.40, 0.48, 0.60, 0.80], iris_radius_frac="value",
    )
    sweeps["iris_decentre"] = _cases(
        "iris_decentre_fx_lpmm", [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], iris_decentre_fx_lpmm="value",
    )
    sweeps["hv_shift"] = _cases(
        "hv_shift_um", [0.0, 8.0, 16.0, 40.0, 80.0, 160.0], hv_shift_x_m=lambda v: float(v) * 1.0e-6,
    )
    sweeps["axicon_decentre"] = _cases(
        "axicon_decentre_mm", [0.0, 0.05, 0.1, 0.2, 0.5, 1.0], axicon_decentre_x_m=lambda v: float(v) * 1.0e-3,
    )
    sweeps["z_offset"] = _cases(
        "z_offset_mm", [-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0], z_offset_m=lambda v: float(v) * 1.0e-3,
    )
    return sweeps


def mode2s_combined_cases() -> tuple[Mode2SPerturbation, ...]:
    """Tier-2 representative combined lab cases (mild / moderate / bad)."""

    return (
        Mode2SPerturbation(
            label="combined_mild_lab",
            slm_aperture_clip=True, phase_levels=256, fill_factor=0.95,
            hv_shift_x_m=8.0e-6, hv_piston_rad=0.1, hv_amplitude_ratio=1.05,
            qwp_angle_error_rad=float(np.deg2rad(0.25)), qwp_retardance_error_rad=float(np.deg2rad(1.0)),
            iris_decentre_fx_lpmm=0.25, axicon_decentre_x_m=0.05e-3, z_offset_m=1.0e-3,
            zernike_common={"defocus": 0.1},
        ),
        Mode2SPerturbation(
            label="combined_moderate_lab",
            slm_aperture_clip=True, phase_levels=256, fill_factor=0.90,
            hv_shift_x_m=24.0e-6, hv_rotation_rad=float(np.deg2rad(0.2)), hv_piston_rad=0.3, hv_amplitude_ratio=1.1,
            qwp_angle_error_rad=float(np.deg2rad(1.0)), qwp_retardance_error_rad=float(np.deg2rad(2.0)),
            iris_radius_frac=0.36, iris_decentre_fx_lpmm=0.5, axicon_decentre_x_m=0.2e-3, z_offset_m=5.0e-3,
            zernike_common={"defocus": 0.3, "astig0": 0.2},
        ),
        Mode2SPerturbation(
            label="combined_bad_lab",
            slm_aperture_clip=True, phase_levels=256, fill_factor=0.85,
            hv_shift_x_m=80.0e-6, hv_rotation_rad=float(np.deg2rad(0.5)), hv_piston_rad=1.0, hv_amplitude_ratio=1.2,
            qwp_angle_error_rad=float(np.deg2rad(2.0)), qwp_retardance_error_rad=float(np.deg2rad(5.0)),
            iris_radius_frac=0.32, iris_decentre_fx_lpmm=1.0, axicon_decentre_x_m=0.5e-3,
            axicon_tilt_x_rad=float(np.deg2rad(0.2)), z_offset_m=10.0e-3,
            zernike_common={"defocus": 0.5, "astig0": 0.4, "coma_x": 0.3},
        ),
    )


def mode2s_tolerance_from_sweep(
    parameter: str,
    cases: Sequence[Mapping[str, Any]],
    *,
    nominal_value: float = 0.0,
    wrap_period: float | None = None,
) -> dict[str, Any]:
    """Extract the maximum passing deviation from a Tier-1 sweep.

    ``wrap_period`` treats the swept parameter as periodic (e.g. a phase piston),
    so a value near the period counts as a small negative deviation.
    """

    def _deviation(value: float) -> float:
        dev = abs(value - float(nominal_value))
        if wrap_period is not None:
            period = float(wrap_period)
            dev = min(dev % period, period - dev % period)
        return dev

    rows = sorted(cases, key=lambda c: float(c["sweep_value"]))
    passing = [float(c["sweep_value"]) for c in rows if bool(c["passes"])]
    failing = [(float(c["sweep_value"]), str(c["failure_mode"])) for c in rows if not bool(c["passes"])]
    if passing:
        finite = [v for v in passing if np.isfinite(v)]
        max_dev = float(max(_deviation(v) for v in finite)) if finite else float("inf")
    else:
        max_dev = 0.0
    dominant = ""
    if failing:
        modes = [mode for _, mode in failing]
        dominant = max(set(modes), key=modes.count)
    return {
        "parameter": str(parameter),
        "nominal_value": float(nominal_value),
        "n_cases": int(len(rows)),
        "n_pass": int(len(passing)),
        "n_fail": int(len(failing)),
        "max_passing_deviation": max_dev,
        "all_cases_pass": bool(not failing),
        "dominant_failure_mode": dominant,
    }


def run_mode2s_precompensation(
    data: Mapping[str, Any],
    v0: Mode2NRouteResult,
    backward: Mode2QBackwardField,
    perturbation: Mode2SPerturbation,
    *,
    maxiter: int = 80,
) -> tuple[Mode2SCorrection, dict[str, Any]]:
    """Bounded Nelder-Mead precompensation for one degraded case.

    Sixteen physically interpretable variables (six sector pistons, global V
    piston, sector rotation, duty scale, QWP angle correction, defocus/astig
    corrections, iris recentre, hologram/mask recentre x/y).  Values are
    clipped to physical bounds inside the loss and the simplex starts at
    lab-meaningful step sizes, so the optimiser cannot invent an unrelated
    hologram: the result is always 'clean Nathan inverse mask + interpretable
    correction'.  The loss uses the fast single-plane forward.
    """

    from scipy.optimize import minimize

    bounds = {
        "sector_piston": 0.5 * np.pi,
        "v_piston": np.pi,
        "rotation": float(np.deg2rad(15.0)),
        "duty_scale": (0.7, 1.3),
        "qwp": float(np.deg2rad(3.0)),
        "zernike": 1.0,
        "iris": 1.5,
        "mask_recentre": 1.0e-3,
    }

    def _correction(params: np.ndarray) -> Mode2SCorrection:
        p = np.asarray(params, dtype=float)
        return Mode2SCorrection(
            sector_pistons_rad=tuple(float(np.clip(v, -bounds["sector_piston"], bounds["sector_piston"])) for v in p[:6]),
            global_v_piston_rad=float(np.clip(p[6], -bounds["v_piston"], bounds["v_piston"])),
            sector_rotation_rad=float(np.clip(p[7], -bounds["rotation"], bounds["rotation"])),
            sector_duty_scale=float(np.clip(1.0 + p[8], *bounds["duty_scale"])),
            qwp_angle_correction_rad=float(np.clip(p[9], -bounds["qwp"], bounds["qwp"])),
            defocus_rad=float(np.clip(p[10], -bounds["zernike"], bounds["zernike"])),
            astig0_rad=float(np.clip(p[11], -bounds["zernike"], bounds["zernike"])),
            astig45_rad=float(np.clip(p[12], -bounds["zernike"], bounds["zernike"])),
            iris_recentre_fx_lpmm=float(np.clip(p[13], -bounds["iris"], bounds["iris"])),
            mask_recentre_x_m=float(np.clip(p[14], -bounds["mask_recentre"], bounds["mask_recentre"])),
            mask_recentre_y_m=float(np.clip(p[15], -bounds["mask_recentre"], bounds["mask_recentre"])),
        )

    def _loss(params: np.ndarray) -> float:
        case = run_mode2s_degraded_forward(
            data, v0, backward, perturbation, correction=_correction(params), fast_single_plane=True,
        )
        cmp_ = case["comparison"]
        strict = case["strict_gate"]
        sym = dict(strict["symmetry"])
        c3_excess = max(0.0, float(strict["c120_minus_c60"]) - MODE2Q_C120_MINUS_C60_TOL)
        imbalance = max(0.0, float(sym.get("six_sector_max_over_min", 1.0)) - 1.0)
        dark_err = max(0.0, float(strict["dark_core_ratio"]) - MODE1_DARK_CORE_HOLLOW_THRESHOLD)
        scaled = np.asarray(params, dtype=float) / _scales
        reg = 0.002 * float(np.mean(np.square(scaled)))
        return (
            (1.0 - float(cmp_["z60_full_field_correlation"]))
            + 0.5 * (1.0 - float(cmp_["angular_profile_correlation_to_v0"]))
            + c3_excess + 0.25 * imbalance + 0.5 * dark_err + reg
        )

    # Lab-meaningful initial simplex steps; Nelder-Mead's default steps at x0 = 0
    # are ~2.5e-4 per coordinate, far below any physically relevant scale.
    _scales = np.asarray([0.3] * 6 + [0.5, 0.05, 0.05, 0.01, 0.2, 0.2, 0.2, 0.3, 5.0e-5, 5.0e-5], dtype=float)
    x0 = np.zeros(16, dtype=float)
    # Measured-decentre seeding: a real bench measures the beam axis on the camera and
    # recentres the hologram onto it directly, so the search starts from that calibration.
    x0[14] = float(np.clip(perturbation.axicon_decentre_x_m, -bounds["mask_recentre"], bounds["mask_recentre"]))
    x0[15] = float(np.clip(perturbation.axicon_decentre_y_m, -bounds["mask_recentre"], bounds["mask_recentre"]))
    simplex = np.vstack([x0] + [x0 + _scales * np.eye(16)[i] for i in range(16)])
    result = minimize(
        _loss, x0, method="Nelder-Mead",
        options={"maxiter": int(maxiter), "initial_simplex": simplex, "xatol": 1e-3, "fatol": 1e-5},
    )
    correction = _correction(np.asarray(result.x, dtype=float))
    return correction, {
        "variables": (
            "six sector pistons, global V piston, sector rotation, sector duty scale, "
            "QWP angle correction, defocus/astig0/astig45 corrections, iris recentre fx, "
            "hologram/mask recentre x/y"
        ),
        "n_parameters": 16,
        "bounds": {
            "sector_piston_rad": bounds["sector_piston"],
            "global_v_piston_rad": bounds["v_piston"],
            "sector_rotation_rad": bounds["rotation"],
            "sector_duty_scale": list(bounds["duty_scale"]),
            "qwp_angle_correction_rad": bounds["qwp"],
            "zernike_rad": bounds["zernike"],
            "iris_recentre_lpmm": bounds["iris"],
            "mask_recentre_m": bounds["mask_recentre"],
        },
        "initial_loss": float(_loss(x0)),
        "final_loss": float(result.fun),
        "n_evaluations": int(result.nfev),
        "converged": bool(result.success),
    }


def mode2s_outcome_report(
    *,
    clean_case: Mapping[str, Any],
    clean_reference_correlation: float,
    fit_report: Mapping[str, Any],
    tolerances: Sequence[Mapping[str, Any]],
    combined_cases: Sequence[Mapping[str, Any]],
    compensated_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose exactly one M2S-A/B/C/D/E outcome from the tolerance evidence."""

    clean_ok = bool(clean_case["passes"])
    clean_matches_reference = bool(
        np.isfinite(float(clean_case["comparison"]["z60_full_field_correlation"]))
        and abs(float(clean_case["comparison"]["z60_full_field_correlation"]) - float(clean_reference_correlation)) <= 0.02
    )
    typical = MODE2S_TYPICAL_LAB_ERRORS
    typical_map = {
        "hv_piston_rad": typical["hv_piston_rad"],
        "hv_amplitude_ratio": typical["hv_amplitude_ratio_dev"],
        "qwp_angle_error_deg": typical["qwp_angle_error_deg"],
        "qwp_retardance_error_deg": typical["qwp_retardance_error_deg"],
        "iris_radius_frac": typical["iris_radius_frac_dev"],
        "iris_decentre_fx_lpmm": typical["iris_decentre_lpmm"],
        "hv_shift_um": typical["hv_shift_um"],
        "axicon_decentre_mm": typical["axicon_decentre_mm"],
        "z_offset_mm": typical["z_offset_mm"],
    }
    tight: list[str] = []
    for tol in tolerances:
        param = str(tol["parameter"])
        if param == "phase_levels":
            # 8-bit is the typical panel; handled separately via min_passing_levels below.
            continue
        if param in typical_map and not bool(tol["all_cases_pass"]):
            if float(tol["max_passing_deviation"]) < float(typical_map[param]):
                tight.append(param)
    quant_rows = [t for t in tolerances if str(t["parameter"]) == "phase_levels"]
    eight_bit_ok = True
    if quant_rows and not bool(quant_rows[0]["all_cases_pass"]):
        eight_bit_ok = float(quant_rows[0].get("min_passing_levels", 256)) <= 256

    mild = next((c for c in combined_cases if "mild" in str(c["label"])), None)
    moderate = next((c for c in combined_cases if "moderate" in str(c["label"])), None)
    uncompensated_lab_ok = bool(mild is not None and mild["passes"]) and bool(moderate is not None and moderate["passes"])
    attempted = list(compensated_cases)
    recovered = [c for c in attempted if bool(c["passes"])]
    compensation_reliable = bool(attempted) and len(recovered) == len(attempted)

    if not (clean_ok and clean_matches_reference):
        outcome = "M2S-E"
        statement = (
            "The degraded-bench model does not reproduce the clean M2N carrier/4F baseline, so realistic "
            "source-scale hardware cannot yet be represented without inconsistent grid/pixel/Fourier semantics."
        )
    elif (uncompensated_lab_ok and not tight) or (not tight and not attempted):
        outcome = "M2S-A"
        statement = (
            "The realistic degraded source-scale bench remains within tolerance (mild and moderate combined "
            "lab cases pass uncompensated, and every single-parameter tolerance is at or beyond the typical "
            "lab setting error). Source-scale lab implementation is plausible."
        )
    elif compensation_reliable and (uncompensated_lab_ok or len(tight) <= 2):
        outcome = "M2S-A"
        statement = (
            "Realistic degradations that exceed tolerance are reliably restored by the bounded "
            "precompensator with physically interpretable corrections. Source-scale lab implementation is "
            "plausible with the precompensation step."
        )
    elif attempted and not recovered:
        outcome = "M2S-C"
        statement = (
            "The source-scale route is highly sensitive to realistic errors and the bounded precompensator "
            "cannot recover the near-recoverable failures. Alternative hardware or encoding is needed."
        )
    elif len(tight) == 1:
        outcome = "M2S-D"
        statement = (
            f"A single component dominates the failure budget: {tight[0]} has a tolerance tighter than the "
            "typical lab setting error while every other parameter is comfortable."
        )
    else:
        outcome = "M2S-B"
        statement = (
            "The clean bench works, but several realistic error channels have tolerances tighter than typical "
            "lab setting errors; lab implementation needs careful active calibration (and the bounded "
            "precompensator where it helps)."
        )

    best_unc = max(
        (c for c in [*combined_cases] if not c["compensated"]),
        key=lambda c: float(c["comparison"]["z60_full_field_correlation"]),
        default=None,
    )
    best_comp = max(
        (c for c in compensated_cases),
        key=lambda c: float(c["comparison"]["z60_full_field_correlation"]),
        default=None,
    )

    def _summary(case: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if case is None:
            return None
        return {
            "label": str(case["label"]),
            "compensated": bool(case["compensated"]),
            "z60_full_field_correlation": float(case["comparison"]["z60_full_field_correlation"]),
            "strict_class": str(case["strict_gate"]["strict_class"]),
            "passes": bool(case["passes"]),
            "failure_mode": str(case["failure_mode"]),
        }

    return {
        "stage": MODE2S_STAGE,
        "suggested_outcome": outcome,
        "allowed_outcomes": MODE2S_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "scope": MODE2S_SCOPE_STATEMENT,
        "clean_baseline_passes": clean_ok,
        "clean_baseline_matches_m2n_reference": clean_matches_reference,
        "clean_baseline_correlation": float(clean_case["comparison"]["z60_full_field_correlation"]),
        "m2n_reference_correlation": float(clean_reference_correlation),
        "slm_window_fits_vertically": bool(fit_report["window_fits_vertically"]),
        "beam_power_clipped_by_active_area_fraction": float(fit_report["beam_power_clipped_by_active_area_fraction"]),
        "eight_bit_quantisation_ok": bool(eight_bit_ok),
        "tight_tolerance_parameters": tight,
        "typical_lab_errors_reference": dict(MODE2S_TYPICAL_LAB_ERRORS),
        "typical_lab_errors_note": "planning estimates only, not calibrated lab data",
        "mild_case_passes_uncompensated": bool(mild is not None and mild["passes"]),
        "moderate_case_passes_uncompensated": bool(moderate is not None and moderate["passes"]),
        "n_compensation_attempts": int(len(attempted)),
        "n_compensation_recovered": int(len(recovered)),
        "best_uncompensated": _summary(best_unc),
        "best_compensated": _summary(best_comp),
        "tolerances": [dict(t) for t in tolerances],
        "inherited_objective_sample_geometry_used": False,
        "microfabrication_sample_plane_claim": False,
        "micro_scale_note": (
            "MODE 2S is a source-scale tolerance result only; the microfabrication branch remains separate "
            "and blocked (MODE 1C/M1E)."
        ),
    }


def mode2s_scope_manifest(outcome: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Machine-readable MODE 2S simulation-scope manifest."""

    return {
        "stage": MODE2S_STAGE,
        "scope": MODE2S_SCOPE_STATEMENT,
        "git_commit": _mode1_git_commit_short(),
        "source_scale_only": True,
        "inherited_objective_sample_geometry": False,
        "micro_scale_sample_plane_simulated": False,
        "microfabrication_sample_plane_claim": False,
        "degradations_modelled": (
            "SLM active-aperture clip", "phase quantisation (8/10-bit, LUT gamma)", "pixel fill factor",
            "phase stroke", "H/V shift/rotation/magnification", "H/V amplitude ratio", "H/V piston",
            "QWP angle and retardance errors", "4F iris radius/decentre/shape", "axicon base angle/decentre/tilt/index",
            "observation z offset", "low-order Zernike input aberrations (common and differential)",
        ),
        "pixelation_note": (
            "the 8 um SLM pitch is finer than the source-scale simulation grid step, so per-pixel structure "
            "is not resolvable; the sampling audit is reported instead of a fake pixel model"
        ),
        "precompensation": "bounded, physically interpretable corrections only; no unconstrained pixel-level holograms",
        "suggested_outcome": None if outcome is None else outcome.get("suggested_outcome"),
        "claim_boundary": {
            "model_status": MODEL_STATUS,
            "final_export_allowed": FINAL_EXPORT_ALLOWED,
            "material_model": False,
            "camera_model": False,
            "judged_by_mode1c_m1e_microfabrication_constraints": False,
        },
    }


def run_mode2s_lab_realism(
    config: NathanSourceParityConfig | None = None,
    *,
    grid_n: int = 384,
    z_planes: int = 9,
    run_compensation: bool = True,
    compensation_maxiter: int = 40,
    max_compensated_cases: int = 3,
) -> dict[str, Any]:
    """Run the full MODE 2S study: baseline, Tier-1 sweeps, combined cases, precompensation."""

    data = mode2n_source_target(config, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    backward = run_mode2q_backward_initialisation(data)
    clean_reference = run_mode2n_dual_slm_4f_route(data, v0)
    clean_reference_corr = float(clean_reference.v0_comparison["z60_full_field_correlation"])

    clean_case = run_mode2s_degraded_forward(
        data, v0, backward, Mode2SPerturbation(label="clean_baseline"),
        sweep_parameter="clean", sweep_value=0.0,
    )
    aperture_case = run_mode2s_degraded_forward(
        data, v0, backward, Mode2SPerturbation(label="clean_plus_real_slm_aperture", slm_aperture_clip=True),
        sweep_parameter="slm_aperture_clip", sweep_value=1.0,
    )
    fit_report = mode2s_slm_aperture_fit_report(data)

    sweeps = mode2s_tier1_sweeps()
    sweep_cases: dict[str, list[dict[str, Any]]] = {}
    tolerances: list[dict[str, Any]] = []
    for name, spec in sweeps.items():
        rows: list[dict[str, Any]] = []
        for value, perturbation in spec["cases"]:
            rows.append(
                run_mode2s_degraded_forward(
                    data, v0, backward, perturbation,
                    sweep_parameter=str(spec["parameter"]), sweep_value=float(value),
                )
            )
        sweep_cases[name] = rows
        nominal = {"hv_amplitude_ratio": 1.0, "iris_radius_frac": MODE2N_DEFAULT_IRIS_RADIUS_FRAC}.get(str(spec["parameter"]), 0.0)
        wrap_period = TWOPI if str(spec["parameter"]) == "hv_piston_rad" else None
        tol = mode2s_tolerance_from_sweep(str(spec["parameter"]), rows, nominal_value=float(nominal), wrap_period=wrap_period)
        if name == "phase_quantisation":
            passing_levels = [float(r["sweep_value"]) for r in rows if bool(r["passes"]) and np.isfinite(float(r["sweep_value"]))]
            tol["min_passing_levels"] = float(min(passing_levels)) if passing_levels else float("inf")
        tolerances.append(tol)

    combined = [
        run_mode2s_degraded_forward(data, v0, backward, perturbation, sweep_parameter="combined", sweep_value=float(idx))
        for idx, perturbation in enumerate(mode2s_combined_cases())
    ]

    compensated: list[dict[str, Any]] = []
    compensation_meta: list[dict[str, Any]] = []
    if run_compensation:
        failing = [
            c for c in [*combined, *(r for rows in sweep_cases.values() for r in rows)]
            if not bool(c["passes"]) and float(c["comparison"]["z60_full_field_correlation"]) >= MODE2S_NEAR_RECOVERABLE_CORRELATION
        ]
        failing.sort(key=lambda c: float(c["comparison"]["z60_full_field_correlation"]), reverse=True)
        for case in failing[: max(0, int(max_compensated_cases))]:
            correction, meta = run_mode2s_precompensation(
                data, v0, backward, case["perturbation"], maxiter=int(compensation_maxiter),
            )
            comp_case = run_mode2s_degraded_forward(
                data, v0, backward, replace(case["perturbation"], label=f"{case['label']}_compensated"),
                correction=correction,
                sweep_parameter="compensated", sweep_value=float("nan"),
            )
            compensated.append(comp_case)
            compensation_meta.append({"label": str(case["label"]), **meta, "correction": correction.as_row()})

    outcome = mode2s_outcome_report(
        clean_case=clean_case,
        clean_reference_correlation=clean_reference_corr,
        fit_report=fit_report,
        tolerances=tolerances,
        combined_cases=combined,
        compensated_cases=compensated,
    )
    all_cases = [clean_case, aperture_case, *(r for rows in sweep_cases.values() for r in rows)]
    return {
        "data": data,
        "v0": v0,
        "backward": backward,
        "fit_report": fit_report,
        "clean_case": clean_case,
        "aperture_case": aperture_case,
        "clean_reference_correlation": clean_reference_corr,
        "sweep_cases": sweep_cases,
        "tolerances": tolerances,
        "combined_cases": combined,
        "compensated_cases": compensated,
        "compensation_meta": compensation_meta,
        "tier1_rows": tuple(mode2s_case_row(c) for c in all_cases),
        "combined_rows": tuple(mode2s_case_row(c) for c in combined),
        "compensated_rows": tuple(mode2s_case_row(c) for c in compensated),
        "outcome": outcome,
        "manifest": mode2s_scope_manifest(outcome),
    }


# ---------------------------------------------------------------------------
# MODE 2S figures (package functions; the notebook calls these, no duplicated physics)
# ---------------------------------------------------------------------------


def plot_mode2s_case_z60(
    case: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    crop_fraction: float = 0.35,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    grid = data["grid"]
    plane = np.asarray(case["reference_plane"], dtype=float)
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    crop, crop_grid = _mode1b_even_axis_crop(plane, grid, float(crop_fraction))
    xc = np.asarray(crop_grid["x"], dtype=float) / 1e-3
    ext_c = [float(xc[0]), float(xc[-1]), float(xc[0]), float(xc[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    axes[0].imshow(_normalise_image(plane, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1)
    axes[0].set_title(f"xy at z = {case['reference_z_m'] / 1e-3:.1f} mm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[1].imshow(_normalise_image(crop, local=True), origin="lower", extent=ext_c, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("central crop")
    axes[1].set_xlabel("x (mm)")
    corr = float(case["comparison"]["z60_full_field_correlation"])
    fig.suptitle(
        f"MODE 2S {case['label']}: {case['strict_gate']['strict_class']}, "
        f"V0 corr {corr:.4f}, passes={case['passes']}, failure={case['failure_mode']}"
    )
    _save_fig(fig, output_path)
    return fig, axes


def plot_mode2s_sweep(
    cases: Sequence[Mapping[str, Any]],
    *,
    parameter_label: str,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    rows = sorted(cases, key=lambda c: float(c["sweep_value"]))
    x = np.asarray([float(c["sweep_value"]) for c in rows], dtype=float)
    corr = np.asarray([float(c["comparison"]["z60_full_field_correlation"]) for c in rows], dtype=float)
    passes = np.asarray([bool(c["passes"]) for c in rows], dtype=bool)
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.plot(x, corr, color="0.4", lw=1.0, zorder=1)
    ax.scatter(x[passes], corr[passes], color="tab:green", label="strict pass", zorder=2)
    ax.scatter(x[~passes], corr[~passes], color="tab:red", label="fail", zorder=2)
    ax.axhline(MODE2S_PASS_CORRELATION, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel(parameter_label)
    ax.set_ylabel("z = 60 mm correlation to V0")
    ax.set_title(f"MODE 2S sweep: {parameter_label}")
    ax.legend(fontsize=8)
    _save_fig(fig, output_path)
    return fig, ax


def plot_mode2s_tolerance_summary(
    tolerances: Sequence[Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    rows = [t for t in tolerances if str(t["parameter"]) != "phase_levels"]
    labels = [str(t["parameter"]) for t in rows]
    values = [float(t["max_passing_deviation"]) for t in rows]
    all_pass = [bool(t["all_cases_pass"]) for t in rows]
    fig, ax = plt.subplots(figsize=(10.2, 4.6), constrained_layout=True)
    colors = ["tab:green" if ok else "tab:orange" for ok in all_pass]
    ax.bar(labels, values, color=colors)
    for idx, (value, ok) in enumerate(zip(values, all_pass, strict=True)):
        text = "all pass" if ok else f"{value:.3g}"
        ax.text(idx, value, text, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("max passing deviation (sweep units)")
    ax.set_title("MODE 2S single-parameter tolerance summary (green = entire swept range passes)")
    ax.tick_params(axis="x", rotation=25)
    _save_fig(fig, output_path)
    return fig, ax


def plot_mode2s_slm_fit(
    fit_report: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    w = float(fit_report["window_m"]) / 1e-3
    aw = float(fit_report["slm_active_width_m"]) / 1e-3
    ah = float(fit_report["slm_active_height_m"]) / 1e-3
    beam = float(fit_report["beam_1e_radius_m"]) / 1e-3
    ax.add_patch(Rectangle((-w / 2, -w / 2), w, w, fill=False, edgecolor="tab:blue", lw=1.4, label=f"10 mm source window"))
    ax.add_patch(Rectangle((-aw / 2, -ah / 2), aw, ah, fill=False, edgecolor="tab:red", lw=1.4, label=f"SLM active {aw:.2f} x {ah:.2f} mm"))
    ax.add_patch(Circle((0, 0), beam, fill=False, edgecolor="tab:green", lw=1.2, ls="--", label=f"beam 1/e radius {beam:.1f} mm"))
    ax.add_patch(Circle((0, 0), 2 * beam, fill=False, edgecolor="tab:green", lw=0.8, ls=":", label="2x beam radius"))
    ax.set_xlim(-9, 9)
    ax.set_ylim(-7, 7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    clip = float(fit_report["beam_power_clipped_by_active_area_fraction"])
    ax.set_title(
        f"MODE 2S SLM aperture fit: window fits vertically = {fit_report['window_fits_vertically']}, "
        f"beam power clipped {clip:.2e}"
    )
    ax.legend(fontsize=8, loc="upper right")
    _save_fig(fig, output_path)
    return fig, ax


def plot_mode2s_failure_examples(
    cases: Sequence[Mapping[str, Any]],
    data: Mapping[str, Any],
    *,
    max_examples: int = 4,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    import matplotlib.pyplot as plt

    failures = [c for c in cases if not bool(c["passes"])]
    failures.sort(key=lambda c: float(c["comparison"]["z60_full_field_correlation"]))
    chosen = failures[: max(1, int(max_examples))] if failures else list(cases)[:1]
    grid = data["grid"]
    x_mm = _mode2n_mm_axis(grid)
    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4), constrained_layout=True, squeeze=False)
    for ax, case in zip(axes[0], chosen, strict=True):
        plane = np.asarray(case["reference_plane"], dtype=float)
        crop, crop_grid = _mode1b_even_axis_crop(plane, grid, 0.35)
        xc = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext_c = [float(xc[0]), float(xc[-1]), float(xc[0]), float(xc[-1])]
        ax.imshow(_normalise_image(crop, local=True), origin="lower", extent=ext_c, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{case['label']}\n{case['failure_mode']}", fontsize=8)
        ax.set_xlabel("x (mm)")
    axes[0][0].set_ylabel("y (mm)")
    fig.suptitle("MODE 2S failure examples (central crops)")
    _save_fig(fig, output_path)
    return fig, axes


def write_mode2s_outputs(
    config: NathanSourceParityConfig | None = None,
    *,
    output_dir: str | Path = "outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance",
    report: Mapping[str, Any] | None = None,
    **run_kwargs: Any,
) -> dict[str, Path]:
    """Run (or reuse) the MODE 2S study and write all required artefacts."""

    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = dict(report) if report is not None else run_mode2s_lab_realism(config, **run_kwargs)
    data = study["data"]

    paths: dict[str, Path] = {
        "single_parameter_tolerances_csv": out / "mode2s_single_parameter_tolerances.csv",
        "single_parameter_tolerances_json": out / "mode2s_single_parameter_tolerances.json",
        "combined_cases_csv": out / "mode2s_combined_cases.csv",
        "combined_cases_json": out / "mode2s_combined_cases.json",
        "compensation_results_csv": out / "mode2s_compensation_results.csv",
        "compensation_results_json": out / "mode2s_compensation_results.json",
        "outcome_report": out / "mode2s_outcome_report.json",
        "scope_manifest": out / "simulation_scope_manifest.json",
    }
    _write_rows(paths["single_parameter_tolerances_csv"], study["tier1_rows"])
    paths["single_parameter_tolerances_json"].write_text(
        json.dumps(_json_ready({"tolerances": study["tolerances"], "cases": study["tier1_rows"]}), indent=2), encoding="utf-8",
    )
    _write_rows(paths["combined_cases_csv"], study["combined_rows"])
    paths["combined_cases_json"].write_text(json.dumps(_json_ready(study["combined_rows"]), indent=2), encoding="utf-8")
    comp_payload = {"cases": study["compensated_rows"], "optimiser": study["compensation_meta"]}
    if study["compensated_rows"]:
        _write_rows(paths["compensation_results_csv"], study["compensated_rows"])
    else:
        paths["compensation_results_csv"].write_text("no compensation runs were required\n", encoding="utf-8")
    paths["compensation_results_json"].write_text(json.dumps(_json_ready(comp_payload), indent=2), encoding="utf-8")
    paths["outcome_report"].write_text(json.dumps(_json_ready(dict(study["outcome"])), indent=2), encoding="utf-8")
    paths["scope_manifest"].write_text(json.dumps(_json_ready(dict(study["manifest"])), indent=2), encoding="utf-8")

    fig, _ = plot_mode2s_case_z60(study["clean_case"], data, output_path=out / "mode2s_clean_baseline.png")
    plt.close(fig)
    paths["clean_baseline"] = out / "mode2s_clean_baseline.png"
    fig, _ = plot_mode2s_slm_fit(study["fit_report"], data, output_path=out / "mode2s_slm_pixel_grid_fit.png")
    plt.close(fig)
    paths["slm_pixel_grid_fit"] = out / "mode2s_slm_pixel_grid_fit.png"
    fig, _ = plot_mode2s_tolerance_summary(study["tolerances"], output_path=out / "mode2s_tolerance_summary_plot.png")
    plt.close(fig)
    paths["tolerance_summary_plot"] = out / "mode2s_tolerance_summary_plot.png"

    sweep_figures = {
        "qwp_angle": ("qwp angle error (deg)", "mode2s_qwp_angle_sweep.png"),
        "hv_piston": ("H/V piston (rad)", "mode2s_hv_piston_sweep.png"),
        "iris_radius": ("iris radius (fraction of carrier)", "mode2s_iris_sweep.png"),
        "hv_shift": ("H/V lateral shift (um)", "mode2s_registration_sweep.png"),
        "axicon_decentre": ("axicon decentre (mm)", "mode2s_axicon_alignment_sweep.png"),
    }
    for key, (label, filename) in sweep_figures.items():
        fig, _ = plot_mode2s_sweep(study["sweep_cases"][key], parameter_label=label, output_path=out / filename)
        plt.close(fig)
        paths[f"sweep_{key}"] = out / filename

    uncompensated = [*study["combined_cases"]]
    best_unc = max(uncompensated, key=lambda c: float(c["comparison"]["z60_full_field_correlation"]))
    fig, _ = plot_mode2s_case_z60(best_unc, data, output_path=out / "mode2s_best_uncompensated_z60.png")
    plt.close(fig)
    paths["best_uncompensated_z60"] = out / "mode2s_best_uncompensated_z60.png"
    if study["compensated_cases"]:
        best_comp = max(study["compensated_cases"], key=lambda c: float(c["comparison"]["z60_full_field_correlation"]))
        fig, _ = plot_mode2s_case_z60(best_comp, data, output_path=out / "mode2s_best_compensated_z60.png")
        plt.close(fig)
    else:
        fig, _ = plot_mode2s_case_z60(best_unc, data, output_path=out / "mode2s_best_compensated_z60.png")
        plt.close(fig)
    paths["best_compensated_z60"] = out / "mode2s_best_compensated_z60.png"

    all_cases = [
        *study["combined_cases"],
        *(r for rows in study["sweep_cases"].values() for r in rows),
    ]
    fig, _ = plot_mode2s_failure_examples(all_cases, data, output_path=out / "mode2s_failure_examples.png")
    plt.close(fig)
    paths["failure_examples"] = out / "mode2s_failure_examples.png"
    return paths


__all__ = [
    "DEFAULT_FAST_N",
    "DEFAULT_FAST_Z_PLANES",
    "FINAL_EXPORT_ALLOWED",
    "DEFAULT_NATHAN_FIGURE4_REFERENCE",
    "DownstreamRouteResult",
    "FieldComparison",
    "FOCUS_VALIDATION_UNRESOLVED_STATEMENT",
    "HexagonPlaneMetrics",
    "MODEL_STATUS",
    "NathanHexagonConfig",
    "NathanMicroHexagonConfig",
    "NathanSourceParityConfig",
    "PatternedHWPConfig",
    "RouteComparisonReport",
    "RouteFieldResult",
    "RoutePropagationResult",
    "VISUAL_LADDER_STATUS",
    "V0_ALLOWED_VISUAL_VERDICTS",
    "VisualLadderReport",
    "VisualLadderStageResult",
    "MODE1_STAGE",
    "MODE1_ALLOWED_OUTCOMES",
    "MODE1_SYMMETRY_CLASSES",
    "Mode1Result",
    "run_mode1_ideal_p2_downstream",
    "mode1_symmetry_class",
    "mode1_hexagonal_bessel_survival_metrics",
    "mode1_centre_treatment_diagnostic",
    "mode1_completion_gate",
    "mode1_scope_manifest",
    "write_mode1_scope_manifest",
    "plot_mode1_p2_input_diagnostics",
    "plot_mode1_sample_region",
    "plot_mode1_f0_vs_f2",
    "plot_mode1_v0_to_mode1_scale",
    "MODE1B_STAGE",
    "MODE1B_ALLOWED_OUTCOMES",
    "Mode1BFeasibility",
    "Mode1BPlaneRadius",
    "Mode1BRingCountAudit",
    "Mode1BTargetTemplate",
    "Mode1BTemplateScore",
    "Mode1BCandidate",
    "Mode1BSearchResult",
    "effective_ring_count",
    "effective_ring_count_for_plane",
    "one_over_e_field_radius_from_intensity",
    "one_over_e_field_radius_from_vector_field",
    "audit_mode1b_ring_count_planes",
    "write_mode1b_ring_count_audit",
    "mode1b_feasibility",
    "angular_profile_on_ring",
    "circular_profile_correlation",
    "build_mode1b_target_template",
    "compare_to_v0_template",
    "mode1b_candidate_passes_hexagon_gate",
    "run_mode1b_candidate",
    "mode1b_candidate_row",
    "run_mode1b_geometry_search",
    "mode1b_completion_gate",
    "mode1b_parameter_search",
    "mode1b_scope_manifest",
    "plot_mode1b_target_template",
    "plot_mode1b_candidate",
    "plot_mode1b_current_inherited_failure",
    "write_mode1b_geometry_search_outputs",
    "MODE1C_STAGE",
    "MODE1C_ALLOWED_OUTCOMES",
    "Mode1CKMapping",
    "Mode1CApertureRingLimit",
    "Mode1CConstrainedCandidate",
    "audit_mode1c_kr_mapping",
    "audit_mode1c_aperture_ring_limit",
    "make_mode1c_twin_with_axicon_base_angle",
    "make_mode1c_twin_with_target_kr",
    "mode1c_candidate_row",
    "run_mode1c_constrained_search",
    "mode1c_outcome_report",
    "plot_mode1c_feasibility",
    "write_mode1c_outputs",
    "MODE1D_STAGE",
    "MODE1D_ALLOWED_OUTCOMES",
    "MODE1D_DEFAULT_RING_SWEEP",
    "MODE1D_ACCEPTABLE_TEMPLATE_CORRELATION",
    "MODE1D_ACCEPTABLE_XY_CORRELATION",
    "MODE1D_SCOPE_STATEMENT",
    "Mode1DDesignContext",
    "Mode1DSourceSweepCase",
    "required_pre_kr_for_ring_count",
    "required_surface_kr",
    "required_na",
    "required_radius_for_ring_count",
    "required_mapping_for_ring_count",
    "mode1d_design_context",
    "mode1d_required_na_table",
    "mode1d_required_radius_table",
    "mode1d_required_mapping_table",
    "mode1d_achievable_ring_count_table",
    "run_mode1d_source_ring_count_case",
    "mode1d_source_sweep_row",
    "run_mode1d_source_ring_count_sweep",
    "mode1d_minimum_accepted_ring_count",
    "mode1d_outcome_report",
    "mode1d_scope_manifest",
    "run_mode1d_inverse_redesign",
    "plot_mode1d_source_ring_count_sweep",
    "plot_mode1d_redesign_budget",
    "plot_mode1d_source_case",
    "write_mode1d_inverse_redesign_outputs",
    "MODE1E_STAGE",
    "MODE1E_ALLOWED_OUTCOMES",
    "MODE1E_PRIMARY_TEMPLATE_RING_COUNT",
    "MODE1E_REFERENCE_TEMPLATE_RING_COUNT",
    "MODE1E_DEFAULT_RING_COUNTS",
    "MODE1E_DEFAULT_P2_RADIUS_FACTORS",
    "MODE1E_KR_MATCH_RTOL",
    "MODE1E_F0_F2_CONSISTENCY_MIN",
    "MODE1E_SCOPE_STATEMENT",
    "Mode1EDesignCandidate",
    "Mode1ESourceTemplate",
    "Mode1ECandidateResult",
    "mode1e_required_pre_kr",
    "mode1e_surface_kr_from_mapping",
    "mode1e_na_required",
    "mode1e_base_angle_from_pre_kr",
    "make_mode1e_redesigned_config",
    "build_mode1e_source_template",
    "mode1e_template_gate",
    "run_mode1e_candidate",
    "add_mode1e_f2_reference",
    "run_mode1e_current_inherited_control",
    "mode1e_candidate_row",
    "mode1e_outcome_report",
    "mode1e_scope_manifest",
    "run_mode1e_redesigned_downstream",
    "plot_mode1e_source_template",
    "plot_mode1e_candidate_planes",
    "plot_mode1e_template_comparison",
    "write_mode1e_outputs",
    "MODE2N_STAGE",
    "MODE2N_ALLOWED_OUTCOMES",
    "MODE2N_PRE_AXICON_OVERLAP_PASS",
    "MODE2N_IDEAL_PROPAGATED_CORRELATION_PASS",
    "MODE2N_REALISTIC_PROPAGATED_CORRELATION_PASS",
    "MODE2N_DEFAULT_CARRIER_LPMM",
    "MODE2N_DEFAULT_IRIS_RADIUS_FRAC",
    "MODE2N_SCOPE_STATEMENT",
    "Mode2NRouteResult",
    "mode2n_source_target",
    "mode2n_propagate_through_source_axicon",
    "mode2n_compare_stacks_to_v0",
    "mode2n_route_passes_v0_match",
    "run_mode2n_v0_reference",
    "run_mode2n_patterned_hwp_route",
    "run_mode2n_dual_slm_qwp_route",
    "run_mode2n_dual_slm_4f_route",
    "mode2n_route_metric_row",
    "mode2n_outcome_report",
    "mode2n_scope_manifest",
    "run_mode2n_source_replication",
    "plot_mode2n_pre_axicon",
    "plot_mode2n_z60",
    "plot_mode2n_xz_comparison",
    "write_mode2n_outputs",
    "MODE2Q_STAGE",
    "MODE2Q_ALLOWED_OUTCOMES",
    "MODE2Q_BACKWARD_RECOVERY_OVERLAP_PASS",
    "MODE2Q_FORWARD_DIRECT_CORRELATION_PASS",
    "MODE2Q_FORWARD_4F_CORRELATION_PASS",
    "MODE2Q_C120_MINUS_C60_TOL",
    "MODE2Q_SCOPE_STATEMENT",
    "Mode2QBackwardField",
    "mode2q_forward_propagate_vector",
    "mode2q_backpropagate_vector",
    "mode2q_inverse_axicon",
    "mode2q_inverse_retarder",
    "mode2q_4f_passband_report",
    "mode2q_v0_complex_target",
    "run_mode2q_backward_initialisation",
    "mode2q_strict_hexagon_gate",
    "run_mode2q_forward_candidate",
    "mode2q_candidate_row",
    "mode2q_outcome_report",
    "mode2q_scope_manifest",
    "run_mode2q_lowdim_optimisation",
    "run_mode2q_backward_mask_synthesis",
    "plot_mode2q_backward_vs_raw",
    "plot_mode2q_required_hv",
    "plot_mode2q_masks",
    "plot_mode2q_candidate_z60",
    "plot_mode2q_zstack_summary",
    "write_mode2q_outputs",
    "MODE2S_STAGE",
    "MODE2S_ALLOWED_OUTCOMES",
    "MODE2S_PASS_CORRELATION",
    "MODE2S_NEAR_RECOVERABLE_CORRELATION",
    "MODE2S_FAILURE_MODES",
    "MODE2S_TYPICAL_LAB_ERRORS",
    "MODE2S_SCOPE_STATEMENT",
    "Mode2SPerturbation",
    "Mode2SCorrection",
    "mode2s_slm_aperture_fit_report",
    "mode2s_quantise_phase",
    "mode2s_zernike_phase",
    "mode2s_apply_4f",
    "mode2s_failure_mode",
    "mode2s_case_row",
    "mode2s_tier1_sweeps",
    "mode2s_combined_cases",
    "mode2s_tolerance_from_sweep",
    "run_mode2s_degraded_forward",
    "run_mode2s_precompensation",
    "mode2s_outcome_report",
    "mode2s_scope_manifest",
    "run_mode2s_lab_realism",
    "plot_mode2s_case_z60",
    "plot_mode2s_sweep",
    "plot_mode2s_tolerance_summary",
    "plot_mode2s_slm_fit",
    "plot_mode2s_failure_examples",
    "write_mode2s_outputs",
    "MODE2P_STAGE",
    "MODE2P_ALLOWED_OUTCOMES",
    "MODE2P_ACCEPTANCE_OVERLAP",
    "MODE2P_SCOPE_STATEMENT",
    "JonesChainResult",
    "nathan_alpha_map",
    "rot",
    "linear_retarder",
    "hwp",
    "qwp",
    "apply_jones_map",
    "apply_uniform_jones",
    "apply_hwp_map_to_horizontal",
    "wrap_2pi",
    "mode2p_target_arrays",
    "complex_vector_inner",
    "complex_vector_overlap",
    "best_global_phase",
    "phase_aligned_rms",
    "jones_stokes_rms",
    "alpha_angle_rms_mod_pi",
    "jones_power_ratio",
    "jones_metric_row",
    "synthesize_with_patterned_hwp",
    "route_patterned_hwp_ideal",
    "synthesize_from_circular_components",
    "route_dual_slm_circular_ideal",
    "route_dual_slm_linear_then_qwp_ideal",
    "mode2p_centre_treatment_report",
    "mode2p_outcome_report",
    "mode2p_scope_manifest",
    "run_mode2p_jones_synthesis",
    "plot_mode2p_target_alpha_and_sector_map",
    "plot_mode2p_route_vs_target",
    "write_mode2p_jones_synthesis_outputs",
    "air_z_values",
    "build_visual_reproduction_ladder_report",
    "build_control_suite",
    "build_downstream_focus_validation_gate",
    "build_downstream_model_comparison_gate",
    "build_default_routes",
    "build_route_comparison_report",
    "build_route_propagations",
    "canonical_target_diagnostics",
    "canonical_target_field",
    "compare_vector_fields",
    "default_nathan_grid",
    "digital_twin_plane_map_rows",
    "downstream_control_fields",
    "downstream_focus_multiscale_sampling_rows",
    "downstream_sampling_audit_rows",
    "hexagon_metrics_for_stack",
    "hollow_hexagon_score",
    "hwp_robustness_sweep",
    "inherited_parameter_rows",
    "lattice_control_report",
    "nathan_alpha",
    "nathan_specific_parameter_rows",
    "nathan_sector_mask",
    "nathan_literal_segmented_ra_input",
    "patterned_hwp_axis_map",
    "plot_route_xy_xz_profiles",
    "plot_visual_ladder_stage",
    "plot_v0_direct_visual_comparison",
    "plot_v0_field_views",
    "plot_v0_reference_literal_project_comparison",
    "propagate_route_through_shared_axicon",
    "run_v0_numerical_convergence",
    "run_v0_source_parity_visual_control",
    "run_v1_inherited_preobjective_visual_gate",
    "run_v2_full_sample_visual_gate",
    "route_xy_xz_profile_arrays",
    "route_conclusion",
    "run_patterned_hwp_route",
    "run_serial_slm_route",
    "serial_slm_robustness_sweep",
    "source_convention_validation_control",
    "source_parity_comparison",
    "source_parity_grid",
    "v0_numerical_resolution_status",
    "v0_source_output_parity_report",
    "v0_source_parameter_parity",
    "vector_config_from_existing_twin",
    "vectorial_pupil_spectrum_reference",
    "visual_ladder_stage_arrays",
    "write_report_csv",
]
