"""Data and governance layer for the Phase 2E report visual bible.

Phase 2E is presentation and diagnostic screening only. It reconstructs the
accepted Phase 2A/2B endpoints in memory, verifies them against their governed
tables, and writes exclusively to a new Phase 2E output root. The compact
parameter sweep is a separately labelled analytic-screening artifact; it does
not replace the fixed-bench or vector objective references.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import (
    _after_axicon,
    _prepare_projected_spectrum,
    _realistic_common_4f_field,
    build_mode2y_input_fields,
)
from vbb_study.digital_twin.phase2a_canonical import (
    PHASE2A_VARIANTS,
    _axicon_phase,
    _fourier_first_order,
    _normalised_power,
    _panel_from_manifest,
    _run_scalar_case,
    _variant_settings,
)
from vbb_study.digital_twin.phase2a_contracts import (
    PHASE2A_CANONICAL_SLM_MODEL,
    canonical_hardware_manifest,
    hardware_value,
)
from vbb_study.digital_twin.phase2b_visual_cases import (
    PHASE2B_HEX_REFERENCE_M,
    Phase2BCaseResult,
    Phase2BConfig,
    Phase2BHexPackage,
    _scalar_seed,
    build_hex_package,
    build_scalar_cases,
)
from vbb_study.digital_twin.phase2e_spectral_propagation import (
    DensePropagationMap,
    build_dense_spectral_propagation,
    map_correlation,
    on_axis_spectral_intensity,
)
from vbb_study.equations.fields import fft2c, ifft2c, make_xy_grid
from vbb_study.equations.propagation import scalable_angular_spectrum_propagate
from vbb_study.slm_model import apply_slm, slm_active_aperture


EPS = np.finfo(float).eps
PHASE2E_STAGE = "phase2e_report_visualisation_and_parameter_sweeps"
PHASE2E_OUTPUT_ROOT = Path("outputs/figures/phase2e_report_visualisation")
PHASE2E_DOC_PATH = Path("docs/94_phase2e_report_visualisation_and_parameter_sweeps.md")
PHASE2E_CASE_IDS = ("G0", "B0", "V1", "V3", "H1_CONTINUOUS", "H1_AVERAGED")
PHASE2E_3D_CASE_IDS = ("B0", "V1", "V3", "H1_CONTINUOUS", "H1_AVERAGED")
PHASE2E_UPSTREAM_FILES = (
    Path("outputs/validation/phase2a/canonical_hardware_manifest.json"),
    Path("outputs/validation/phase2a/canonical_case_summary.csv"),
    Path("outputs/validation/phase2a/canonical_power_ledgers.csv"),
    Path("outputs/validation/phase2a/slm_model_comparison.csv"),
    Path("outputs/figures/phase2b_visual_diagnostics/00_manifests/phase2b_final_manifest.json"),
    Path("outputs/figures/phase2b_visual_diagnostics/00_manifests/figure_provenance.csv"),
    Path("outputs/figures/phase2b_visual_diagnostics/08_summary_tables/phase2b_endpoint_reproduction_audit.csv"),
    Path("outputs/validation/phase2c/phase2c_case_summary.csv"),
    Path("outputs/validation/phase2c/phase2c_objective_benchmark.csv"),
    Path("outputs/validation/phase2c/phase2c_interface_benchmark.csv"),
    Path("outputs/validation/phase2c/phase2c_figure_manifest.json"),
    Path("outputs/figures/digital_twin/nathan_mode2y_continuous_vs_averaged/continuous_vs_averaged_summary.csv"),
    Path("outputs/figures/digital_twin/nathan_mode2z_orientation_interpolation/07_highN_confirmation/mode2z_highn_summary.csv"),
)


@dataclass(frozen=True)
class ReportFigureStyle:
    """Shared visual contract for every Phase 2E figure."""

    font_family: str = "DejaVu Sans"
    font_size: float = 9.2
    title_size: float = 10.4
    panel_label_size: float = 10.0
    line_width: float = 1.7
    intensity_cmap: str = "inferno"
    phase_cmap: str = "twilight"
    difference_cmap: str = "RdBu_r"
    signed_cmap: str = "coolwarm"
    axis_length_unit: str = "mm"
    intensity_label_linear: str = "normalised intensity (linear)"
    intensity_label_log: str = "normalised intensity (log10)"
    global_normalisation_label: str = "globally normalised"
    panel_normalisation_label: str = "per-panel normalised"
    scalar_focus_halfwidth_m: float = 0.25e-3
    h1_focus_halfwidth_m: float = 0.30e-3
    sweep_focus_halfwidth_m: float = 0.18e-3
    effective_na_sweep_halfwidth_m: float = 0.65e-3
    realism_focus_halfwidth_m: float = 0.80e-3
    scalar_display_resample_factor: int = 4
    h1_display_resample_factor: int = 3
    propagation_z_limits_m: tuple[float, float] = (0.0, 0.20)
    surface_z_limits: tuple[float, float] = (0.0, 1.0)
    surface_elevation_deg: float = 34.0
    surface_azimuth_deg: float = -58.0
    surface_topdown_elevation_deg: float = 90.0
    surface_topdown_azimuth_deg: float = -90.0
    surface_render_max_n: int = 360
    raster_dpi: int = 450
    panel_labels: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz")

    def validate(self) -> None:
        if self.axis_length_unit != "mm":
            raise ValueError("Phase 2E source-scale figures use millimetres consistently")
        if min(
            self.scalar_focus_halfwidth_m,
            self.h1_focus_halfwidth_m,
            self.sweep_focus_halfwidth_m,
            self.effective_na_sweep_halfwidth_m,
            self.realism_focus_halfwidth_m,
        ) <= 0.0:
            raise ValueError("focus crops must be positive")
        if self.surface_z_limits != (0.0, 1.0):
            raise ValueError("all Phase 2E intensity surfaces must use z in [0, 1]")
        if self.surface_render_max_n < 96:
            raise ValueError("3D display grid is too small for report use")
        if self.scalar_display_resample_factor < 1 or self.h1_display_resample_factor < 1:
            raise ValueError("display resampling factors must be positive integers")
        if self.raster_dpi < 300:
            raise ValueError("report raster output must be at least 300 dpi")

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class Phase2EConfig:
    """Numerical and presentation controls for the new, isolated phase."""

    phase2b: Phase2BConfig = field(default_factory=Phase2BConfig)
    style: ReportFigureStyle = field(default_factory=ReportFigureStyle)
    sweep_grid_n: int = 512
    sweep_window_m: float = 10.0e-3
    sweep_pad_factor: int = 2
    sweep_reference_z_m: float = PHASE2B_HEX_REFERENCE_M
    sweep_case_id: str = "V1"
    propagation_scalar_grid_n: int = 1024
    propagation_convergence_grid_n: int = 768
    propagation_transverse_samples: int = 1025
    propagation_z_samples: int = 601
    propagation_boundary_audit_grid_n: int = 768
    publication_quality: bool = True

    def validate(self) -> None:
        self.phase2b.validate()
        self.style.validate()
        if self.sweep_grid_n < 384:
            raise ValueError("Phase 2E sweep requires N>=384")
        if self.sweep_window_m <= 0.0 or self.sweep_pad_factor < 1:
            raise ValueError("invalid Phase 2E sweep sampling")
        if self.sweep_case_id != "V1":
            raise ValueError("the diagnostic sweep baseline is fixed to V1")
        if self.propagation_scalar_grid_n < 768:
            raise ValueError("publication propagation synthesis requires scalar N>=768")
        if not 512 < self.propagation_convergence_grid_n < self.propagation_scalar_grid_n:
            raise ValueError(
                "dense scalar convergence grid must lie between N=512 and the report grid"
            )
        if self.propagation_transverse_samples < 1025 or self.propagation_z_samples < 401:
            raise ValueError("publication propagation maps require dense physical coordinates")
        if self.propagation_boundary_audit_grid_n < 768:
            raise ValueError("pupil-boundary propagation audit requires N>=768")


@dataclass(frozen=True)
class ScalarVisualCase:
    case_id: str
    accepted_result: Phase2BCaseResult = field(repr=False, compare=False)
    input_field: np.ndarray = field(repr=False, compare=False)
    input_grid: Mapping[str, Any] = field(repr=False, compare=False)
    focus_field: np.ndarray = field(repr=False, compare=False)
    focus_grid: Mapping[str, Any] = field(repr=False, compare=False)
    propagation: DensePropagationMap = field(repr=False, compare=False)
    sas_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class IdealScalarVisualCase:
    """Smooth finite-energy scalar control used for headline beam physics."""

    case_id: str
    input_field: np.ndarray = field(repr=False, compare=False)
    input_grid: Mapping[str, Any] = field(repr=False, compare=False)
    focus_field: np.ndarray = field(repr=False, compare=False)
    focus_grid: Mapping[str, Any] = field(repr=False, compare=False)
    propagation: DensePropagationMap = field(repr=False, compare=False)
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class H1PolarisationCase:
    case_id: str
    input_orientation_rad: np.ndarray = field(repr=False, compare=False)
    input_intensity: np.ndarray = field(repr=False, compare=False)
    input_grid: Mapping[str, Any] = field(repr=False, compare=False)
    ex: np.ndarray = field(repr=False, compare=False)
    ey: np.ndarray = field(repr=False, compare=False)
    ez: np.ndarray = field(repr=False, compare=False)
    stokes: Mapping[str, np.ndarray] = field(repr=False, compare=False)


@dataclass(frozen=True)
class SweepPlane:
    sweep_id: str
    parameter_name: str
    parameter_value: float
    parameter_unit: str
    display_label: str
    intensity: np.ndarray = field(repr=False, compare=False)
    grid: Mapping[str, Any] = field(repr=False, compare=False)
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class PropagationBoundaryAudit:
    """B0 controls that separate finite-pupil physics from propagation errors."""

    ideal_untruncated: DensePropagationMap = field(repr=False, compare=False)
    hard_pupil: DensePropagationMap = field(repr=False, compare=False)
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class Phase2EData:
    config: Phase2EConfig
    scalar_cases: Mapping[str, ScalarVisualCase] = field(repr=False, compare=False)
    ideal_scalar_cases: Mapping[str, IdealScalarVisualCase] = field(repr=False, compare=False)
    hex_package: Phase2BHexPackage = field(repr=False, compare=False)
    h1_polarisation: Mapping[str, H1PolarisationCase] = field(repr=False, compare=False)
    h1_propagation: Mapping[str, DensePropagationMap] = field(repr=False, compare=False)
    propagation_boundary_audit: PropagationBoundaryAudit = field(repr=False, compare=False)
    realism_cases: Mapping[str, Mapping[str, Any]] = field(repr=False, compare=False)
    sweep_planes: Mapping[str, tuple[SweepPlane, ...]] = field(repr=False, compare=False)
    endpoint_audit: tuple[Mapping[str, Any], ...]
    upstream_hashes: Mapping[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase2e_upstream_hashes() -> dict[str, str]:
    missing = [str(path) for path in PHASE2E_UPSTREAM_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing accepted Phase 2E source artifacts: {missing}")
    return {path.as_posix(): _sha256(path) for path in PHASE2E_UPSTREAM_FILES}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _complex_sas_focus(
    field: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    *,
    z_m: float,
    pad_factor: int,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    output, output_grid, metadata = scalable_angular_spectrum_propagate(
        np.asarray(field, dtype=np.complex128),
        dict(grid),
        float(wavelength_m),
        float(z_m),
        n_medium=1.0,
        pad_factor=int(pad_factor),
        bandlimit=True,
        skip_final_phase=True,
        allow_invalid=False,
    )
    return np.asarray(output, dtype=np.complex128), output_grid, metadata


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.array(a, dtype=float, copy=True).ravel()
    bb = np.array(b, dtype=float, copy=True).ravel()
    aa -= float(np.mean(aa))
    bb -= float(np.mean(bb))
    return float(np.dot(aa, bb) / max(float(np.linalg.norm(aa) * np.linalg.norm(bb)), EPS))


def _sas_line_correlations(
    propagation: DensePropagationMap,
    sas_intensity: np.ndarray,
    sas_grid: Mapping[str, Any],
    *,
    z_m: float = PHASE2B_HEX_REFERENCE_M,
    comparison_halfwidth_m: float | None = None,
) -> dict[str, float]:
    """Compare fixed-coordinate spectral lines with the physical SAS endpoint."""

    plane = np.asarray(sas_intensity, dtype=float)
    x = np.asarray(sas_grid["x"], dtype=float)
    y = np.asarray(sas_grid.get("y", x), dtype=float)
    # Interpolate the orthogonal coordinate to the physical axis first. This
    # treats zero-straddled and axis-sampled grids identically.
    physical_x_axis = np.asarray(
        [np.interp(0.0, y, plane[:, column]) for column in range(plane.shape[1])]
    )
    physical_y_axis = np.asarray(
        [np.interp(0.0, x, plane[row]) for row in range(plane.shape[0])]
    )
    sas_x = np.interp(propagation.x_m, x, physical_x_axis)
    sas_y = np.interp(propagation.y_m, y, physical_y_axis)
    zi = int(np.argmin(np.abs(propagation.z_m - float(z_m))))
    x_mask = np.ones(propagation.x_m.shape, dtype=bool)
    y_mask = np.ones(propagation.y_m.shape, dtype=bool)
    if comparison_halfwidth_m is not None:
        x_mask = np.abs(propagation.x_m) <= float(comparison_halfwidth_m)
        y_mask = np.abs(propagation.y_m) <= float(comparison_halfwidth_m)
    return {
        "sas_reference_z_m": float(propagation.z_m[zi]),
        "sas_comparison_halfwidth_m": comparison_halfwidth_m,
        "sas_x_line_correlation": _safe_corr(
            propagation.xz_intensity[zi, x_mask], sas_x[x_mask]
        ),
        "sas_y_line_correlation": _safe_corr(
            propagation.yz_intensity[zi, y_mask], sas_y[y_mask]
        ),
    }


def _full_propagation_coordinates(
    grid: Mapping[str, Any],
    samples: int,
) -> np.ndarray:
    """Cover the complete source-grid window while retaining physical coordinates."""

    native = np.asarray(grid["x"], dtype=float)
    return np.linspace(float(native[0]), float(native[-1]), int(samples))


def _bessel_zone_metadata(
    radial_wavevector_m_inv: float,
    *,
    hard_pupil_active: bool = True,
) -> dict[str, Any]:
    manifest = canonical_hardware_manifest()
    wavelength_m = float(hardware_value(manifest, "wavelength_m"))
    beam_radius_m = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    pupil_radius_m = float(hardware_value(manifest, "objective_pupil_radius_m"))
    k0_m_inv = 2.0 * np.pi / wavelength_m
    cone_sine = abs(float(radial_wavevector_m_inv)) / k0_m_inv
    if cone_sine <= EPS:
        return {
            "axicon_transverse_wavevector_m_inv": 0.0,
            "axicon_cone_sine": 0.0,
            "pupil_radius_m": pupil_radius_m if hard_pupil_active else None,
            "beam_radius_m": beam_radius_m,
            "geometric_pupil_bessel_zone_m": None,
            "gaussian_radius_bessel_zone_m": None,
        }
    return {
        "axicon_transverse_wavevector_m_inv": float(radial_wavevector_m_inv),
        "axicon_cone_sine": float(cone_sine),
        "pupil_radius_m": pupil_radius_m if hard_pupil_active else None,
        "beam_radius_m": beam_radius_m,
        "geometric_pupil_bessel_zone_m": (
            float(pupil_radius_m / cone_sine) if hard_pupil_active else None
        ),
        "gaussian_radius_bessel_zone_m": float(beam_radius_m / cone_sine),
    }


def _canonical_axicon_kr_m_inv() -> float:
    manifest = canonical_hardware_manifest()
    wavelength_m = float(hardware_value(manifest, "wavelength_m"))
    base_angle_rad = np.deg2rad(
        float(hardware_value(manifest, "axicon_base_angle_deg"))
    )
    index_delta = float(hardware_value(manifest, "axicon_refractive_index")) - float(
        hardware_value(manifest, "axicon_external_medium_index")
    )
    return float(2.0 * np.pi / wavelength_m * index_delta * np.tan(base_angle_rad))


def _dense_scalar_map(
    config: Phase2EConfig,
    case_id: str,
    ell: int,
    wavelength_m: float,
) -> DensePropagationMap:
    z_values = np.linspace(
        config.style.propagation_z_limits_m[0],
        config.style.propagation_z_limits_m[1],
        int(config.propagation_z_samples),
    )
    high_field, high_grid, high_metadata = _scalar_seed(
        case_id, ell, grid_n=int(config.propagation_scalar_grid_n)
    )
    coordinates = _full_propagation_coordinates(
        high_grid,
        int(config.propagation_transverse_samples),
    )
    high = build_dense_spectral_propagation(
        grid=high_grid,
        wavelength_m=wavelength_m,
        z_values_m=z_values,
        transverse_coordinates_m=coordinates,
        scalar_field=high_field,
        source_label=f"{case_id} canonical source-scale field at N={config.propagation_scalar_grid_n}",
    )
    control_field, control_grid, _ = _scalar_seed(
        case_id,
        ell,
        grid_n=int(config.propagation_convergence_grid_n),
    )
    low = build_dense_spectral_propagation(
        grid=control_grid,
        wavelength_m=wavelength_m,
        z_values_m=z_values,
        transverse_coordinates_m=coordinates,
        scalar_field=control_field,
        source_label=f"{case_id} adjacent-grid convergence control at N={control_grid['N']}",
    )
    high_focus, high_focus_grid, _ = _complex_sas_focus(
        high_field,
        high_grid,
        wavelength_m,
        z_m=PHASE2B_HEX_REFERENCE_M,
        pad_factor=config.phase2b.sas_pad_factor,
    )
    return replace(
        high,
        metadata={
            **dict(high.metadata),
            **_bessel_zone_metadata(
                float(high_metadata["radial_wavevector_m_inv"])
            ),
            "convergence_control_grid_n": int(control_grid["N"]),
            **map_correlation(high, low),
            **_sas_line_correlations(
                high,
                np.abs(high_focus) ** 2,
                high_focus_grid,
                comparison_halfwidth_m=config.style.scalar_focus_halfwidth_m,
            ),
        },
    )


def _ideal_scalar_seed(
    case_id: str,
    ell: int,
    *,
    grid_n: int,
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any]]:
    """Build a smooth finite-energy Gaussian/axicon beam-family control."""

    manifest = canonical_hardware_manifest()
    wavelength_m = float(hardware_value(manifest, "wavelength_m"))
    beam_radius_m = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    grid = make_xy_grid(int(grid_n), 10.0e-3 / float(grid_n))
    radius = np.asarray(grid["R"], dtype=float)
    phase = np.exp(1j * float(ell) * np.asarray(grid["PHI"], dtype=float))
    field = np.exp(-(radius**2) / beam_radius_m**2) * phase
    radial_wavevector_m_inv = 0.0
    if case_id != "G0":
        axicon, radial_wavevector_m_inv = _axicon_phase(
            grid,
            manifest,
            _variant_settings("ideal_optical_route"),
        )
        field = field * axicon
    return np.asarray(field, dtype=np.complex128), grid, {
        "wavelength_m": wavelength_m,
        "beam_radius_m": beam_radius_m,
        "radial_wavevector_m_inv": float(radial_wavevector_m_inv),
        "source_model": "smooth finite-energy Gaussian axicon control",
        "hard_pupil_active": False,
        "slm_or_4f_errors_active": False,
    }


def _build_ideal_scalar_visual_cases(
    config: Phase2EConfig,
) -> dict[str, IdealScalarVisualCase]:
    cases: dict[str, IdealScalarVisualCase] = {}
    z_values = np.linspace(
        config.style.propagation_z_limits_m[0],
        config.style.propagation_z_limits_m[1],
        int(config.propagation_z_samples),
    )
    for case_id, ell in (("G0", 0), ("B0", 0), ("V1", 1), ("V3", 3)):
        field, grid, metadata = _ideal_scalar_seed(
            case_id,
            ell,
            grid_n=int(config.phase2b.scalar_grid_n),
        )
        focus, focus_grid, sas = _complex_sas_focus(
            field,
            grid,
            float(metadata["wavelength_m"]),
            z_m=PHASE2B_HEX_REFERENCE_M,
            pad_factor=config.phase2b.sas_pad_factor,
        )
        propagation_field, propagation_grid, propagation_metadata = _ideal_scalar_seed(
            case_id,
            ell,
            grid_n=int(config.propagation_scalar_grid_n),
        )
        propagation = build_dense_spectral_propagation(
            grid=propagation_grid,
            wavelength_m=float(propagation_metadata["wavelength_m"]),
            z_values_m=z_values,
            transverse_coordinates_m=_full_propagation_coordinates(
                propagation_grid,
                int(config.propagation_transverse_samples),
            ),
            scalar_field=propagation_field,
            source_label=f"{case_id} smooth finite-energy Gaussian axicon control",
        )
        propagation = replace(
            propagation,
            metadata={
                **dict(propagation.metadata),
                **_bessel_zone_metadata(
                    float(propagation_metadata["radial_wavevector_m_inv"]),
                    hard_pupil_active=False,
                ),
                "display_role": "headline_beam_physics_control",
                "accepted_fixed_bench_replacement": False,
            },
        )
        cases[case_id] = IdealScalarVisualCase(
            case_id=case_id,
            input_field=field,
            input_grid=grid,
            focus_field=focus,
            focus_grid=focus_grid,
            propagation=propagation,
            metadata={**dict(metadata), "sas": dict(sas)},
        )
    return cases


def _build_scalar_visual_cases(
    config: Phase2EConfig,
    accepted: Mapping[str, Phase2BCaseResult],
) -> dict[str, ScalarVisualCase]:
    cases: dict[str, ScalarVisualCase] = {}
    for case_id, ell in (("G0", 0), ("B0", 0), ("V1", 1), ("V3", 3)):
        field, grid, metadata = _scalar_seed(case_id, ell, grid_n=config.phase2b.scalar_grid_n)
        focus, focus_grid, sas = _complex_sas_focus(
            field,
            grid,
            float(metadata["wavelength_m"]),
            z_m=PHASE2B_HEX_REFERENCE_M,
            pad_factor=config.phase2b.sas_pad_factor,
        )
        propagation = _dense_scalar_map(
            config,
            case_id,
            ell,
            float(metadata["wavelength_m"]),
        )
        accepted_sas_correlation = _sas_line_correlations(
            propagation,
            np.abs(focus) ** 2,
            focus_grid,
            comparison_halfwidth_m=config.style.scalar_focus_halfwidth_m,
        )
        propagation = replace(
            propagation,
            metadata={
                **dict(propagation.metadata),
                "accepted_grid_sas_x_line_correlation": accepted_sas_correlation[
                    "sas_x_line_correlation"
                ],
                "accepted_grid_sas_y_line_correlation": accepted_sas_correlation[
                    "sas_y_line_correlation"
                ],
            },
        )
        cases[case_id] = ScalarVisualCase(
            case_id=case_id,
            accepted_result=accepted[case_id],
            input_field=field,
            input_grid=grid,
            focus_field=focus,
            focus_grid=focus_grid,
            propagation=propagation,
            sas_metadata={
                **dict(sas),
                "input_source": "Phase 2A realistic_fixed_bench_route reconstructed in memory",
                "metric_use": "none; Phase 2B native arrays remain metric-bearing",
                "display_interpolation": "none; physical SAS resampling",
            },
        )
    return cases


def _fields_from_prepared(
    prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    z_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax, ay, az, kz = prepared
    transfer = np.exp(1j * kz * float(z_m))
    return ifft2c(ax * transfer), ifft2c(ay * transfer), ifft2c(az * transfer)


def _stokes(ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> dict[str, np.ndarray]:
    ex2 = np.abs(ex) ** 2
    ey2 = np.abs(ey) ** 2
    s0 = np.asarray(ex2 + ey2 + np.abs(ez) ** 2, dtype=float)
    s1 = np.asarray(ex2 - ey2, dtype=float)
    cross = ex * np.conj(ey)
    s2 = np.asarray(2.0 * np.real(cross), dtype=float)
    s3 = np.asarray(-2.0 * np.imag(cross), dtype=float)
    orientation = 0.5 * np.arctan2(s2, s1)
    return {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "orientation_rad": orientation}


def _build_h1_polarisation(
    package: Phase2BHexPackage,
    config: Phase2EConfig,
) -> tuple[dict[str, H1PolarisationCase], dict[str, DensePropagationMap]]:
    data = package.bench["data"]
    inputs = build_mode2y_input_fields(data)
    amplitude = np.asarray(data["A"], dtype=float)
    result: dict[str, H1PolarisationCase] = {}
    propagation: dict[str, DensePropagationMap] = {}
    z_values = np.linspace(
        config.style.propagation_z_limits_m[0],
        config.style.propagation_z_limits_m[1],
        int(config.propagation_z_samples),
    )
    for case_id, label, alpha in (
        ("H1_CONTINUOUS", "continuous", inputs.continuous_alpha_rad),
        ("H1_AVERAGED", "sector_averaged", inputs.averaged_alpha_rad),
    ):
        prefield, _ = _realistic_common_4f_field(
            amplitude,
            alpha,
            data,
            carrier_lpmm=6.25,
            iris_radius_frac=0.40,
        )
        after, _ = _after_axicon(prefield, data)
        prepared = _prepare_projected_spectrum(after)
        ex, ey, ez = _fields_from_prepared(prepared, PHASE2B_HEX_REFERENCE_M)
        coordinates = _full_propagation_coordinates(
            after.grid,
            int(config.propagation_transverse_samples),
        )
        dense = build_dense_spectral_propagation(
            grid=after.grid,
            wavelength_m=float(after.wavelength_m),
            z_values_m=z_values,
            transverse_coordinates_m=coordinates,
            projected_spectra=prepared[:3],
            kz_m_inv=prepared[3],
            n_medium=float(after.medium_index),
            source_label=f"{case_id} realistic common-4F projected vector spectrum",
        )
        highn = package.highn_hero
        highn_correlation = _sas_line_correlations(
            dense,
            np.asarray(highn["sas_planes"][label]),
            highn["sas_grids"][label],
            comparison_halfwidth_m=config.style.h1_focus_halfwidth_m,
        )
        dense = replace(
            dense,
            metadata={
                **dict(dense.metadata),
                **_bessel_zone_metadata(
                    _canonical_axicon_kr_m_inv(),
                    hard_pupil_active=False,
                ),
                "sas_line_validation_applicability": "not_applicable_to_projected_vector_route",
                "highn_cross_grid_sas_x_line_correlation": highn_correlation[
                    "sas_x_line_correlation"
                ],
                "highn_cross_grid_sas_y_line_correlation": highn_correlation[
                    "sas_y_line_correlation"
                ],
                "highn_endpoint_grid_n": int(highn["native_grid_n"]),
            },
        )
        propagation[case_id] = dense
        result[case_id] = H1PolarisationCase(
            case_id=case_id,
            input_orientation_rad=np.asarray(alpha, dtype=float),
            input_intensity=np.asarray(amplitude**2, dtype=float),
            input_grid=data["grid"],
            ex=np.asarray(ex),
            ey=np.asarray(ey),
            ez=np.asarray(ez),
            stokes=_stokes(ex, ey, ez),
        )
    return result, propagation


def _build_realism_cases(grid_n: int) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    manifest = canonical_hardware_manifest()
    accepted_rows = _read_csv(Path("outputs/validation/phase2a/canonical_case_summary.csv"))
    accepted = {
        (row["case_id"], row["route_variant"]): row
        for row in accepted_rows
        if row["case_id"] in {"B0", "V1", "V3"}
    }
    payloads: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    for case_id, ell in (("B0", 0), ("V1", 1), ("V3", 3)):
        for variant in (
            "ideal_optical_route",
            "realistic_fixed_bench_route",
            "mild_error_route",
            "deliberately_degraded_route",
        ):
            payload = _run_scalar_case(case_id, ell, variant, manifest, grid_n=int(grid_n))
            key = f"{case_id}:{variant}"
            payloads[key] = payload
            expected = accepted[(case_id, variant)]
            for metric in (
                "dominant_off_axis_ring_radius_m",
                "central_intensity_ratio",
                "peak_intensity_au",
                "propagation_power_drift_fraction",
            ):
                actual = float(payload["row"][metric])
                target = float(expected[metric])
                audit.append({
                    "source": "Phase 2A canonical_case_summary.csv",
                    "case_id": case_id,
                    "route_variant": variant,
                    "metric": metric,
                    "expected": target,
                    "actual": actual,
                    "absolute_difference": abs(actual - target),
                    "reproduced": bool(np.isclose(actual, target, rtol=2e-7, atol=2e-9)),
                })
    return payloads, audit


def _axial_ripple_metrics(
    intensity: np.ndarray,
    z_m: np.ndarray,
    *,
    z_min_m: float = 20.0e-3,
    z_max_m: float = 100.0e-3,
    smoothing_width_m: float = 15.0e-3,
) -> dict[str, float]:
    values = np.asarray(intensity, dtype=float)
    coordinates = np.asarray(z_m, dtype=float)
    normalised = values / max(float(np.max(values)), EPS)
    step = float(np.mean(np.diff(coordinates)))
    window = max(3, int(round(float(smoothing_width_m) / step)))
    if window % 2 == 0:
        window += 1
    smooth = np.convolve(normalised, np.ones(window) / float(window), mode="same")
    mask = (coordinates >= float(z_min_m)) & (coordinates <= float(z_max_m))
    residual = normalised[mask] - smooth[mask]
    return {
        "ripple_interval_z_min_m": float(z_min_m),
        "ripple_interval_z_max_m": float(z_max_m),
        "ripple_smoothing_width_m": float(window * step),
        "ripple_rms_normalised": float(np.sqrt(np.mean(residual**2))),
        "ripple_peak_to_peak_normalised": float(np.ptp(residual)),
    }


def _build_propagation_boundary_audit(
    config: Phase2EConfig,
    realistic: DensePropagationMap,
) -> PropagationBoundaryAudit:
    """Isolate the accepted hard-pupil effect from BL-ASM and sampling effects."""

    manifest = canonical_hardware_manifest()
    n = int(config.propagation_boundary_audit_grid_n)
    grid = make_xy_grid(n, 10.0e-3 / float(n))
    wavelength_m = float(hardware_value(manifest, "wavelength_m"))
    beam_radius_m = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    pupil_radius_m = float(hardware_value(manifest, "objective_pupil_radius_m"))
    settings = _variant_settings("realistic_fixed_bench_route")
    axicon, radial_wavevector_m_inv = _axicon_phase(grid, manifest, settings)
    zone = _bessel_zone_metadata(float(radial_wavevector_m_inv))
    radius = np.asarray(grid["R"], dtype=float)
    gaussian = np.exp(-(radius**2) / beam_radius_m**2)
    ideal_field = np.asarray(gaussian * axicon, dtype=np.complex128)
    hard_pupil_field = np.where(radius <= pupil_radius_m, ideal_field, 0.0)
    common = {
        "grid": grid,
        "wavelength_m": wavelength_m,
        "z_values_m": realistic.z_m,
        "transverse_coordinates_m": realistic.x_m,
    }
    ideal = build_dense_spectral_propagation(
        **common,
        scalar_field=ideal_field,
        source_label="ideal Gaussian axicon without objective-pupil clipping",
    )
    ideal = replace(ideal, metadata={**dict(ideal.metadata), **zone})
    hard = build_dense_spectral_propagation(
        **common,
        scalar_field=hard_pupil_field,
        source_label="ideal Gaussian axicon clipped by the canonical hard objective pupil",
    )
    hard = replace(hard, metadata={**dict(hard.metadata), **zone})
    zero_index = int(np.argmin(np.abs(realistic.x_m)))
    ideal_axis = 0.5 * (
        ideal.xz_intensity[:, zero_index] + ideal.yz_intensity[:, zero_index]
    )
    hard_axis = 0.5 * (
        hard.xz_intensity[:, zero_index] + hard.yz_intensity[:, zero_index]
    )
    realistic_axis = 0.5 * (
        realistic.xz_intensity[:, zero_index]
        + realistic.yz_intensity[:, zero_index]
    )
    unbandlimited_axis = on_axis_spectral_intensity(
        grid=grid,
        wavelength_m=wavelength_m,
        z_values_m=realistic.z_m,
        scalar_field=hard_pupil_field,
        bandlimit=False,
    )
    hard_normalised = hard_axis / max(float(np.max(hard_axis)), EPS)
    unbandlimited_normalised = unbandlimited_axis / max(
        float(np.max(unbandlimited_axis)), EPS
    )
    metrics = {
        "audit_grid_n": n,
        "audit_grid_dx_m": float(grid["dx"]),
        "wavelength_m": wavelength_m,
        **zone,
        "hard_pupil_power_fraction": float(
            np.sum(np.abs(hard_pupil_field) ** 2)
            / max(float(np.sum(np.abs(ideal_field) ** 2)), EPS)
        ),
        "ideal_untruncated": _axial_ripple_metrics(ideal_axis, realistic.z_m),
        "hard_pupil": _axial_ripple_metrics(hard_axis, realistic.z_m),
        "realistic_route": _axial_ripple_metrics(realistic_axis, realistic.z_m),
        "hard_pupil_to_realistic_map": map_correlation(hard, realistic),
        "hard_pupil_to_realistic_on_axis_correlation": _safe_corr(
            hard_axis, realistic_axis
        ),
        "bandlimited_to_unbandlimited_on_axis_correlation": _safe_corr(
            hard_axis, unbandlimited_axis
        ),
        "bandlimited_to_unbandlimited_max_abs_normalised_difference": float(
            np.max(np.abs(hard_normalised - unbandlimited_normalised))
        ),
        "diagnosis": (
            "axial modulation and post-zone flare are finite hard-pupil effects; "
            "they are not caused by the Matsushima band limit"
        ),
    }
    return PropagationBoundaryAudit(
        ideal_untruncated=ideal,
        hard_pupil=hard,
        metrics=metrics,
    )


def _radial_profile(intensity: np.ndarray, grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = np.hypot(X, Y)
    dr = min(abs(float(x[1] - x[0])), abs(float(y[1] - y[0])))
    bins = np.floor(radius.ravel() / max(dr, EPS)).astype(int)
    totals = np.bincount(bins, weights=values.ravel())
    counts = np.bincount(bins)
    profile = totals / np.maximum(counts, 1)
    radii = (np.arange(profile.size, dtype=float) + 0.5) * dr
    return radii, profile


def _profile_metrics(intensity: np.ndarray, grid: Mapping[str, Any]) -> dict[str, float]:
    radii, profile = _radial_profile(intensity, grid)
    peak = max(float(np.max(profile)), EPS)
    normalised = profile / peak
    peak_index = int(np.argmax(profile))
    centre = float(profile[0] / peak)
    after = normalised[peak_index + 2 :]
    side_lobe = float(np.max(after)) if after.size else 0.0
    return {
        "ring_radius_m": float(radii[peak_index]),
        "central_intensity_ratio": centre,
        "side_lobe_ratio": side_lobe,
        "peak_intensity_au": float(np.max(intensity)),
        "integrated_intensity_au": float(np.sum(intensity)),
    }


def phase2e_sweep_registry(
    *,
    baseline_kr_m_inv: float,
) -> tuple[dict[str, Any], ...]:
    return (
        {"sweep_id": "vortex_charge", "parameter_name": "ell", "unit": "1", "values": (0, 1, 2, 3, 5)},
        {
            "sweep_id": "radial_wavevector",
            "parameter_name": "k_r_m_inv",
            "unit": "rad/m",
            "values": tuple(float(baseline_kr_m_inv) * value for value in (0.50, 0.75, 1.0, 1.25, 1.50)),
        },
        {
            "sweep_id": "input_beam_radius",
            "parameter_name": "beam_radius_m",
            "unit": "m",
            "values": (0.60e-3, 1.20e-3, 2.0e-3, 3.20e-3, 4.40e-3),
        },
        {
            "sweep_id": "propagation_distance",
            "parameter_name": "z_m",
            "unit": "m",
            "values": (10e-3, 30e-3, 60e-3, 120e-3, 200e-3),
        },
        {
            "sweep_id": "aperture_radius",
            "parameter_name": "aperture_radius_m",
            "unit": "m",
            "values": (0.75e-3, 1.50e-3, 2.50e-3, 3.50e-3, 4.75e-3),
        },
        {
            "sweep_id": "effective_objective_na",
            "parameter_name": "effective_NA",
            "unit": "1",
            "values": (0.006, 0.010, 0.016, 0.025, 0.040),
        },
        {
            "sweep_id": "defocus_aberration",
            "parameter_name": "defocus_waves",
            "unit": "waves",
            "values": (0.0, 0.25, 0.50, 1.0, 1.5),
        },
    )


def phase2e_error_sweep_registry() -> tuple[dict[str, Any], ...]:
    """Source-scale perturbations applied at their physical optical planes."""

    waves = (0.0, 0.10, 0.25, 0.50, 1.0)
    return (
        {"sweep_id": "error_input_beam_decentre", "parameter_name": "input_decentre_m", "unit": "m", "values": (0.0, 50e-6, 100e-6, 200e-6, 400e-6), "plane": "input amplitude"},
        {"sweep_id": "error_input_beam_tilt", "parameter_name": "input_tilt_rad", "unit": "rad", "values": (0.0, 0.25e-3, 0.50e-3, 1.0e-3, 2.0e-3), "plane": "input phase"},
        {"sweep_id": "error_slm_phase", "parameter_name": "slm_phase_error_rms_rad", "unit": "rad", "values": (0.0, 0.05, 0.10, 0.25, 0.50), "plane": "SLM phase"},
        {"sweep_id": "error_fourier_iris_offset", "parameter_name": "iris_offset_fraction", "unit": "carrier fraction", "values": (0.0, 0.10, 0.25, 0.45, 0.65), "plane": "common 4F Fourier plane"},
        {"sweep_id": "error_pupil_decentre", "parameter_name": "pupil_offset_fraction", "unit": "pupil radius", "values": (0.0, 0.05, 0.10, 0.20, 0.35), "plane": "objective pupil"},
        {"sweep_id": "error_axicon_decentre", "parameter_name": "axicon_decentre_m", "unit": "m", "values": (0.0, 50e-6, 100e-6, 250e-6, 500e-6), "plane": "axicon"},
        {"sweep_id": "error_zernike_defocus", "parameter_name": "zernike_defocus_waves", "unit": "waves", "values": waves, "plane": "objective pupil"},
        {"sweep_id": "error_zernike_astigmatism", "parameter_name": "zernike_astigmatism_waves", "unit": "waves", "values": waves, "plane": "objective pupil"},
        {"sweep_id": "error_zernike_coma", "parameter_name": "zernike_coma_waves", "unit": "waves", "values": waves, "plane": "objective pupil"},
        {"sweep_id": "error_zernike_spherical", "parameter_name": "zernike_spherical_waves", "unit": "waves", "values": waves, "plane": "objective pupil"},
    )


def _sweep_label(sweep_id: str, value: float) -> str:
    if sweep_id == "vortex_charge":
        return f"ell={int(value)}"
    if sweep_id == "radial_wavevector":
        return f"k_r={value / 1e3:.1f} krad/m"
    if sweep_id in {"input_beam_radius", "aperture_radius"}:
        return f"{value / 1e-3:.2g} mm"
    if sweep_id == "propagation_distance":
        return f"z={value / 1e-3:.0f} mm"
    if sweep_id == "effective_objective_na":
        return f"NA_eff={value:.3f}"
    if sweep_id == "defocus_aberration":
        return f"{value:.2g} waves"
    if sweep_id in {"error_input_beam_decentre", "error_axicon_decentre"}:
        return f"{value / 1e-6:.0f} um"
    if sweep_id == "error_input_beam_tilt":
        return f"{value / 1e-3:.2g} mrad"
    if sweep_id in {"error_slm_phase", "error_fourier_iris_offset", "error_pupil_decentre"}:
        return f"{value:.2g}"
    if sweep_id.startswith("error_zernike_"):
        return f"{value:.2g} waves"
    return f"{value:.2g}"


def _build_sweep_planes(config: Phase2EConfig) -> dict[str, tuple[SweepPlane, ...]]:
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, "wavelength_m"))
    baseline_kr = float(
        next(
            row["radial_wavevector_m_inv"]
            for row in _read_csv(Path("outputs/validation/phase2a/canonical_case_summary.csv"))
            if row["case_id"] == "V1" and row["route_variant"] == "realistic_fixed_bench_route"
        )
    )
    n = int(config.sweep_grid_n)
    dx = float(config.sweep_window_m) / n
    grid = make_xy_grid(n, dx)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    radius = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    base = {
        "ell": 1.0,
        "k_r_m_inv": baseline_kr,
        "beam_radius_m": 2.0e-3,
        "z_m": float(config.sweep_reference_z_m),
        "aperture_radius_m": 4.0e-3,
        "effective_NA": None,
        "defocus_waves": 0.0,
    }
    results: dict[str, tuple[SweepPlane, ...]] = {}
    for spec in phase2e_sweep_registry(baseline_kr_m_inv=baseline_kr):
        planes: list[SweepPlane] = []
        for value in spec["values"]:
            parameters = dict(base)
            parameters[str(spec["parameter_name"])] = float(value)
            beam_radius = float(parameters["beam_radius_m"])
            aperture_radius = float(parameters["aperture_radius_m"])
            rho = radius / max(aperture_radius, EPS)
            amplitude = np.exp(-(radius**2) / beam_radius**2) * (radius <= aperture_radius)
            defocus = 2.0 * np.pi * float(parameters["defocus_waves"]) * (2.0 * rho**2 - 1.0)
            phase = (
                float(parameters["ell"]) * theta
                - float(parameters["k_r_m_inv"]) * radius
                + defocus
            )
            field0 = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)
            effective_na = parameters["effective_NA"]
            if effective_na is not None:
                k0 = 2.0 * np.pi / wavelength
                kt = 2.0 * np.pi * np.hypot(np.asarray(grid["FX"]), np.asarray(grid["FY"]))
                field0 = ifft2c(fft2c(field0) * (kt <= k0 * float(effective_na)))
            propagated, output_grid, sas = _complex_sas_focus(
                field0,
                grid,
                wavelength,
                z_m=float(parameters["z_m"]),
                pad_factor=int(config.sweep_pad_factor),
            )
            intensity = np.asarray(np.abs(propagated) ** 2, dtype=np.float32)
            samples_per_period = float((2.0 * np.pi / float(parameters["k_r_m_inv"])) / dx)
            metrics = {
                **_profile_metrics(intensity, output_grid),
                "sweep_id": spec["sweep_id"],
                "parameter_name": spec["parameter_name"],
                "parameter_value": float(value),
                "parameter_unit": spec["unit"],
                "input_grid_n": n,
                "input_dx_m": dx,
                "output_grid_n": int(output_grid["N"]),
                "output_dx_m": float(output_grid["dx"]),
                "samples_per_radial_period": samples_per_period,
                "nyquist_pass": bool(samples_per_period >= 2.0),
                "sas_valid": bool(sas["valid"]),
                "metrics_computed_on_native_sas_array": True,
                "display_interpolation_used_for_metrics": False,
                "maturity": "diagnostic_screening_only",
                "mapping_mode": "analytic_bessel_gauss_screening",
            }
            planes.append(SweepPlane(
                sweep_id=str(spec["sweep_id"]),
                parameter_name=str(spec["parameter_name"]),
                parameter_value=float(value),
                parameter_unit=str(spec["unit"]),
                display_label=_sweep_label(str(spec["sweep_id"]), float(value)),
                intensity=intensity,
                grid=output_grid,
                metrics=metrics,
                provenance={
                    "source_model": "analytic finite-energy vortex-Bessel screening field",
                    "baseline_case": config.sweep_case_id,
                    "baseline_parameters": base,
                    "actual_parameters": parameters,
                    "propagator": "scalable angular spectrum",
                    "accepted_result_replaced": False,
                    "claim_scope": "visual sensitivity and trend screening only",
                },
            ))
        results[str(spec["sweep_id"])] = tuple(planes)
    return results


def _physical_vortex_error_field(
    manifest: Mapping[str, Any],
    grid: Mapping[str, Any],
    controls: Mapping[str, float],
    *,
    ell: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build the canonical realistic V1 field with one physical perturbation."""

    wavelength = float(hardware_value(manifest, "wavelength_m"))
    beam_radius = float(hardware_value(manifest, "beam_radius_on_slm_m"))
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    bx = float(controls.get("input_decentre_m", 0.0))
    raw_input = np.exp(-((X - bx) ** 2 + Y**2) / beam_radius**2)
    tilt = float(controls.get("input_tilt_rad", 0.0))
    field0 = np.asarray(
        raw_input * np.exp(1j * (2.0 * np.pi / wavelength) * np.sin(tilt) * X),
        dtype=np.complex128,
    )
    panel = _panel_from_manifest(manifest)
    panel_aperture = slm_active_aperture(dict(grid), panel)
    input_capture = _normalised_power(np.where(panel_aperture, field0, 0.0)) / max(
        _normalised_power(field0), EPS
    )
    theta = np.arctan2(Y, X)
    radius_norm = np.hypot(X, Y) / max(2.0 * beam_radius, EPS)
    phase_error = float(controls.get("slm_phase_error_rms_rad", 0.0)) * (
        2.0 * radius_norm**2 - 1.0
    )
    slm1 = apply_slm(
        field0,
        float(ell) * theta + phase_error,
        dict(grid),
        panel,
        quantise_phase=True,
        apply_fill_factor=True,
        apply_carrier=False,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )
    slm2 = apply_slm(
        slm1.total,
        0.5 * phase_error,
        dict(grid),
        panel,
        quantise_phase=True,
        apply_fill_factor=True,
        apply_carrier=True,
        fill_factor_model=PHASE2A_CANONICAL_SLM_MODEL,
    )
    filtered, first_order = _fourier_first_order(
        slm2.total,
        dict(grid),
        float(hardware_value(manifest, "carrier_frequency_cpm")),
        float(hardware_value(manifest, "fourier_iris_radius_cpm")),
        float(controls.get("iris_offset_fraction", 0.0)),
    )

    pupil_radius = float(hardware_value(manifest, "objective_pupil_radius_m"))
    pupil_offset = float(controls.get("pupil_offset_fraction", 0.0)) * pupil_radius
    xp = (X - pupil_offset) / max(pupil_radius, EPS)
    yp = Y / max(pupil_radius, EPS)
    rho2 = xp**2 + yp**2
    pupil = rho2 <= 1.0
    pupil_before = _normalised_power(filtered)
    aberration_waves = (
        float(controls.get("zernike_defocus_waves", 0.0)) * (2.0 * rho2 - 1.0)
        + float(controls.get("zernike_astigmatism_waves", 0.0)) * (xp**2 - yp**2)
        + float(controls.get("zernike_coma_waves", 0.0)) * (3.0 * rho2 - 2.0) * xp
        + float(controls.get("zernike_spherical_waves", 0.0)) * (6.0 * rho2**2 - 6.0 * rho2 + 1.0)
    )
    pupil_field = np.where(
        pupil, filtered * np.exp(2j * np.pi * aberration_waves), 0.0
    )
    pupil_capture = _normalised_power(pupil_field) / max(pupil_before, EPS)
    settings = _variant_settings("realistic_fixed_bench_route")
    settings["axicon_decentre_m"] = (
        float(controls.get("axicon_decentre_m", 0.0)),
        0.0,
    )
    axicon, kr = _axicon_phase(dict(grid), manifest, settings)
    return np.asarray(pupil_field * axicon, dtype=np.complex128), {
        "wavelength_m": wavelength,
        "input_aperture_fraction": float(input_capture),
        "first_order_efficiency": float(first_order),
        "objective_pupil_fraction": float(pupil_capture),
        "radial_wavevector_m_inv": float(kr),
        "represented_power_fraction": float(input_capture * first_order * pupil_capture),
    }


def _centroid_on_grid(intensity: np.ndarray, grid: Mapping[str, Any]) -> tuple[float, float]:
    values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    total = max(float(np.sum(values)), EPS)
    return (
        float(np.sum(np.sum(values, axis=0) * x) / total),
        float(np.sum(np.sum(values, axis=1) * y) / total),
    )


def _normalised_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float) / max(float(np.max(a)), EPS)
    bb = np.asarray(b, dtype=float) / max(float(np.max(b)), EPS)
    return float(np.linalg.norm(aa - bb) / max(float(np.linalg.norm(aa)), EPS))


def _build_error_sweep_planes(config: Phase2EConfig) -> dict[str, tuple[SweepPlane, ...]]:
    manifest = canonical_hardware_manifest()
    n = int(config.sweep_grid_n)
    grid = make_xy_grid(n, float(config.sweep_window_m) / n)
    baseline_field, baseline_meta = _physical_vortex_error_field(manifest, grid, {})
    baseline_output, baseline_grid, baseline_sas = _complex_sas_focus(
        baseline_field,
        grid,
        baseline_meta["wavelength_m"],
        z_m=float(config.sweep_reference_z_m),
        pad_factor=int(config.sweep_pad_factor),
    )
    baseline_intensity = np.asarray(np.abs(baseline_output) ** 2, dtype=np.float32)
    baseline_centroid = _centroid_on_grid(baseline_intensity, baseline_grid)
    baseline_peak = max(float(np.max(baseline_intensity)), EPS)
    cache: dict[tuple[str, float], tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any], Mapping[str, float]]] = {}
    results: dict[str, tuple[SweepPlane, ...]] = {}
    for spec in phase2e_error_sweep_registry():
        planes: list[SweepPlane] = []
        for value in spec["values"]:
            if float(value) == 0.0:
                intensity = baseline_intensity
                output_grid = baseline_grid
                sas = baseline_sas
                field_meta = baseline_meta
            else:
                cache_key = (str(spec["parameter_name"]), float(value))
                if cache_key not in cache:
                    field, field_meta = _physical_vortex_error_field(
                        manifest,
                        grid,
                        {str(spec["parameter_name"]): float(value)},
                    )
                    output, output_grid, sas = _complex_sas_focus(
                        field,
                        grid,
                        field_meta["wavelength_m"],
                        z_m=float(config.sweep_reference_z_m),
                        pad_factor=int(config.sweep_pad_factor),
                    )
                    cache[cache_key] = (
                        np.asarray(np.abs(output) ** 2, dtype=np.float32),
                        output_grid,
                        sas,
                        field_meta,
                    )
                intensity, output_grid, sas, field_meta = cache[cache_key]
            centroid = _centroid_on_grid(intensity, output_grid)
            samples_per_period = float(
                (2.0 * np.pi / baseline_meta["radial_wavevector_m_inv"])
                / float(grid["dx"])
            )
            metrics = {
                **_profile_metrics(intensity, output_grid),
                "sweep_id": spec["sweep_id"],
                "parameter_name": spec["parameter_name"],
                "parameter_value": float(value),
                "parameter_unit": spec["unit"],
                "physical_plane": spec["plane"],
                "input_grid_n": n,
                "input_dx_m": float(grid["dx"]),
                "output_grid_n": int(output_grid["N"]),
                "output_dx_m": float(output_grid["dx"]),
                "samples_per_radial_period": samples_per_period,
                "nyquist_pass": bool(samples_per_period >= 2.0),
                "sas_valid": bool(sas["valid"]),
                "metrics_computed_on_native_sas_array": True,
                "display_interpolation_used_for_metrics": False,
                "maturity": "diagnostic_physical_error_sweep",
                "mapping_mode": "canonical_source_scale_realistic_route",
                "centroid_shift_m": float(np.hypot(centroid[0] - baseline_centroid[0], centroid[1] - baseline_centroid[1])),
                "morphology_correlation_to_baseline": _safe_corr(intensity, baseline_intensity),
                "morphology_relative_l2_to_baseline": _normalised_l2(baseline_intensity, intensity),
                "peak_relative_to_baseline": float(np.max(intensity) / baseline_peak),
                "represented_power_fraction": float(field_meta["represented_power_fraction"]),
            }
            planes.append(SweepPlane(
                sweep_id=str(spec["sweep_id"]),
                parameter_name=str(spec["parameter_name"]),
                parameter_value=float(value),
                parameter_unit=str(spec["unit"]),
                display_label=_sweep_label(str(spec["sweep_id"]), float(value)),
                intensity=intensity,
                grid=output_grid,
                metrics=metrics,
                provenance={
                    "source_model": "canonical source-scale dual-SLM/filter/pupil/axicon route",
                    "baseline_case": "V1 realistic_fixed_bench_route",
                    "sweep_kind": "physical_error",
                    "physical_plane": spec["plane"],
                    "actual_controls": {str(spec["parameter_name"]): float(value)},
                    "zernike_convention": "unnormalised pupil polynomials; coefficient in waves",
                    "propagator": "scalable angular spectrum",
                    "accepted_result_replaced": False,
                    "claim_scope": "source-scale one-at-a-time physical sensitivity diagnostic",
                },
            ))
        results[str(spec["sweep_id"])] = tuple(planes)
    return results


def build_phase2e_data(config: Phase2EConfig | None = None) -> Phase2EData:
    """Build the complete in-memory visual pack while guarding upstream hashes."""

    cfg = config or Phase2EConfig()
    cfg.validate()
    hashes_before = phase2e_upstream_hashes()
    scalar_results = build_scalar_cases(cfg.phase2b)
    scalar_visual = _build_scalar_visual_cases(cfg, scalar_results)
    ideal_scalar_visual = _build_ideal_scalar_visual_cases(cfg)
    propagation_boundary_audit = _build_propagation_boundary_audit(
        cfg,
        scalar_visual["B0"].propagation,
    )
    hex_package = build_hex_package(cfg.phase2b)
    h1_polarisation, h1_propagation = _build_h1_polarisation(hex_package, cfg)
    realism, scalar_audit = _build_realism_cases(cfg.phase2b.scalar_grid_n)
    endpoint_rows = list(scalar_audit)
    endpoint_rows.extend(dict(row) for row in hex_package.endpoint_audit)
    sweep_planes = _build_sweep_planes(cfg)
    sweep_planes.update(_build_error_sweep_planes(cfg))
    hashes_after = phase2e_upstream_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("accepted upstream artifact changed while Phase 2E was running")
    if not all(bool(row.get("reproduced", False)) for row in endpoint_rows):
        failures = [row for row in endpoint_rows if not bool(row.get("reproduced", False))]
        raise AssertionError(f"Phase 2E endpoint reproduction failed: {failures[:3]}")
    if not all(
        bool(plane.metrics["nyquist_pass"]) and bool(plane.metrics["sas_valid"])
        for planes in sweep_planes.values()
        for plane in planes
    ):
        raise AssertionError("Phase 2E diagnostic sweep failed sampling or SAS validity")
    return Phase2EData(
        config=cfg,
        scalar_cases=scalar_visual,
        ideal_scalar_cases=ideal_scalar_visual,
        hex_package=hex_package,
        h1_polarisation=h1_polarisation,
        h1_propagation=h1_propagation,
        propagation_boundary_audit=propagation_boundary_audit,
        realism_cases=realism,
        sweep_planes=sweep_planes,
        endpoint_audit=tuple(endpoint_rows),
        upstream_hashes=hashes_before,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


__all__ = [
    "PHASE2E_3D_CASE_IDS",
    "PHASE2E_CASE_IDS",
    "PHASE2E_DOC_PATH",
    "PHASE2E_OUTPUT_ROOT",
    "PHASE2E_STAGE",
    "PHASE2E_UPSTREAM_FILES",
    "H1PolarisationCase",
    "IdealScalarVisualCase",
    "Phase2EConfig",
    "Phase2EData",
    "PropagationBoundaryAudit",
    "ReportFigureStyle",
    "ScalarVisualCase",
    "SweepPlane",
    "build_phase2e_data",
    "json_ready",
    "phase2e_sweep_registry",
    "phase2e_upstream_hashes",
]
