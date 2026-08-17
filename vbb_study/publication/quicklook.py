"""Stage 8.7 quick-look simulator orchestration.

This module wraps the locked scalar engine for short diagnostic runs.  It does
not redefine beam physics; it chooses small, labelled grids, writes explicit
metadata, and routes all figures through the publication visual helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vbb_study.config import TwinConfig, fs as BT_FS, kHz as BT_KHZ, mm as BT_MM, nm as BT_NM, uJ as BT_UJ, um as BT_UM
from vbb_study.design import compute_design_from_config as _compute_design_from_config, default_config as _default_twin_config
from vbb_study.equations.fields import gaussian_amplitude, make_xy_grid
from vbb_study.equations.scalar_bessel import build_bessel_gauss_field_ideal
from vbb_study.facade import core as _bt
from vbb_study import vbb_sample_study, vbb_studies, vbb_style
from vbb_study.publication import figure_registry, visuals

QUICKLOOK_SCHEMA_VERSION = "8.7.0"
QUICKLOOK_CAVEAT = (
    "Stage 8.7 quick-look diagnostic: visual interpolation is display-only, "
    "metrics use raw sampled grids, and material response is a planning proxy."
)
VISUAL_SANITY_FAILED_TEXT = "visual sanity failed — do not use this as a beam prediction."
FIRST_ORDER_SANITY_FAILED_TEXT = "first-order/filter sanity failed; lab-realistic field is diagnostic only."
FLATTENED_PHASE_CONVENTION = (
    "Flattened pre-axicon phase means the pre-axicon azimuthal vortex term "
    "is removed before the axicon/blaze terms are applied; this is a diagnostic "
    "phase convention, not a new hardware calibration."
)

ALLOWED_COMPUTATIONAL_MODES = ("quick_preview", "balanced", "publication", "custom")

FOUR_CONDITION_PHASE_COMPARISON_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "no_vortex_standard_axicon",
        "label": "no vortex + standard axicon",
        "vortex_phase_on": False,
        "flatten_pre_axicon_phase": False,
        "effective_ell": 0,
        "caveat": "ell=0 reference; standard axicon phase.",
    },
    {
        "case_id": "no_vortex_flattened_pre_axicon",
        "label": "no vortex + flattened pre-axicon phase",
        "vortex_phase_on": False,
        "flatten_pre_axicon_phase": True,
        "effective_ell": 0,
        "caveat": "Flattening is a no-op when no vortex term is present.",
    },
    {
        "case_id": "vortex_standard_axicon",
        "label": "vortex + standard axicon",
        "vortex_phase_on": True,
        "flatten_pre_axicon_phase": False,
        "effective_ell": None,
        "caveat": "Nominal vortex term is retained before the axicon.",
    },
    {
        "case_id": "vortex_flattened_pre_axicon",
        "label": "vortex + flattened pre-axicon phase",
        "vortex_phase_on": True,
        "flatten_pre_axicon_phase": True,
        "effective_ell": 0,
        "caveat": FLATTENED_PHASE_CONVENTION,
    },
)


@dataclass(frozen=True)
class QuicklookConfig:
    """Editable Stage 8.7 quick-look configuration."""

    computational_mode: str = "balanced"
    save_outputs: bool = False
    run_label: str = "stage8_7_quicklook"
    preset: str = "fast"
    grid_size: int = 384
    axial_points: int = 41
    axial_range_um: float = 180.0
    crop_pixels: int = 256
    vortex_phase_on: bool = True
    vortex_charge: int = 3
    ell: int | None = None
    beam_family: str = "vortex_bessel"
    flatten_phase_before_axicon: bool = False
    include_axicon: bool = True
    include_vortex: bool = True
    cone_angle_deg: float | None = None
    kr_um_inv: float | None = None
    target_core_diameter_um: float = 3.0
    target_bessel_length_um: float = 150.0
    wavelength_nm: float = 1029.0
    pulse_energy_uJ: float = 10.0
    pulse_duration_fs: float = 260.0
    rep_rate_kHz: float = 100.0
    repetition_rate_kHz: float | None = None
    beam_radius_on_slm_mm: float = 2.0
    input_beam_radius_mm: float | None = None
    objective_na: float = 0.45
    objective_NA: float | None = None
    objective_f_eff_mm: float = 4.0
    relay_magnification: float = 1.0
    slm_pixel_pitch_um: float = 8.0
    slm_phase_bits: int = 8
    slm_fill_factor: float = 0.93
    blaze_period_px: int = 20
    first_order_filter_radius_lpmm: float = 2.5
    first_order_filter_radius: float | None = None
    material_name: str = "Cr:ZnSe"
    material_refractive_index: float = 2.44
    sample_refractive_index: float | None = None
    write_depth_um: float = 300.0
    sample_depth_um: float | None = None
    single_pulse_threshold_J_cm2: float = 2.0
    material_threshold_J_cm2: float | None = None
    incubation_exponent: float = 0.84
    pulse_count: int = 1
    scan_speed_mm_s: float = 1.0
    feature_width_um: float = 3.0
    axicon_index: float = 1.5
    include_blaze: bool = True
    include_quantization: bool = True
    include_fill_factor: bool = True
    include_active_aperture: bool = True
    include_first_order_isolation: bool = True
    correct_interface: bool = False
    interface_correction_mode: str = "uncorrected_interface"
    display_interpolation: bool = False
    display_interpolation_method: str = "nearest"
    plot_interpolation: str = "nearest"
    display_interpolation_only: bool = True
    save_dpi: int = 300
    shared_colour_scale: bool = True
    crop_to_feature: bool = True
    auto_center_crop: bool = True
    xy_crop_um: float = 30.0
    xz_crop_um: float = 30.0
    known_good_visual_preset: str = "balanced_vortex_ell3"
    visual_sanity_min_first_order_fraction: float = 0.01
    visual_sanity_max_beam_offset_um: float = 12.0
    run_four_condition_phase_comparison: bool = True
    four_condition_phase_path: str = "ideal"
    four_condition_phase_axial_points: int = 31
    run_four_condition_lab_realistic_if_sane: bool = True
    enable_parameter_delta_comparison: bool = False
    delta_vortex_charge: int = 5
    delta_target_core_diameter_um: float = 4.0

    def validate(self) -> None:
        if self.computational_mode not in ALLOWED_COMPUTATIONAL_MODES:
            raise ValueError(f"computational_mode must be one of {ALLOWED_COMPUTATIONAL_MODES!r}")
        if int(self.grid_size) < 96:
            raise ValueError("grid_size must be at least 96 for quick-look propagation.")
        if int(self.axial_points) < 3:
            raise ValueError("axial_points must be at least 3.")
        if int(self.four_condition_phase_axial_points) < 3:
            raise ValueError("four_condition_phase_axial_points must be at least 3.")
        if int(self.slm_phase_bits) < 1:
            raise ValueError("slm_phase_bits must be positive.")
        if str(self.interface_correction_mode) not in {"uncorrected_interface", "ideal_numerical_correction"}:
            raise ValueError("interface_correction_mode must be 'uncorrected_interface' or 'ideal_numerical_correction'.")
        if str(self.plot_interpolation) not in {"nearest", "bilinear", "bicubic"}:
            raise ValueError("plot_interpolation must be nearest, bilinear, or bicubic.")
        if int(self.pulse_count) < 1:
            raise ValueError("pulse_count must be positive.")
        if float(self.visual_sanity_min_first_order_fraction) < 0.0:
            raise ValueError("visual_sanity_min_first_order_fraction must be non-negative.")
        if float(self.visual_sanity_max_beam_offset_um) <= 0.0:
            raise ValueError("visual_sanity_max_beam_offset_um must be positive.")


@dataclass
class QuicklookPreview:
    """Live quick-look result with arrays, metrics, and caveats."""

    resolved_config: QuicklookConfig
    preview_kind: str
    metrics: dict[str, Any] = field(default_factory=dict)
    metrics_dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    qa_dataframe: pd.DataFrame = field(default_factory=pd.DataFrame)
    visual_sanity: dict[str, Any] = field(default_factory=dict)
    captions: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=lambda: [QUICKLOOK_CAVEAT])
    figure_paths: list[Path] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    slm_phase_rad: np.ndarray | None = None
    slm_x_mm: np.ndarray | None = None
    slm_y_mm: np.ndarray | None = None
    ideal_xy_intensity: np.ndarray | None = None
    ideal_x_um: np.ndarray | None = None
    ideal_y_um: np.ndarray | None = None
    ideal_xz_intensity: np.ndarray | None = None
    ideal_z_um: np.ndarray | None = None
    ideal_transverse_um: np.ndarray | None = None
    lab_xy_intensity: np.ndarray | None = None
    lab_xz_intensity: np.ndarray | None = None
    lab_z_um: np.ndarray | None = None
    lab_transverse_um: np.ndarray | None = None
    through_sample_xz_intensity: np.ndarray | None = None
    through_sample_z_um: np.ndarray | None = None
    through_sample_transverse_um: np.ndarray | None = None
    material_proxy_xy: np.ndarray | None = None
    material_x_um: np.ndarray | None = None
    material_y_um: np.ndarray | None = None


@dataclass
class QuicklookComparison:
    """Two-config quick-look comparison result."""

    config_a: QuicklookConfig
    config_b: QuicklookConfig
    preview_a: QuicklookPreview
    preview_b: QuicklookPreview
    metric_delta: pd.DataFrame
    qa_dataframe: pd.DataFrame
    caveats: list[str] = field(default_factory=lambda: [QUICKLOOK_CAVEAT])


@dataclass
class QuicklookSweep:
    """One-parameter quick-look sweep result."""

    base_config: QuicklookConfig
    parameter: str
    values: list[Any]
    previews: list[QuicklookPreview]
    summary: pd.DataFrame
    qa_dataframe: pd.DataFrame
    caveats: list[str] = field(default_factory=lambda: [QUICKLOOK_CAVEAT])


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    """Return the current runner run id or a standalone quick-look id."""

    return os.environ.get("STRUCTURED_BEAM_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _config_field_names() -> set[str]:
    return set(QuicklookConfig.__dataclass_fields__)


def resolve_config(config: QuicklookConfig) -> QuicklookConfig:
    """Resolve notebook-friendly aliases into the engine-facing config fields."""

    values = asdict(config)
    if values.get("ell") is not None:
        values["vortex_charge"] = int(values["ell"])
    values["ell"] = int(values["vortex_charge"])
    if values.get("repetition_rate_kHz") is not None:
        values["rep_rate_kHz"] = float(values["repetition_rate_kHz"])
    if values.get("input_beam_radius_mm") is not None:
        values["beam_radius_on_slm_mm"] = float(values["input_beam_radius_mm"])
    if values.get("objective_NA") is not None:
        values["objective_na"] = float(values["objective_NA"])
    if values.get("first_order_filter_radius") is not None:
        values["first_order_filter_radius_lpmm"] = float(values["first_order_filter_radius"])
    if values.get("sample_refractive_index") is not None:
        values["material_refractive_index"] = float(values["sample_refractive_index"])
    if values.get("sample_depth_um") is not None:
        values["write_depth_um"] = float(values["sample_depth_um"])
    if values.get("material_threshold_J_cm2") is not None:
        values["single_pulse_threshold_J_cm2"] = float(values["material_threshold_J_cm2"])
    if values.get("plot_interpolation"):
        values["display_interpolation_method"] = str(values["plot_interpolation"])
        values["display_interpolation"] = str(values["plot_interpolation"]) != "nearest"
    values["correct_interface"] = str(values.get("interface_correction_mode")) == "ideal_numerical_correction"
    cfg = QuicklookConfig(**values)
    cfg.validate()
    return cfg


def with_updates(config: QuicklookConfig, **updates: Any) -> QuicklookConfig:
    """Return a resolved copy of ``config`` with notebook-local edits applied."""

    allowed = _config_field_names()
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise KeyError(f"Unknown quicklook config field(s): {unknown}")
    return resolve_config(replace(config, **updates))


def default_quicklook_config(**updates: Any) -> QuicklookConfig:
    """Return an editable in-notebook quick-look config."""

    cfg = resolve_config(QuicklookConfig())
    return with_updates(cfg, **updates) if updates else cfg


def default_config() -> QuicklookConfig:
    """Backward-compatible alias for :func:`default_quicklook_config`."""

    return default_quicklook_config()


def default_config_path() -> Path:
    """Return the static editable JSON config path."""

    return Path(__file__).resolve().parents[2] / "config" / "quicklook_config.json"


def config_to_jsonable(config: QuicklookConfig) -> dict[str, Any]:
    """Return a JSON-safe config mapping with schema metadata."""

    return {
        "source_schema_version": QUICKLOOK_SCHEMA_VERSION,
        "allowed_computational_modes": list(ALLOWED_COMPUTATIONAL_MODES),
        "flattened_phase_convention": FLATTENED_PHASE_CONVENTION,
        "comparison_presets": {
            "four_condition_phase_comparison": list(FOUR_CONDITION_PHASE_COMPARISON_PRESETS),
        },
        "config": asdict(config),
    }


def load_quicklook_config(path: str | Path | None = None) -> QuicklookConfig:
    """Load a quick-look config JSON file, falling back to defaults."""

    cfg_path = default_config_path() if path is None else Path(path)
    if not cfg_path.exists():
        cfg = default_quicklook_config()
        cfg.validate()
        return cfg
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    values = payload.get("config", payload)
    cfg = QuicklookConfig(**{k: values[k] for k in asdict(default_quicklook_config()).keys() if k in values})
    return resolve_config(cfg)


def load_config(path: str | Path | None = None) -> QuicklookConfig:
    """Backward-compatible alias for :func:`load_quicklook_config`."""

    return load_quicklook_config(path)


def save_quicklook_config(config: QuicklookConfig, path: str | Path | None = None) -> Path:
    """Write one quick-look config JSON preset."""

    cfg = resolve_config(config)
    cfg_path = default_config_path() if path is None else Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config_to_jsonable(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cfg_path


def save_default_config(path: str | Path | None = None) -> Path:
    """Write the default quick-look config JSON."""

    cfg_path = default_config_path() if path is None else Path(path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config_to_jsonable(default_quicklook_config()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cfg_path


def _mode_preset(config: QuicklookConfig) -> str:
    if config.computational_mode == "balanced":
        return "balanced"
    if config.computational_mode == "publication":
        return "publication"
    return config.preset


def _device_downsample(config: QuicklookConfig) -> int:
    max_res = 1920
    return max(1, int(math.ceil(max_res / max(1, int(config.grid_size)))))


def make_twin_config(
    config: QuicklookConfig,
    *,
    effective_ell: int | None = None,
    nominal_ell: int | None = None,
) -> TwinConfig:
    """Build a locked-engine ``TwinConfig`` from quick-look settings."""

    config = resolve_config(config)
    preset = _mode_preset(config)
    base = _default_twin_config(preset)
    ell = int(config.vortex_charge if (config.vortex_phase_on and config.include_vortex and not config.flatten_phase_before_axicon) else 0)
    if effective_ell is not None:
        ell = int(effective_ell)
    nominal = int(config.vortex_charge if nominal_ell is None else nominal_ell)
    grid_size = int(config.grid_size)
    grid = replace(
        base.grid,
        N=grid_size,
        ideal_N=grid_size,
        device_downsample=max(_device_downsample(config), int(getattr(base.grid, "device_downsample", 1))),
        axial_range_m=float(config.axial_range_um) * BT_UM,
        axial_points=int(config.axial_points),
        crop_pixels=min(int(config.crop_pixels), grid_size),
        label=str(config.computational_mode),
    )
    laser = replace(
        base.laser,
        wavelength_m=float(config.wavelength_nm) * BT_NM,
        pulse_duration_s=float(config.pulse_duration_fs) * BT_FS,
        input_pulse_energy_J=float(config.pulse_energy_uJ) * BT_UJ,
        rep_rate_Hz=float(config.rep_rate_kHz) * BT_KHZ,
        beam_radius_on_slm_m=float(config.beam_radius_on_slm_mm) * BT_MM,
    )
    energy = replace(base.energy, pulse_energy_in_J=float(config.pulse_energy_uJ) * BT_UJ)
    target = replace(
        base.target,
        ell=ell,
        target_core_diameter_m=float(config.target_core_diameter_um) * BT_UM,
        target_bessel_length_m=float(config.target_bessel_length_um) * BT_UM,
        n_axicon=float(config.axicon_index),
    )
    objective = replace(
        base.objective,
        NA=float(config.objective_na),
        f_eff_m=float(config.objective_f_eff_mm) * BT_MM,
    )
    relay = replace(
        base.relay,
        magnification_to_sample=float(config.relay_magnification),
    )
    slm = replace(
        base.slm,
        pixel_pitch_m=float(config.slm_pixel_pitch_um) * BT_UM,
        phase_bits=int(config.slm_phase_bits),
        fill_factor=float(config.slm_fill_factor),
        blaze_period_px=int(config.blaze_period_px),
        first_order_filter_radius_lpmm=float(config.first_order_filter_radius_lpmm),
    )
    material = replace(
        base.material,
        name=str(config.material_name),
        refractive_index=float(config.material_refractive_index),
        write_depth_m=float(config.write_depth_um) * BT_UM,
        single_pulse_threshold_J_cm2=float(config.single_pulse_threshold_J_cm2),
        incubation_exponent=float(config.incubation_exponent),
        scan_speed_m_s=float(config.scan_speed_mm_s) * BT_MM,
        feature_width_m=float(config.feature_width_um) * BT_UM,
        static_or_scan="static" if int(config.pulse_count) > 1 else base.material.static_or_scan,
        n_static_pulses=max(1, int(config.pulse_count)),
    )
    return replace(
        base,
        laser=laser,
        energy=energy,
        target=target,
        objective=objective,
        relay=relay,
        slm=slm,
        material=material,
        grid=grid,
        include_blaze=bool(config.include_blaze),
        include_quantization=bool(config.include_quantization),
        include_fill_factor=bool(config.include_fill_factor),
        include_active_aperture=bool(config.include_active_aperture),
        include_first_order_isolation=bool(config.include_first_order_isolation),
        apply_interface=True,
        correct_interface=bool(config.correct_interface),
        study_kind="beam_to_surface",
        random_seed=12345 + nominal,
    )


def config_summary_frame(config: QuicklookConfig) -> pd.DataFrame:
    """Return a compact table of user-facing quick-look parameters."""

    rows = [
        ("computational_mode", config.computational_mode, "quick-look mode"),
        ("known_good_visual_preset", config.known_good_visual_preset, "default visual sanity preset"),
        ("grid_size", config.grid_size, "square numerical grid samples"),
        ("axial_points", config.axial_points, "air-path z samples"),
        ("four_condition_phase_axial_points", config.four_condition_phase_axial_points, "four-condition z samples"),
        ("vortex_phase_on", config.vortex_phase_on, "toggle helical phase"),
        ("vortex_charge", config.vortex_charge, "nominal topological charge"),
        ("target_core_diameter_um", config.target_core_diameter_um, "target core/equivalent J0 diameter"),
        ("target_bessel_length_um", config.target_bessel_length_um, "target Bessel length"),
        ("objective_na", config.objective_na, "objective NA"),
        ("blaze_period_px", config.blaze_period_px, "SLM blaze period controlling +1 carrier"),
        ("first_order_filter_radius_lpmm", config.first_order_filter_radius_lpmm, "first-order stop radius in lp/mm"),
        ("visual_sanity_min_first_order_fraction", config.visual_sanity_min_first_order_fraction, "minimum selected fraction guardrail"),
        ("write_depth_um", config.write_depth_um, "sample write depth"),
        ("display_interpolation", config.display_interpolation, "visual-only interpolation"),
        ("plot_interpolation", config.plot_interpolation, "display interpolation method only"),
        ("auto_center_crop", config.auto_center_crop, "display crop follows measured beam centre"),
        (
            "run_four_condition_phase_comparison",
            config.run_four_condition_phase_comparison,
            "four-condition phase-state comparison",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "meaning"])


def _air_z_values(config: QuicklookConfig, points: int | None = None) -> np.ndarray:
    z_end = max(float(config.axial_range_um), 1.35 * float(config.target_bessel_length_um)) * BT_UM
    return np.linspace(0.0, z_end, int(points or config.axial_points))


def _sample_z_values(config: QuicklookConfig) -> np.ndarray:
    z_end = max(float(config.write_depth_um), 1.25 * float(config.target_bessel_length_um)) * BT_UM
    return np.linspace(0.0, z_end, int(config.axial_points))


def _phase_from_config(twin: TwinConfig) -> dict[str, Any]:
    air = vbb_studies.beam_air_config(twin)
    design = _compute_design_from_config(air)
    field = _bt().build_realistic_slm_field(air, design)
    return {"design": design, "field": field, "air_config": air}


def _metrics_dataframe(metrics: Mapping[str, Any], *, case_id: str | None = None) -> pd.DataFrame:
    row = dict(metrics)
    if case_id is not None:
        row.setdefault("case_id", case_id)
    return pd.DataFrame([row])


def _qa_dataframe(metrics: Mapping[str, Any], config: QuicklookConfig, *, preview_kind: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "preview_kind": preview_kind,
                "computational_mode": config.computational_mode,
                "diagnostic_only": config.computational_mode == "quick_preview",
                "display_interpolation": config.plot_interpolation,
                "display_interpolation_only": bool(config.display_interpolation_only),
                "propagation_power_label": metrics.get("propagation_power_label", "not_applicable"),
                "visual_sanity_label": metrics.get("visual_sanity_label", "not_evaluated"),
                "visual_sanity_warning": metrics.get("visual_sanity_warning", ""),
                "first_order_sanity_label": metrics.get("first_order_sanity_label", "not_applicable"),
                "caveat": QUICKLOOK_CAVEAT,
            }
        ]
    )


def _axis_um_from_grid(grid: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(grid["x"], dtype=float) / BT_UM


def _grid_from_axis_um(axis_um: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(axis_um, dtype=float) * BT_UM
    dx = float(abs(x[1] - x[0])) if len(x) > 1 else BT_UM
    X, Y = np.meshgrid(x, x, indexing="xy")
    return {"N": len(x), "dx": dx, "x": x, "X": X, "Y": Y, "R": np.hypot(X, Y), "PHI": np.arctan2(Y, X)}


def _positive_max(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    arr = np.maximum(arr, 0.0)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(finite))


def centre_intensity_fraction(intensity: Any) -> float:
    """Return I(center) / I(max) for a raw intensity plane."""

    arr = np.asarray(intensity, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return float("nan")
    arr = np.maximum(arr, 0.0)
    peak = _positive_max(arr)
    if peak <= np.finfo(float).eps:
        return float("nan")
    return float(arr[arr.shape[0] // 2, arr.shape[1] // 2] / peak)


def _axis_for_intensity(n: int, axis_um: Sequence[float] | None = None) -> np.ndarray:
    if axis_um is not None:
        axis = np.asarray(axis_um, dtype=float)
        if axis.size == int(n):
            return axis
    return np.arange(int(n), dtype=float) - 0.5 * (int(n) - 1)


def _beam_centre_xy_um(
    intensity: Any,
    x_um: Sequence[float] | None = None,
    y_um: Sequence[float] | None = None,
) -> tuple[float, float]:
    arr = np.asarray(intensity, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return float("nan"), float("nan")
    weights = np.maximum(arr, 0.0)
    weights[~np.isfinite(weights)] = 0.0
    total = float(np.sum(weights))
    if total <= np.finfo(float).eps:
        return float("nan"), float("nan")
    x = _axis_for_intensity(arr.shape[1], x_um)
    y = _axis_for_intensity(arr.shape[0], y_um)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return float(np.sum(weights * X) / total), float(np.sum(weights * Y) / total)


def beam_centre_offset_um(
    intensity: Any,
    x_um: Sequence[float] | None = None,
    y_um: Sequence[float] | None = None,
) -> float:
    """Return intensity centroid offset from the display/physical origin."""

    cx, cy = _beam_centre_xy_um(intensity, x_um, y_um)
    if not np.isfinite(cx) or not np.isfinite(cy):
        return float("nan")
    return float(math.hypot(cx, cy))


def _radial_profile_for_intensity(
    intensity: Any,
    x_um: Sequence[float] | None = None,
    y_um: Sequence[float] | None = None,
    *,
    bins: int = 80,
    centre_xy_um: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(intensity, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    arr = np.maximum(arr, 0.0)
    arr[~np.isfinite(arr)] = 0.0
    x = _axis_for_intensity(arr.shape[1], x_um)
    y = _axis_for_intensity(arr.shape[0], y_um)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X - float(centre_xy_um[0]), Y - float(centre_xy_um[1]))
    r_max = float(np.nanmax(R)) if R.size else 0.0
    if r_max <= np.finfo(float).eps:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    edges = np.linspace(0.0, r_max, max(8, int(bins)) + 1)
    idx = np.digitize(R.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)
    sums = np.bincount(idx[valid], weights=arr.ravel()[valid], minlength=len(edges) - 1)
    counts = np.bincount(idx[valid], minlength=len(edges) - 1)
    profile = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    radii = 0.5 * (edges[:-1] + edges[1:])
    return radii, profile


def ringness_score(
    intensity: Any,
    x_um: Sequence[float] | None = None,
    y_um: Sequence[float] | None = None,
) -> float:
    """Return a simple annular-maximum score for a raw intensity plane."""

    radii, profile = _radial_profile_for_intensity(intensity, x_um, y_um)
    if profile.size < 6:
        return float("nan")
    peak = _positive_max(profile)
    if peak <= np.finfo(float).eps:
        return float("nan")
    centre = centre_intensity_fraction(intensity)
    if not np.isfinite(centre):
        centre = float(np.nanmean(profile[: max(1, min(3, profile.size // 10))])) / peak
    outer = float(np.nanmax(profile[max(3, profile.size // 20) :]))
    return float(max(0.0, outer / peak - centre))


def radial_symmetry_score(
    intensity: Any,
    x_um: Sequence[float] | None = None,
    y_um: Sequence[float] | None = None,
) -> float:
    """Return an approximate 0..1 circular-symmetry score."""

    arr = np.asarray(intensity, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return float("nan")
    arr = np.maximum(arr, 0.0)
    arr[~np.isfinite(arr)] = 0.0
    peak = _positive_max(arr)
    if peak <= np.finfo(float).eps:
        return float("nan")
    x = _axis_for_intensity(arr.shape[1], x_um)
    y = _axis_for_intensity(arr.shape[0], y_um)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    radii, profile = _radial_profile_for_intensity(arr, x, y)
    if radii.size < 4:
        return float("nan")
    reconstructed = np.interp(R.ravel(), radii, profile, left=profile[0], right=profile[-1]).reshape(arr.shape)
    rms = float(np.sqrt(np.mean((arr / peak - reconstructed / peak) ** 2)))
    spread = float(np.std(arr / peak)) + np.finfo(float).eps
    return float(np.clip(1.0 - rms / spread, 0.0, 1.0))


def expected_ring_radius_error(
    intensity: Any,
    x_um: Sequence[float] | None = None,
    y_um: Sequence[float] | None = None,
    expected_radius_um: float | None = None,
) -> dict[str, float]:
    """Measure the dominant radial maximum and compare it with an expected radius."""

    radii, profile = _radial_profile_for_intensity(intensity, x_um, y_um)
    if profile.size < 4 or _positive_max(profile) <= np.finfo(float).eps:
        measured = float("nan")
    else:
        start = max(1, min(4, profile.size // 20))
        measured = float(radii[start + int(np.nanargmax(profile[start:]))])
    expected = float(expected_radius_um) if expected_radius_um is not None and np.isfinite(float(expected_radius_um)) else float("nan")
    absolute = abs(measured - expected) if np.isfinite(measured) and np.isfinite(expected) else float("nan")
    relative = absolute / max(abs(expected), np.finfo(float).eps) if np.isfinite(absolute) and np.isfinite(expected) else float("nan")
    return {
        "measured_ring_radius_um": measured,
        "expected_ring_radius_um": expected,
        "ring_radius_abs_error_um": float(absolute),
        "ring_radius_rel_error": float(relative),
    }


def xz_structure_score(xz_intensity: Any) -> float:
    """Return a guardrail score for nontrivial axial structure in an XZ map."""

    arr = np.asarray(xz_intensity, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return float("nan")
    arr = np.maximum(arr, 0.0)
    arr[~np.isfinite(arr)] = 0.0
    peak = _positive_max(arr)
    if peak <= np.finfo(float).eps:
        return 0.0
    column_power = np.sum(arr, axis=0)
    active = column_power > (1.0e-6 * float(np.nanmax(column_power)))
    active_fraction = float(np.count_nonzero(active) / max(1, column_power.size))
    column_peak = np.max(arr, axis=0)
    axial_cv = float(np.std(column_peak) / (np.mean(column_peak) + np.finfo(float).eps))
    row_power = np.sum(arr, axis=1)
    transverse_active = float(np.count_nonzero(row_power > 1.0e-6 * float(np.nanmax(row_power))) / max(1, row_power.size))
    return float(np.clip(active_fraction * (0.5 * axial_cv + 0.5 * transverse_active), 0.0, 1.0))


def first_order_selection_sanity(
    metrics: Mapping[str, Any],
    *,
    min_fraction: float = 0.01,
) -> dict[str, Any]:
    """Check that a lab-realistic first-order isolation did not collapse."""

    fraction = float(metrics.get("first_order_selected_fraction", np.nan))
    geometry_valid = bool(metrics.get("first_order_geometry_valid", True))
    carrier = float(metrics.get("first_order_carrier_lpmm", np.nan))
    effective = float(metrics.get("first_order_effective_filter_radius_lpmm", metrics.get("first_order_filter_radius_lpmm", np.nan)))
    cone = float(metrics.get("first_order_axicon_cone_radius_lpmm", np.nan))
    reasons: list[str] = []
    if not geometry_valid:
        reasons.append("first-order carrier/filter geometry is invalid")
    if np.isfinite(fraction) and fraction < float(min_fraction):
        reasons.append(f"selected fraction {fraction:.4g} is below {float(min_fraction):.4g}")
    elif not np.isfinite(fraction):
        reasons.append("selected fraction is not finite")
    passed = not reasons
    return {
        "first_order_sanity_pass": bool(passed),
        "first_order_sanity_label": "pass" if passed else "fail",
        "first_order_sanity_warning": "" if passed else FIRST_ORDER_SANITY_FAILED_TEXT,
        "first_order_sanity_reason": "; ".join(reasons),
        "first_order_selected_fraction": fraction,
        "first_order_carrier_lpmm": carrier,
        "first_order_effective_filter_radius_lpmm": effective,
        "first_order_axicon_cone_radius_lpmm": cone,
    }


def beam_visual_sanity_metrics(
    *,
    xy_intensity: Any,
    x_um: Sequence[float] | None = None,
    y_um: Sequence[float] | None = None,
    xz_intensity: Any | None = None,
    metrics: Mapping[str, Any] | None = None,
    config: QuicklookConfig | None = None,
    ell: int | None = None,
    include_first_order: bool = False,
) -> dict[str, Any]:
    """Return raw-array visual sanity guardrails for quicklook plots."""

    metric_map = dict(metrics or {})
    cfg = resolve_config(config) if isinstance(config, QuicklookConfig) else default_quicklook_config()
    ell_i = int(metric_map.get("ell", cfg.vortex_charge if ell is None else ell))
    centre_fraction = centre_intensity_fraction(xy_intensity)
    ringness = ringness_score(xy_intensity, x_um, y_um)
    symmetry = radial_symmetry_score(xy_intensity, x_um, y_um)
    offset = beam_centre_offset_um(xy_intensity, x_um, y_um)
    expected_radius = float(metric_map.get("vortex_main_ring_radius_um", np.nan)) if ell_i > 0 else float(metric_map.get("core_first_zero_radius_um", np.nan))
    radius_report = expected_ring_radius_error(xy_intensity, x_um, y_um, expected_radius)
    xz_score = xz_structure_score(xz_intensity) if xz_intensity is not None else float("nan")
    xz_arr = np.asarray(xz_intensity, dtype=float) if xz_intensity is not None else np.asarray([], dtype=float)
    active_slices = 0
    if xz_arr.ndim == 2 and xz_arr.size:
        power = np.sum(np.maximum(xz_arr, 0.0), axis=0)
        active_slices = int(np.count_nonzero(power > 1.0e-6 * _positive_max(power)))

    reasons: list[str] = []
    if not np.isfinite(centre_fraction):
        reasons.append("centre intensity is not finite")
    elif ell_i == 0 and centre_fraction < 0.12:
        reasons.append("ell=0 Bessel-like case has a dark centre")
    elif ell_i != 0 and centre_fraction > 0.45:
        reasons.append("vortex case lacks centre suppression")
    if ell_i != 0:
        if not np.isfinite(ringness) or ringness <= 0.05:
            reasons.append("vortex ringness is too low")
        if not np.isfinite(radius_report["measured_ring_radius_um"]) or radius_report["measured_ring_radius_um"] <= 0.0:
            reasons.append("vortex ring radius is not finite and positive")
    if xz_intensity is not None:
        if active_slices <= 1:
            reasons.append("XZ map has fewer than two active axial slices")
        if not np.isfinite(xz_score) or xz_score <= 0.02:
            reasons.append("XZ structure score is too low")
    if np.isfinite(offset) and offset > float(cfg.visual_sanity_max_beam_offset_um):
        reasons.append(f"beam centre offset {offset:.3g} um exceeds limit")

    first_order_report: dict[str, Any] = {}
    if include_first_order:
        first_order_report = first_order_selection_sanity(
            metric_map,
            min_fraction=float(cfg.visual_sanity_min_first_order_fraction),
        )
        if not first_order_report["first_order_sanity_pass"]:
            reasons.append(str(first_order_report["first_order_sanity_reason"]))

    passed = not reasons
    return {
        "visual_sanity_pass": bool(passed),
        "visual_sanity_label": "pass" if passed else "fail",
        "visual_sanity_warning": "" if passed else VISUAL_SANITY_FAILED_TEXT,
        "visual_sanity_reason": "; ".join(reason for reason in reasons if reason),
        "centre_intensity_fraction": centre_fraction,
        "ringness_score": ringness,
        "radial_symmetry_score": symmetry,
        "beam_centre_offset_um": offset,
        "xz_structure_score": xz_score,
        "xz_active_slices": active_slices,
        **radius_report,
        **first_order_report,
    }


def _crop_xy_for_display(
    values: Any,
    grid: Mapping[str, Any],
    config: QuicklookConfig,
    *,
    centre_xy_um: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    axis = _axis_um_from_grid(grid)
    if not config.crop_to_feature or not np.isfinite(float(config.xy_crop_um)) or float(config.xy_crop_um) <= 0.0:
        return arr, axis, axis
    cx = float(centre_xy_um[0]) if centre_xy_um is not None and np.isfinite(float(centre_xy_um[0])) else 0.0
    cy = float(centre_xy_um[1]) if centre_xy_um is not None and np.isfinite(float(centre_xy_um[1])) else 0.0
    keep_x = np.abs(axis - cx) <= float(config.xy_crop_um)
    keep_y = np.abs(axis - cy) <= float(config.xy_crop_um)
    if np.count_nonzero(keep_x) < 8 or np.count_nonzero(keep_y) < 8:
        return arr, axis, axis
    return arr[np.ix_(keep_y, keep_x)], axis[keep_x], axis[keep_y]


def _crop_xz_for_display(
    values: Any,
    grid: Mapping[str, Any],
    config: QuicklookConfig,
    *,
    centre_x_um: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    transverse = _axis_um_from_grid(grid)
    if not config.crop_to_feature or not np.isfinite(float(config.xz_crop_um)) or float(config.xz_crop_um) <= 0.0:
        return arr, transverse
    cx = float(centre_x_um) if centre_x_um is not None and np.isfinite(float(centre_x_um)) else 0.0
    keep = np.abs(transverse - cx) <= float(config.xz_crop_um)
    if np.count_nonzero(keep) < 8:
        return arr, transverse
    return arr[keep, :], transverse[keep]


def _beam_preview_from_result(
    config: QuicklookConfig,
    result: Mapping[str, Any],
    *,
    preview_kind: str,
) -> QuicklookPreview:
    volume = result["volume"]
    raw_xy = np.asarray(volume["planes"]["peak"], dtype=float)
    raw_xz = np.asarray(volume["xz"], dtype=float)
    raw_axis_um = _axis_um_from_grid(volume["crop_grid"])
    z_um = np.asarray(volume["z"], dtype=float) / BT_UM
    metrics = dict(result.get("metrics", {}))
    sanity = beam_visual_sanity_metrics(
        xy_intensity=raw_xy,
        x_um=raw_axis_um,
        y_um=raw_axis_um,
        xz_intensity=raw_xz,
        metrics=metrics,
        config=config,
        include_first_order=preview_kind == "lab",
    )
    metrics.update(sanity)
    centre_xy = _beam_centre_xy_um(raw_xy, raw_axis_um, raw_axis_um) if bool(config.auto_center_crop) else None
    xy, x_um, y_um = _crop_xy_for_display(volume["planes"]["peak"], volume["crop_grid"], config, centre_xy_um=centre_xy)
    xz, transverse_um = _crop_xz_for_display(
        volume["xz"],
        volume["crop_grid"],
        config,
        centre_x_um=None if centre_xy is None else centre_xy[0],
    )
    caveats = [QUICKLOOK_CAVEAT]
    if not bool(sanity.get("visual_sanity_pass", False)):
        caveats.append(VISUAL_SANITY_FAILED_TEXT)
    if preview_kind == "lab" and not bool(sanity.get("first_order_sanity_pass", True)):
        caveats.append(FIRST_ORDER_SANITY_FAILED_TEXT)
    preview = QuicklookPreview(
        resolved_config=config,
        preview_kind=preview_kind,
        metrics=metrics,
        metrics_dataframe=_metrics_dataframe(metrics, case_id=metrics.get("case_id", preview_kind)),
        qa_dataframe=_qa_dataframe(metrics, config, preview_kind=preview_kind),
        visual_sanity=sanity,
        captions=[f"{preview_kind} quick-look preview."],
        caveats=caveats,
        raw={"result": result},
    )
    if preview_kind in {"ideal", "ideal_target", "conical"}:
        preview.ideal_xy_intensity = xy
        preview.ideal_x_um = x_um
        preview.ideal_y_um = y_um
        preview.ideal_xz_intensity = xz
        preview.ideal_z_um = z_um
        preview.ideal_transverse_um = transverse_um
    else:
        preview.lab_xy_intensity = xy
        preview.lab_xz_intensity = xz
        preview.lab_z_um = z_um
        preview.lab_transverse_um = transverse_um
    return preview


def _display_interpolation(config: QuicklookConfig) -> tuple[bool, str]:
    method = str(config.plot_interpolation or config.display_interpolation_method)
    return method != "nearest", method


def run_slm_phase_preview(config: QuicklookConfig) -> QuicklookPreview:
    """Return the SLM phase mask preview without saving figures."""

    cfg = resolve_config(config)
    twin = make_twin_config(cfg)
    phase_info = _phase_from_config(twin)
    field = phase_info["field"]
    design = phase_info["design"]
    grid = field["grid"]
    metrics = {
        "case_id": "slm_phase_preview",
        "ell": int(twin.target.ell),
        "vortex_phase_on": bool(cfg.vortex_phase_on),
        "flatten_phase_before_axicon": bool(cfg.flatten_phase_before_axicon),
        "phase_bits": int(cfg.slm_phase_bits),
        "gamma_slm_deg": float(design.gamma_slm_deg),
        "computational_mode": cfg.computational_mode,
    }
    return QuicklookPreview(
        resolved_config=cfg,
        preview_kind="slm_phase",
        metrics=metrics,
        metrics_dataframe=_metrics_dataframe(metrics),
        qa_dataframe=_qa_dataframe(metrics, cfg, preview_kind="slm_phase"),
        captions=["SLM wrapped phase preview."],
        raw={"phase_info": phase_info},
        slm_phase_rad=np.asarray(field["phase"], dtype=float),
        slm_x_mm=np.asarray(grid["x"], dtype=float) / BT_MM,
        slm_y_mm=np.asarray(grid["x"], dtype=float) / BT_MM,
    )


def run_ideal_beam_preview(config: QuicklookConfig) -> QuicklookPreview:
    """Run and return the true scalar Bessel-Gauss target preview."""

    cfg = resolve_config(config)
    twin = make_twin_config(cfg)
    air = vbb_studies.beam_air_config(twin)
    design = _compute_design_from_config(air)
    grid = make_xy_grid(int(air.grid.ideal_N), float(air.grid.ideal_dx_m))
    field = build_bessel_gauss_field_ideal(grid, design, air.laser, include_vortex=bool(cfg.include_vortex and cfg.vortex_phase_on))
    volume = _bt().propagate_volume(
        field,
        grid,
        air.laser.wavelength_m,
        _air_z_values(cfg),
        n_medium=1.0,
        crop_pixels=min(int(air.grid.crop_pixels), int(grid["N"])),
        bandlimit=True,
        method=air.propagation.method,
        propagation_config=air.propagation,
    )
    result = {
        "path": "ideal_target",
        "study_kind": "beam_to_surface",
        "design": design,
        "focal_grid": grid,
        "volume": volume,
        "first_order_selected_fraction": 1.0,
        "beam_medium_n": 1.0,
    }
    metrics = _bt().extract_vortex_safe_metrics(result, air)
    metrics.update(
        {
            "case_id": "quicklook_ideal_target",
            "path": "ideal_target",
            "material": "air",
            "generation_method": "true_bessel_gauss_target",
            "propagation_power_label": "diagnostic",
        }
    )
    result["metrics"] = metrics
    return _beam_preview_from_result(cfg, result, preview_kind="ideal_target")


def run_conical_axicon_preview(config: QuicklookConfig) -> QuicklookPreview:
    """Run and return the conical axicon propagated air-path preview."""

    cfg = resolve_config(config)
    twin = make_twin_config(cfg)
    result = _bt().run_case(twin, preset=_mode_preset(cfg), path="ideal", case_id="quicklook_conical_axicon", z_values_m=_air_z_values(cfg))
    return _beam_preview_from_result(cfg, result, preview_kind="conical")


def known_good_visual_config(**updates: Any) -> QuicklookConfig:
    """Return the default visual sanity preset for notebook demonstrations."""

    cfg = default_quicklook_config(
        computational_mode="balanced",
        grid_size=384,
        axial_points=41,
        four_condition_phase_axial_points=31,
        crop_pixels=256,
        blaze_period_px=20,
        plot_interpolation="nearest",
        display_interpolation=False,
    )
    return with_updates(cfg, **updates) if updates else cfg


def quick_preview_config(**updates: Any) -> QuicklookConfig:
    """Return a faster debugging preset that is not the default visual reference."""

    cfg = default_quicklook_config(
        computational_mode="quick_preview",
        grid_size=192,
        axial_points=13,
        four_condition_phase_axial_points=7,
        crop_pixels=128,
        blaze_period_px=20,
        plot_interpolation="nearest",
        display_interpolation=False,
    )
    return with_updates(cfg, **updates) if updates else cfg


def run_gaussian_reference_preview(config: QuicklookConfig | None = None) -> QuicklookPreview:
    """Run a trusted propagated Gaussian visual sanity reference."""

    cfg = resolve_config(config or known_good_visual_config(vortex_phase_on=False, ell=0, vortex_charge=0, include_vortex=False))
    twin = make_twin_config(cfg, effective_ell=0, nominal_ell=0)
    air = vbb_studies.beam_air_config(twin)
    grid = make_xy_grid(int(air.grid.ideal_N), float(air.grid.ideal_dx_m))
    field = gaussian_amplitude(grid["R"], 6.0 * BT_UM)
    volume = _bt().propagate_volume(
        field,
        grid,
        air.laser.wavelength_m,
        _air_z_values(cfg),
        n_medium=1.0,
        crop_pixels=min(int(air.grid.crop_pixels), int(grid["N"])),
        bandlimit=True,
        method=air.propagation.method,
        propagation_config=air.propagation,
    )
    metrics = {
        "case_id": "gaussian_reference",
        "ell": 0,
        "path": "gaussian_reference",
        "generation_method": "trusted_gaussian_propagation",
        "propagation_power_label": "diagnostic",
        "vortex_main_ring_radius_um": np.nan,
        "core_first_zero_radius_um": np.nan,
    }
    result = {
        "path": "gaussian_reference",
        "study_kind": "beam_to_surface",
        "design": _compute_design_from_config(air),
        "focal_grid": grid,
        "volume": volume,
        "first_order_selected_fraction": 1.0,
        "beam_medium_n": 1.0,
        "metrics": metrics,
    }
    return _beam_preview_from_result(cfg, result, preview_kind="ideal_target")


def run_visual_sanity_reference_cases(config: QuicklookConfig | None = None) -> dict[str, QuicklookPreview]:
    """Run the required Stage 8.7D visual sanity reference cases."""

    base = resolve_config(config or known_good_visual_config())
    return {
        "gaussian_reference": run_gaussian_reference_preview(with_updates(base, vortex_phase_on=False, ell=0, vortex_charge=0, include_vortex=False)),
        "scalar_bessel_no_vortex_ideal": run_ideal_beam_preview(with_updates(base, vortex_phase_on=False, ell=0, vortex_charge=0, include_vortex=False)),
        "vortex_bessel_ideal_ell1": run_ideal_beam_preview(with_updates(base, vortex_phase_on=True, ell=1, vortex_charge=1, include_vortex=True)),
        "vortex_bessel_ideal_ell3": run_ideal_beam_preview(with_updates(base, vortex_phase_on=True, ell=3, vortex_charge=3, include_vortex=True)),
        "conical_axicon_propagated_no_vortex": run_conical_axicon_preview(with_updates(base, vortex_phase_on=False, ell=0, vortex_charge=0, include_vortex=False)),
        "conical_axicon_propagated_vortex": run_conical_axicon_preview(with_updates(base, vortex_phase_on=True, ell=3, vortex_charge=3, include_vortex=True)),
        "lab_realistic_holographic_route_known_good": run_lab_realistic_preview(known_good_visual_config()),
    }


def run_lab_realistic_preview(config: QuicklookConfig) -> QuicklookPreview:
    """Run and return a lab-realistic air-path beam preview."""

    cfg = resolve_config(config)
    twin = make_twin_config(cfg)
    result = _bt().run_case(twin, preset=_mode_preset(cfg), path="realistic", case_id="quicklook_lab", z_values_m=_air_z_values(cfg))
    return _beam_preview_from_result(cfg, result, preview_kind="lab")


def _optical_sanity_allows_sample(preview: QuicklookPreview) -> bool:
    return bool(preview.metrics.get("visual_sanity_pass", False)) and bool(preview.metrics.get("first_order_sanity_pass", True))


def _blocked_preview(config: QuicklookConfig, *, preview_kind: str, upstream: QuicklookPreview, reason: str) -> QuicklookPreview:
    metrics = {
        "case_id": f"quicklook_{preview_kind}_blocked",
        "visual_sanity_label": "blocked",
        "visual_sanity_pass": False,
        "visual_sanity_warning": VISUAL_SANITY_FAILED_TEXT,
        "blocked_reason": reason,
        "upstream_preview_kind": upstream.preview_kind,
        "upstream_visual_sanity_label": upstream.metrics.get("visual_sanity_label", "not_evaluated"),
        "upstream_first_order_sanity_label": upstream.metrics.get("first_order_sanity_label", "not_applicable"),
    }
    return QuicklookPreview(
        resolved_config=config,
        preview_kind=preview_kind,
        metrics=metrics,
        metrics_dataframe=_metrics_dataframe(metrics),
        qa_dataframe=_qa_dataframe(metrics, config, preview_kind=preview_kind),
        visual_sanity=dict(metrics),
        captions=[reason],
        caveats=[QUICKLOOK_CAVEAT, VISUAL_SANITY_FAILED_TEXT, reason],
        raw={"upstream_preview": upstream},
    )


def run_through_sample_preview(config: QuicklookConfig, lab_preview: QuicklookPreview | None = None) -> QuicklookPreview:
    """Run and return the through-sample preview fed by the lab-realistic surface field."""

    cfg = resolve_config(config)
    lab = lab_preview or run_lab_realistic_preview(cfg)
    if not _optical_sanity_allows_sample(lab):
        return _blocked_preview(
            cfg,
            preview_kind="through_sample",
            upstream=lab,
            reason="Through-sample preview blocked because upstream lab-realistic optical sanity failed.",
        )
    twin = make_twin_config(cfg)
    lab_result = lab.raw["result"]
    sample = vbb_sample_study.run_through_sample(
        lab_result["surface_field"],
        twin,
        correct_interface=bool(cfg.correct_interface),
        write_depth_m=float(cfg.write_depth_um) * BT_UM,
        z_values_m=_sample_z_values(cfg),
    )
    xz, transverse_um = _crop_xz_for_display(sample.volume_result["xz"], sample.volume_result["crop_grid"], cfg)
    z_um = np.asarray(sample.volume_result["z"], dtype=float) / BT_UM
    metrics = dict(sample.metrics)
    return QuicklookPreview(
        resolved_config=cfg,
        preview_kind="through_sample",
        metrics=metrics,
        metrics_dataframe=_metrics_dataframe(metrics, case_id="quicklook_through_sample"),
        qa_dataframe=_qa_dataframe(metrics, cfg, preview_kind="through_sample"),
        captions=["Through-sample scalar interface preview."],
        raw={"sample": sample, "lab_preview": lab},
        through_sample_xz_intensity=xz,
        through_sample_z_um=z_um,
        through_sample_transverse_um=transverse_um,
    )


def run_material_proxy_preview(config: QuicklookConfig, through_sample_preview: QuicklookPreview | None = None) -> QuicklookPreview:
    """Return the material-facing fluence/threshold proxy preview."""

    cfg = resolve_config(config)
    through = through_sample_preview or run_through_sample_preview(cfg)
    if through.raw.get("sample") is None:
        return _blocked_preview(
            cfg,
            preview_kind="material_proxy",
            upstream=through,
            reason="Material proxy blocked because upstream optical/sample preview failed visual sanity.",
        )
    sample = through.raw["sample"]
    twin = make_twin_config(cfg)
    threshold = float(twin.material.incubated_threshold_J_cm2(twin.laser.rep_rate_Hz))
    fluence = _bt().fluence_from_intensity(
        sample.volume_result["planes"]["peak"],
        sample.volume_result["crop_grid"]["dx"],
        twin.energy.pulse_energy_at_sample_J,
    )
    proxy = fluence / max(threshold, np.finfo(float).eps)
    proxy_crop, x_um, y_um = _crop_xy_for_display(proxy, sample.volume_result["crop_grid"], cfg)
    metrics = {
        "case_id": "material_proxy_preview",
        "incubated_threshold_J_cm2": threshold,
        "proxy_max_ratio": float(np.nanmax(proxy)),
        "material_model_status": "planning_proxy",
    }
    return QuicklookPreview(
        resolved_config=cfg,
        preview_kind="material_proxy",
        metrics=metrics,
        metrics_dataframe=_metrics_dataframe(metrics),
        qa_dataframe=_qa_dataframe(metrics, cfg, preview_kind="material_proxy"),
        captions=["Material fluence/threshold proxy preview; not calibrated material response."],
        caveats=[QUICKLOOK_CAVEAT, "Planning proxy only; not calibrated material damage/modification prediction."],
        raw={"through_sample_preview": through, "sample": sample},
        material_proxy_xy=proxy_crop,
        material_x_um=x_um,
        material_y_um=y_um,
    )


def plot_slm_preview(preview: QuicklookPreview) -> plt.Figure:
    """Plot a live SLM phase preview and return the figure."""

    display, method = _display_interpolation(preview.resolved_config)
    grid = _grid_from_axis_um(np.asarray(preview.slm_x_mm, dtype=float) * 1000.0)
    result = visuals.plot_slm_phase_mask(
        preview.slm_phase_rad,
        grid,
        title="SLM wrapped phase preview",
        unit="mm",
        display_interpolation=display,
        interpolation=method,
        phase_quantisation_bits=int(preview.resolved_config.slm_phase_bits),
        labels={"mode": preview.resolved_config.computational_mode, "ell": preview.metrics.get("ell")},
    )
    return result["fig"]


def plot_ideal_preview(preview: QuicklookPreview) -> plt.Figure:
    """Plot a live ideal beam preview and return the figure."""

    display, method = _display_interpolation(preview.resolved_config)
    grid = _grid_from_axis_um(preview.ideal_x_um)
    failed = not bool(preview.metrics.get("visual_sanity_pass", True))
    title = "Ideal Bessel-Gauss target preview" if preview.preview_kind == "ideal_target" else "Conical axicon propagated preview"
    result = visuals.plot_xy_xz_pair(
        preview.ideal_xy_intensity,
        preview.ideal_xz_intensity,
        grid=grid,
        z_um=preview.ideal_z_um,
        title=title,
        display_interpolation=display,
        interpolation=method,
        shared_color_scale=bool(preview.resolved_config.shared_colour_scale),
        qa_label=str(preview.metrics.get("visual_sanity_label", preview.metrics.get("propagation_power_label", "diagnostic"))),
        caveat_text=VISUAL_SANITY_FAILED_TEXT if failed else "raw-grid visual sanity guardrails passed",
    )
    return result["fig"]


def plot_lab_preview(preview: QuicklookPreview) -> plt.Figure:
    """Plot a live lab-realistic beam preview and return the figure."""

    display, method = _display_interpolation(preview.resolved_config)
    grid = _grid_from_axis_um(preview.lab_transverse_um)
    failed = not bool(preview.metrics.get("visual_sanity_pass", True))
    caveat = VISUAL_SANITY_FAILED_TEXT if failed else "lab-realistic diagnostic; inspect first-order/pupil metrics"
    result = visuals.plot_xy_xz_pair(
        preview.lab_xy_intensity,
        preview.lab_xz_intensity,
        grid=grid,
        z_um=preview.lab_z_um,
        title="Lab-realistic air-path preview",
        display_interpolation=display,
        interpolation=method,
        shared_color_scale=bool(preview.resolved_config.shared_colour_scale),
        qa_label=str(preview.metrics.get("visual_sanity_label", preview.metrics.get("propagation_power_label", "diagnostic"))),
        caveat_text=caveat,
    )
    return result["fig"]


def _warning_figure(title: str, metrics: Mapping[str, Any]) -> plt.Figure:
    vbb_style.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.axis("off")
    lines = [
        VISUAL_SANITY_FAILED_TEXT,
        str(metrics.get("blocked_reason", metrics.get("visual_sanity_reason", ""))),
        f"upstream visual sanity: {metrics.get('upstream_visual_sanity_label', metrics.get('visual_sanity_label', 'not_evaluated'))}",
        f"first-order sanity: {metrics.get('upstream_first_order_sanity_label', metrics.get('first_order_sanity_label', 'not_applicable'))}",
    ]
    ax.set_title(title)
    ax.text(
        0.02,
        0.78,
        "\n".join(line for line in lines if line),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff4b8", "edgecolor": "#D55E00", "linewidth": 1.0},
    )
    return fig


def plot_through_sample_preview(preview: QuicklookPreview) -> plt.Figure:
    """Plot a live through-sample preview and return the figure."""

    if preview.through_sample_xz_intensity is None:
        return _warning_figure("Through-sample preview blocked", preview.metrics)
    display, method = _display_interpolation(preview.resolved_config)
    grid = _grid_from_axis_um(preview.through_sample_transverse_um)
    result = visuals.plot_xz_intensity(
        preview.through_sample_xz_intensity,
        preview.through_sample_z_um,
        None,
        grid=grid,
        title="Through-sample preview: z measured from surface",
        display_interpolation=display,
        interpolation=method,
        qa_label=str(preview.metrics.get("correction", "uncorrected_interface")),
        caveat_text="surface at z=0; scalar interface diagnostic",
    )
    return result["fig"]


def plot_material_proxy_preview(preview: QuicklookPreview) -> plt.Figure:
    """Plot a live material proxy preview and return the figure."""

    if preview.material_proxy_xy is None:
        return _warning_figure("Material proxy blocked", preview.metrics)
    display, method = _display_interpolation(preview.resolved_config)
    grid = _grid_from_axis_um(preview.material_x_um)
    result = visuals.plot_xy_intensity(
        preview.material_proxy_xy,
        grid,
        title="Material fluence / threshold proxy",
        display_interpolation=display,
        interpolation=method,
        colorbar_label="fluence / incubated threshold [planning proxy]",
        caveat_text="uncalibrated planning proxy",
    )
    return result["fig"]


def _numeric_metric_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(a) & set(b)):
        try:
            av = float(a[key])
            bv = float(b[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(av) and np.isfinite(bv):
            rows.append({"metric": key, "config_a": av, "config_b": bv, "delta_b_minus_a": bv - av})
    return pd.DataFrame(rows)


def compare_configs(config_a: QuicklookConfig, config_b: QuicklookConfig, *, path: str = "lab") -> QuicklookComparison:
    """Compare two configs using target, conical, or lab-realistic previews."""

    cfg_a = resolve_config(config_a)
    cfg_b = resolve_config(config_b)
    path_key = str(path).lower().strip()
    if path_key in {"ideal", "target", "ideal_target"}:
        preview_a = run_ideal_beam_preview(cfg_a)
        preview_b = run_ideal_beam_preview(cfg_b)
    elif path_key in {"conical", "axicon", "ideal_conical"}:
        preview_a = run_conical_axicon_preview(cfg_a)
        preview_b = run_conical_axicon_preview(cfg_b)
    else:
        preview_a = run_lab_realistic_preview(cfg_a)
        preview_b = run_lab_realistic_preview(cfg_b)
    delta = _numeric_metric_delta(preview_a.metrics, preview_b.metrics)
    qa = pd.concat([preview_a.qa_dataframe, preview_b.qa_dataframe], ignore_index=True)
    return QuicklookComparison(cfg_a, cfg_b, preview_a, preview_b, delta, qa)


def _case_from_preview(preview: QuicklookPreview, label: str) -> dict[str, Any]:
    if preview.preview_kind in {"ideal", "ideal_target", "conical"}:
        return {
            "label": label,
            "xy": preview.ideal_xy_intensity,
            "xz": preview.ideal_xz_intensity,
            "z_um": preview.ideal_z_um,
            "grid": _grid_from_axis_um(preview.ideal_x_um),
            "metrics": preview.metrics,
            "qa_label": preview.metrics.get("visual_sanity_label", preview.metrics.get("propagation_power_label")),
        }
    return {
        "label": label,
        "xy": preview.lab_xy_intensity,
        "xz": preview.lab_xz_intensity,
        "z_um": preview.lab_z_um,
        "grid": _grid_from_axis_um(preview.lab_transverse_um),
        "metrics": preview.metrics,
        "qa_label": preview.metrics.get("visual_sanity_label", preview.metrics.get("propagation_power_label")),
    }


def plot_config_comparison(comparison: QuicklookComparison) -> plt.Figure:
    """Plot live before/after config comparison with shared colour scales."""

    cfg = comparison.config_a
    display, method = _display_interpolation(cfg)
    result = visuals.plot_case_comparison_grid(
        [
            _case_from_preview(comparison.preview_a, "config A"),
            _case_from_preview(comparison.preview_b, "config B"),
        ],
        title="Quick-look config comparison",
        shared_color_scale=bool(cfg.shared_colour_scale),
        display_interpolation=display,
        interpolation=method,
        include_phase=False,
        include_metrics=False,
    )
    return result["fig"]


def run_parameter_sweep_preview(config: QuicklookConfig, parameter: str, values: Sequence[Any]) -> QuicklookSweep:
    """Run a short lab-realistic one-parameter sweep for live review."""

    cfg = resolve_config(config)
    previews: list[QuicklookPreview] = []
    rows: list[dict[str, Any]] = []
    for value in values:
        trial = with_updates(cfg, **{str(parameter): value})
        preview = run_lab_realistic_preview(trial)
        previews.append(preview)
        row = dict(preview.metrics)
        row["sweep_parameter"] = str(parameter)
        row["sweep_value"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    qa = pd.concat([p.qa_dataframe for p in previews], ignore_index=True) if previews else pd.DataFrame()
    return QuicklookSweep(cfg, str(parameter), list(values), previews, summary, qa)


def plot_parameter_sweep_preview(sweep: QuicklookSweep) -> plt.Figure:
    """Plot compact metric trends for a live parameter sweep."""

    vbb_style.apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    x = sweep.summary["sweep_value"]
    plotted = False
    for metric in ["canonical_zone_um", "strict_bessel_region_um", "core_or_ring_peak_fluence_J_cm2"]:
        if metric in sweep.summary:
            ax.plot(x, sweep.summary[metric], marker="o", label=metric)
            plotted = True
    if not plotted:
        ax.plot(range(len(sweep.values)), np.arange(len(sweep.values)), marker="o", label="case index")
    ax.set_xlabel(sweep.parameter)
    ax.set_ylabel("metric value [reported units]")
    ax.set_title("Quick-look parameter sweep preview")
    ax.legend(loc="best")
    ax.text(
        0.01,
        0.01,
        "diagnostic quick-look; metrics computed from raw grid",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fff4b8", "alpha": 0.95, "linewidth": 0.0},
    )
    return fig


def _attach_visual_sanity_to_result(
    result: Mapping[str, Any],
    config: QuicklookConfig,
    *,
    include_first_order: bool = False,
) -> dict[str, Any]:
    volume = result["volume"]
    grid = volume["crop_grid"]
    axis_um = _axis_um_from_grid(grid)
    metrics = dict(result.get("metrics", {}))
    sanity = beam_visual_sanity_metrics(
        xy_intensity=volume["planes"]["peak"],
        x_um=axis_um,
        y_um=axis_um,
        xz_intensity=volume["xz"],
        metrics=metrics,
        config=config,
        include_first_order=include_first_order,
    )
    metrics.update(sanity)
    if isinstance(result, dict):
        result["metrics"] = metrics
    return metrics


def _volume_case_to_plot_case(
    case_id: str,
    label: str,
    result: Mapping[str, Any],
    phase_info: Mapping[str, Any] | None = None,
    *,
    phase_labels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    volume = result["volume"]
    metrics = dict(result.get("metrics", {}))
    metrics.setdefault("case_id", case_id)
    metrics.setdefault("path", result.get("path", "unknown"))
    qa_label = metrics.get("visual_sanity_label", metrics.get("propagation_power_label"))
    case = {
        "case_id": case_id,
        "label": label,
        "xy": volume["planes"]["peak"],
        "xz": volume["xz"],
        "grid": volume["crop_grid"],
        "z_um": np.asarray(volume["z"], dtype=float) / BT_UM,
        "metrics": metrics,
        "qa_label": qa_label,
    }
    if phase_info is not None:
        field = phase_info["field"]
        case.update(
            {
                "phase": field["phase"],
                "slm_grid": field["grid"],
                "phase_bits": metrics.get("phase_bits"),
                "phase_labels": dict(phase_labels or {}),
            }
        )
    return case


def _metadata_base(config: QuicklookConfig, run_id: str, generated_at_utc: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "source_schema_version": QUICKLOOK_SCHEMA_VERSION,
        "computational_mode": config.computational_mode,
        "parameter_metadata": asdict(config),
        "caveats": QUICKLOOK_CAVEAT,
        "flattened_phase_convention": FLATTENED_PHASE_CONVENTION,
        "registry_status": "diagnostic_allowed",
    }


def _row_from_metrics(
    metrics: Mapping[str, Any],
    *,
    config: QuicklookConfig,
    run_id: str,
    generated_at_utc: str,
    case_id: str,
    stage_label: str,
    caveat: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "generated_at_utc": generated_at_utc,
        "source_schema_version": QUICKLOOK_SCHEMA_VERSION,
        "computational_mode": config.computational_mode,
        "case_id": case_id,
        "stage_label": stage_label,
        "parameter_metadata_json": json.dumps(asdict(config), sort_keys=True),
        "caveats": caveat or QUICKLOOK_CAVEAT,
        "final_export_allowed": False,
        "registry_status": figure_registry.classify_path("outputs/csv/quicklook/quicklook_metric_summary.csv").status,
    }
    for key, value in metrics.items():
        if isinstance(value, (np.integer,)):
            row[key] = int(value)
        elif isinstance(value, (np.floating,)):
            row[key] = float(value)
        else:
            row[key] = value
    if extra:
        row.update(dict(extra))
    return row


def output_directories(output_root: str | Path) -> dict[str, Path]:
    """Return and create quick-look output directories."""

    root = Path(output_root)
    paths = {
        "root": root,
        "csv": root / "csv" / "quicklook",
        "json": root / "json" / "quicklook",
        "figures": root / "figures" / "quicklook",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_quicklook_outputs(
    *,
    config: QuicklookConfig,
    output_root: str | Path,
    metric_rows: Sequence[Mapping[str, Any]] | None = None,
    figure_records: Sequence[Mapping[str, Any]] | None = None,
    run_id: str | None = None,
    generated_at_utc: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write quick-look CSV, JSON, and at least one governed diagnostic figure."""

    config.validate()
    rid = run_id or new_run_id()
    generated = generated_at_utc or utc_now()
    paths = output_directories(output_root)
    rows = list(metric_rows or [])
    if not rows:
        rows = [
            _row_from_metrics(
                {"case_id": "writer_smoke", "propagation_power_label": "diagnostic"},
                config=config,
                run_id=rid,
                generated_at_utc=generated,
                case_id="writer_smoke",
                stage_label="output_writer",
            )
        ]
    metric_path = paths["csv"] / "quicklook_metric_summary.csv"
    metric_df = pd.DataFrame(rows)
    metric_df.to_csv(metric_path, index=False)
    comparison_csvs: list[Path] = []
    if "stage_label" in metric_df.columns:
        comparison_df = metric_df[metric_df["stage_label"].eq("four_condition_phase_comparison")].copy()
        if not comparison_df.empty:
            for name in [
                "quicklook_four_condition_phase_comparison_summary.csv",
                "four_condition_phase_comparison_metrics.csv",
            ]:
                out_csv = paths["csv"] / name
                comparison_df.to_csv(out_csv, index=False)
                comparison_csvs.append(out_csv)

    config_payload = {
        **_metadata_base(config, rid, generated),
        "allowed_computational_modes": list(ALLOWED_COMPUTATIONAL_MODES),
        "comparison_presets": {
            "four_condition_phase_comparison": list(FOUR_CONDITION_PHASE_COMPARISON_PRESETS),
        },
        "registry": {
            "csv": asdict(figure_registry.classify_path("outputs/csv/quicklook/quicklook_metric_summary.csv")),
            "json": asdict(figure_registry.classify_path("outputs/json/quicklook/quicklook_config.json")),
            "figures": asdict(figure_registry.classify_path("outputs/figures/quicklook/example.png")),
        },
        "extra": dict(extra_config or {}),
    }
    config_path = paths["json"] / "quicklook_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    records = list(figure_records or [])
    if not records:
        fig_result = visuals.plot_metric_summary_box(rows[0], title="Quick-look writer smoke")
        records.append(
            {
                "filename": "00_quicklook_writer_smoke.png",
                "fig": fig_result["fig"],
                "caption": "Stage 8.7 quick-look writer smoke figure. " + QUICKLOOK_CAVEAT,
                "metadata": fig_result["metadata"],
            }
        )

    saved_figures: list[Path] = []
    for record in records:
        fig = record["fig"]
        filename = str(record.get("filename") or "quicklook_figure.png")
        caption = str(record.get("caption") or QUICKLOOK_CAVEAT)
        metadata = {
            **_metadata_base(config, rid, generated),
            **dict(record.get("metadata", {})),
        }
        saved = vbb_style.save_figure(fig, paths["figures"] / filename, caption, metadata=metadata)
        saved_figures.append(saved)
        plt.close(fig)

    return {
        "run_id": rid,
        "generated_at_utc": generated,
        "metric_summary_csv": metric_path,
        "four_condition_phase_comparison_csvs": comparison_csvs,
        "config_json": config_path,
        "figures": saved_figures,
        "metric_summary": metric_df,
        "paths": paths,
    }


def _make_figure_record(filename: str, result: Mapping[str, Any], caption: str, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(result, plt.Figure):
        return {
            "filename": filename,
            "fig": result,
            "caption": caption,
            "metadata": dict(metadata or {}),
        }
    return {
        "filename": filename,
        "fig": result["fig"],
        "caption": caption,
        "metadata": dict(result.get("metadata", {})) | dict(metadata or {}),
    }


def build_four_condition_phase_configs(config: QuicklookConfig) -> list[dict[str, Any]]:
    """Return the four neutral phase-state comparison configs."""

    config.validate()
    cases: list[dict[str, Any]] = []
    for preset in FOUR_CONDITION_PHASE_COMPARISON_PRESETS:
        effective_ell = preset["effective_ell"]
        if effective_ell is None:
            effective_ell = int(config.vortex_charge)
        case_cfg = replace(config, vortex_phase_on=bool(preset["vortex_phase_on"]))
        twin = make_twin_config(case_cfg, effective_ell=int(effective_ell), nominal_ell=int(config.vortex_charge))
        cases.append(
            {
                "preset": dict(preset),
                "quicklook_config": case_cfg,
                "twin_config": twin,
                "effective_ell": int(effective_ell),
            }
        )
    return cases


def run_four_condition_phase_comparison(
    config: QuicklookConfig,
    *,
    run_id: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run the four-condition axicon phase-state comparison."""

    rid = run_id or new_run_id()
    generated = generated_at_utc or utc_now()
    ideal_plot_cases: list[dict[str, Any]] = []
    lab_plot_cases: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    lab_block_reasons: list[str] = []
    for case in build_four_condition_phase_configs(config):
        preset = case["preset"]
        case_twin = case["twin_config"]
        phase_case = _phase_from_config(case_twin)
        ideal_result = _bt().run_case(
            case_twin,
            preset=_mode_preset(config),
            path="ideal",
            case_id=str(preset["case_id"]),
            z_values_m=_air_z_values(config, int(config.four_condition_phase_axial_points)),
        )
        ideal_metrics = _attach_visual_sanity_to_result(ideal_result, config, include_first_order=False)
        extra = {
            "phase_comparison_label": preset["label"],
            "vortex_phase_on": bool(preset["vortex_phase_on"]),
            "flatten_pre_axicon_phase": bool(preset["flatten_pre_axicon_phase"]),
            "effective_ell": int(case["effective_ell"]),
            "flattened_phase_convention": FLATTENED_PHASE_CONVENTION,
            "comparison_route": "ideal_conical",
        }
        metric_rows.append(
            _row_from_metrics(
                ideal_metrics,
                config=config,
                run_id=rid,
                generated_at_utc=generated,
                case_id=str(preset["case_id"]),
                stage_label="four_condition_phase_comparison",
                caveat=str(preset["caveat"]),
                extra=extra,
            )
        )
        ideal_plot_cases.append(
            _volume_case_to_plot_case(
                str(preset["case_id"]),
                str(preset["label"]),
                ideal_result,
                phase_case,
                phase_labels={
                    "effective ell": int(case["effective_ell"]),
                    "flatten": bool(preset["flatten_pre_axicon_phase"]),
                    "route": "ideal/conical",
                },
            )
        )
        if bool(config.run_four_condition_lab_realistic_if_sane):
            lab_result = _bt().run_case(
                case_twin,
                preset=_mode_preset(config),
                path="realistic",
                case_id=str(preset["case_id"]),
                z_values_m=_air_z_values(config, int(config.four_condition_phase_axial_points)),
            )
            lab_metrics = _attach_visual_sanity_to_result(lab_result, config, include_first_order=True)
            lab_extra = dict(extra)
            lab_extra["comparison_route"] = "lab_realistic"
            metric_rows.append(
                _row_from_metrics(
                    lab_metrics,
                    config=config,
                    run_id=rid,
                    generated_at_utc=generated,
                    case_id=str(preset["case_id"]),
                    stage_label="four_condition_phase_comparison",
                    caveat=str(preset["caveat"]),
                    extra=lab_extra,
                )
            )
            if bool(lab_metrics.get("visual_sanity_pass", False)) and bool(lab_metrics.get("first_order_sanity_pass", True)):
                lab_plot_cases.append(
                    _volume_case_to_plot_case(
                        str(preset["case_id"]),
                        f"{preset['label']} | lab",
                        lab_result,
                        phase_case,
                        phase_labels={
                            "effective ell": int(case["effective_ell"]),
                            "flatten": bool(preset["flatten_pre_axicon_phase"]),
                            "route": "lab-realistic",
                        },
                    )
                )
            else:
                lab_block_reasons.append(f"{preset['case_id']}: {lab_metrics.get('visual_sanity_reason', VISUAL_SANITY_FAILED_TEXT)}")
    return {
        "metric_rows": metric_rows,
        "plot_cases": ideal_plot_cases,
        "ideal_plot_cases": ideal_plot_cases,
        "lab_plot_cases": lab_plot_cases,
        "lab_blocked": bool(config.run_four_condition_lab_realistic_if_sane and lab_block_reasons),
        "lab_block_reasons": lab_block_reasons,
    }


def plot_four_condition_phase_comparison(
    cases: Sequence[Mapping[str, Any]],
    *,
    panel: str = "grid",
    display_interpolation: bool = True,
    interpolation: str | None = "bilinear",
) -> dict[str, Any]:
    """Plot the four-condition phase-state comparison."""

    panel_key = str(panel).lower().strip()
    if panel_key == "grid":
        return visuals.plot_case_comparison_grid(
            cases,
            title="Four-condition axicon phase-state comparison",
            shared_color_scale=True,
            display_interpolation=display_interpolation,
            interpolation=interpolation,
        )
    if panel_key not in {"phase_masks", "xy_profiles", "xz_maps"}:
        raise ValueError("panel must be one of: grid, phase_masks, xy_profiles, xz_maps")

    vbb_style.apply_style()
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 3.8), squeeze=False, constrained_layout=True)
    ref = None
    if panel_key in {"xy_profiles", "xz_maps"}:
        ref = max(
            _positive_max(case["xy" if panel_key == "xy_profiles" else "xz"])
            for case in cases
        )
    metadata: list[dict[str, Any]] = []
    for ax, case in zip(axes[0], cases):
        label = str(case.get("label", case.get("case_id", "case")))
        metrics = dict(case.get("metrics", {}))
        caveat_text = VISUAL_SANITY_FAILED_TEXT if not bool(metrics.get("visual_sanity_pass", True)) else None
        if panel_key == "phase_masks":
            result = visuals.plot_slm_phase_mask(
                case["phase"],
                case.get("slm_grid") or case.get("grid"),
                ax=ax,
                title=label,
                display_interpolation=display_interpolation,
                interpolation=interpolation,
                phase_quantisation_bits=case.get("phase_bits"),
                labels=case.get("phase_labels"),
            )
        elif panel_key == "xy_profiles":
            result = visuals.plot_xy_intensity(
                case["xy"],
                case.get("grid"),
                ax=ax,
                title=label,
                reference_max=ref,
                display_interpolation=display_interpolation,
                interpolation=interpolation,
                qa_label=case.get("qa_label"),
                caveat_text=caveat_text,
            )
        else:
            result = visuals.plot_xz_intensity(
                case["xz"],
                case.get("z_um"),
                None,
                grid=case.get("grid"),
                ax=ax,
                title=label,
                reference_max=ref,
                display_interpolation=display_interpolation,
                interpolation=interpolation,
                qa_label=case.get("qa_label"),
                caveat_text=caveat_text,
            )
        metadata.append(result["metadata"])
    fig.suptitle("Four-condition axicon phase-state comparison")
    return {"fig": fig, "axes": axes, "metadata": {"panel": panel_key, "cell_metadata": metadata, "case_count": len(cases)}}


def save_four_condition_phase_comparison(
    *,
    config: QuicklookConfig,
    output_root: str | Path,
    metric_rows: Sequence[Mapping[str, Any]],
    figure_records: Sequence[Mapping[str, Any]],
    run_id: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Save neutral four-condition comparison metrics and figures."""

    return write_quicklook_outputs(
        config=config,
        output_root=output_root,
        metric_rows=metric_rows,
        figure_records=figure_records,
        run_id=run_id,
        generated_at_utc=generated_at_utc,
    )


def run_quicklook_pipeline(
    config: QuicklookConfig,
    *,
    output_root: str | Path = Path(__file__).resolve().parents[2] / "outputs",
    save_outputs: bool | None = None,
) -> dict[str, Any]:
    """Run the live quick-look workflow and optionally write diagnostic outputs."""

    cfg = resolve_config(config)
    slm = run_slm_phase_preview(cfg)
    ideal = run_ideal_beam_preview(cfg)
    conical = run_conical_axicon_preview(cfg)
    lab = run_lab_realistic_preview(cfg)
    through = run_through_sample_preview(cfg, lab)
    material = run_material_proxy_preview(cfg, through)
    comparison = run_four_condition_phase_comparison(cfg) if cfg.run_four_condition_phase_comparison else {"metric_rows": [], "plot_cases": []}
    metrics_dataframe = pd.concat(
        [
            ideal.metrics_dataframe.assign(stage_label="ideal_target"),
            conical.metrics_dataframe.assign(stage_label="conical_axicon_air"),
            lab.metrics_dataframe.assign(stage_label="lab_air"),
            through.metrics_dataframe.assign(stage_label="through_sample"),
            material.metrics_dataframe.assign(stage_label="material_proxy"),
            pd.DataFrame(comparison["metric_rows"]),
        ],
        ignore_index=True,
        sort=False,
    )
    qa_dataframe = pd.concat([slm.qa_dataframe, ideal.qa_dataframe, lab.qa_dataframe, through.qa_dataframe, material.qa_dataframe], ignore_index=True)
    payload: dict[str, Any] = {
        "resolved_config": cfg,
        "slm": slm,
        "ideal": ideal,
        "conical": conical,
        "lab": lab,
        "through_sample": through,
        "material_proxy": material,
        "four_condition_phase_comparison": comparison,
        "metrics_dataframe": metrics_dataframe,
        "qa_dataframe": qa_dataframe,
        "caveats": [QUICKLOOK_CAVEAT, FLATTENED_PHASE_CONVENTION],
        "saved": None,
    }
    should_save = bool(cfg.save_outputs if save_outputs is None else save_outputs)
    if should_save:
        payload["saved"] = run_quicklook(cfg, output_root=output_root)
    return payload


def run_quicklook(
    config: QuicklookConfig | None = None,
    *,
    output_root: str | Path = Path(__file__).resolve().parents[2] / "outputs",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute the Stage 8.7 quick-look simulator and write outputs."""

    cfg = resolve_config(default_config() if config is None else config)
    cfg.validate()
    rid = run_id or new_run_id()
    generated = utc_now()
    twin = make_twin_config(cfg)
    phase_info = _phase_from_config(twin)
    slm_preview = run_slm_phase_preview(cfg)
    ideal_preview = run_ideal_beam_preview(cfg)
    conical_preview = run_conical_axicon_preview(cfg)
    lab_preview = run_lab_realistic_preview(cfg)
    through_preview = run_through_sample_preview(cfg, lab_preview)
    material_preview = run_material_proxy_preview(cfg, through_preview)
    lab_result = lab_preview.raw.get("result", {})

    metric_rows: list[dict[str, Any]] = [
        _row_from_metrics(
            ideal_preview.metrics,
            config=cfg,
            run_id=rid,
            generated_at_utc=generated,
            case_id="quicklook_ideal_target",
            stage_label="ideal_target",
        ),
        _row_from_metrics(
            conical_preview.metrics,
            config=cfg,
            run_id=rid,
            generated_at_utc=generated,
            case_id="quicklook_conical_axicon",
            stage_label="conical_axicon_air",
        ),
        _row_from_metrics(
            lab_preview.metrics,
            config=cfg,
            run_id=rid,
            generated_at_utc=generated,
            case_id="quicklook_lab",
            stage_label="lab_air",
        ),
        _row_from_metrics(
            through_preview.metrics,
            config=cfg,
            run_id=rid,
            generated_at_utc=generated,
            case_id="quicklook_through_sample",
            stage_label="through_sample",
            caveat="Through-sample scalar interface diagnostic; material response remains a planning proxy.",
        ),
        _row_from_metrics(
            material_preview.metrics,
            config=cfg,
            run_id=rid,
            generated_at_utc=generated,
            case_id="quicklook_material_proxy",
            stage_label="material_proxy",
            caveat="Material proxy is planning-only and is blocked if upstream optical sanity fails.",
        ),
    ]

    figures: list[dict[str, Any]] = []
    figures.append(
        _make_figure_record(
            "01_quicklook_slm_phase_mask.png",
            visuals.plot_slm_phase_mask(
                phase_info["field"]["phase"],
                phase_info["field"]["grid"],
                title="Quick-look SLM phase mask",
                display_interpolation=cfg.display_interpolation,
                interpolation=cfg.display_interpolation_method,
                phase_quantisation_bits=cfg.slm_phase_bits,
                labels={"ell": twin.target.ell, "mode": cfg.computational_mode},
            ),
            "Stage 8.7 diagnostic SLM wrapped phase mask. Axes are SLM-plane millimetres; phase is wrapped radians. "
            + QUICKLOOK_CAVEAT,
        )
    )
    if "order" in lab_result:
        figures.append(
            _make_figure_record(
                "02_quicklook_fourier_filter_plane.png",
                visuals.plot_fourier_filter_plane(
                    lab_result["order"].get("A", np.zeros_like(phase_info["field"]["phase"])),
                    lab_result.get("pupil_grid"),
                    title="Quick-look first-order filter plane",
                    carrier_lpmm=lab_result["order"].get("carrier_lpmm"),
                    filter_radius_lpmm=lab_result["order"].get("filter_radius_lpmm"),
                    selected_fraction=lab_result["order"].get("selected_fraction"),
                    display_interpolation=cfg.display_interpolation,
                    interpolation=cfg.display_interpolation_method,
                ),
                "Stage 8.7D diagnostic Fourier filter plane. Zero order, expected +1 order, selected aperture, selected fraction, and rejected fraction are displayed. "
                + QUICKLOOK_CAVEAT,
            )
        )
    pupil_grid = lab_result.get("pupil_grid") if isinstance(lab_result, Mapping) else None
    if isinstance(pupil_grid, Mapping):
        pupil = (np.asarray(pupil_grid["R"], dtype=float) <= float(twin.objective.pupil_radius_m)).astype(float)
        figures.append(
            _make_figure_record(
                "03_quicklook_pupil_plane.png",
                visuals.plot_pupil_plane(
                    pupil,
                    pupil_grid,
                    title="Quick-look objective pupil",
                    pupil_radius=twin.objective.pupil_radius_m / BT_MM,
                    display_interpolation=cfg.display_interpolation,
                    interpolation=cfg.display_interpolation_method,
                ),
                "Stage 8.7 diagnostic pupil-plane mask. Axes are pupil-plane millimetres; colourbar is amplitude/mask value. "
                + QUICKLOOK_CAVEAT,
            )
        )

    figures.append(
        _make_figure_record(
            "04_quicklook_ideal_target_xy_xz.png",
            plot_ideal_preview(ideal_preview),
            "Stage 8.7D ideal mathematical Bessel-Gauss target XY/XZ diagnostic. "
            + QUICKLOOK_CAVEAT,
            ideal_preview.visual_sanity,
        )
    )
    figures.append(
        _make_figure_record(
            "05_quicklook_conical_axicon_xy_xz.png",
            plot_ideal_preview(conical_preview),
            "Stage 8.7D conical axicon propagated XY/XZ diagnostic, separated from the true Bessel-Gauss target. "
            + QUICKLOOK_CAVEAT,
            conical_preview.visual_sanity,
        )
    )
    figures.append(
        _make_figure_record(
            "06_quicklook_lab_xy_xz.png",
            plot_lab_preview(lab_preview),
            "Stage 8.7D lab-realistic air-path XY/XZ diagnostic. Failed visual sanity is labelled as not a beam prediction. "
            + QUICKLOOK_CAVEAT,
            lab_preview.visual_sanity,
        )
    )
    figures.append(
        _make_figure_record(
            "07_quicklook_through_sample_xz.png",
            plot_through_sample_preview(through_preview),
            "Stage 8.7D through-sample XZ diagnostic. Blocked if upstream lab-realistic optical sanity fails. "
            + QUICKLOOK_CAVEAT,
            through_preview.visual_sanity,
        )
    )
    figures.append(
        _make_figure_record(
            "08_quicklook_material_proxy_map.png",
            plot_material_proxy_preview(material_preview),
            "Stage 8.7D material-facing fluence/threshold proxy. Blocked if upstream optical/sample sanity fails. "
            + QUICKLOOK_CAVEAT,
            material_preview.visual_sanity,
        )
    )

    four_condition_cases: list[dict[str, Any]] = []
    if cfg.run_four_condition_phase_comparison:
        comparison = run_four_condition_phase_comparison(cfg, run_id=rid, generated_at_utc=generated)
        metric_rows.extend(comparison["metric_rows"])
        four_condition_cases = list(comparison["ideal_plot_cases"])
        figures.append(
            _make_figure_record(
                "four_condition_phase_comparison_phase_masks.png",
                plot_four_condition_phase_comparison(
                    four_condition_cases,
                    panel="phase_masks",
                    display_interpolation=cfg.display_interpolation,
                    interpolation=cfg.display_interpolation_method,
                ),
                "Stage 8.7D four-condition axicon phase-state comparison phase-mask panel. "
                + FLATTENED_PHASE_CONVENTION
                + " "
                + QUICKLOOK_CAVEAT,
                {"flattened_phase_convention": FLATTENED_PHASE_CONVENTION},
            )
        )
        figures.append(
            _make_figure_record(
                "four_condition_phase_comparison_xy_profiles.png",
                plot_four_condition_phase_comparison(
                    four_condition_cases,
                    panel="xy_profiles",
                    display_interpolation=cfg.display_interpolation,
                    interpolation=cfg.display_interpolation_method,
                ),
                "Stage 8.7D four-condition ideal/conical axicon phase-state comparison transverse XY profile panel. "
                + QUICKLOOK_CAVEAT,
                {"flattened_phase_convention": FLATTENED_PHASE_CONVENTION},
            )
        )
        figures.append(
            _make_figure_record(
                "four_condition_phase_comparison_xz_maps.png",
                plot_four_condition_phase_comparison(
                    four_condition_cases,
                    panel="xz_maps",
                    display_interpolation=cfg.display_interpolation,
                    interpolation=cfg.display_interpolation_method,
                ),
                "Stage 8.7D four-condition ideal/conical axicon phase-state comparison axial XZ propagation panel. "
                + QUICKLOOK_CAVEAT,
                {"flattened_phase_convention": FLATTENED_PHASE_CONVENTION},
            )
        )
        lab_cases = list(comparison.get("lab_plot_cases", []))
        if len(lab_cases) == len(four_condition_cases) and lab_cases:
            figures.append(
                _make_figure_record(
                    "four_condition_phase_comparison_lab_xy_profiles.png",
                    plot_four_condition_phase_comparison(
                        lab_cases,
                        panel="xy_profiles",
                        display_interpolation=cfg.display_interpolation,
                        interpolation=cfg.display_interpolation_method,
                    ),
                    "Stage 8.7D four-condition lab-realistic XY comparison, emitted only after visual sanity passes. "
                    + QUICKLOOK_CAVEAT,
                    {"flattened_phase_convention": FLATTENED_PHASE_CONVENTION, "lab_visual_sanity": "passed"},
                )
            )
            figures.append(
                _make_figure_record(
                    "four_condition_phase_comparison_lab_xz_maps.png",
                    plot_four_condition_phase_comparison(
                        lab_cases,
                        panel="xz_maps",
                        display_interpolation=cfg.display_interpolation,
                        interpolation=cfg.display_interpolation_method,
                    ),
                    "Stage 8.7D four-condition lab-realistic XZ comparison, emitted only after visual sanity passes. "
                    + QUICKLOOK_CAVEAT,
                    {"flattened_phase_convention": FLATTENED_PHASE_CONVENTION, "lab_visual_sanity": "passed"},
                )
            )

    if cfg.enable_parameter_delta_comparison:
        delta_cfg = replace(cfg, vortex_charge=int(cfg.delta_vortex_charge), target_core_diameter_um=float(cfg.delta_target_core_diameter_um))
        delta_twin = make_twin_config(delta_cfg)
        delta_phase = _phase_from_config(delta_twin)
        delta_result = _bt().run_case(delta_twin, preset=_mode_preset(delta_cfg), path="realistic", case_id="quicklook_after_delta", z_values_m=_air_z_values(delta_cfg))
        _attach_visual_sanity_to_result(delta_result, delta_cfg, include_first_order=True)
        before_case = _volume_case_to_plot_case("before", "before", lab_result, phase_info)
        after_case = _volume_case_to_plot_case("after", "after delta", delta_result, delta_phase)
        figures.append(
            _make_figure_record(
                "09_quicklook_parameter_delta_comparison.png",
                visuals.plot_parameter_delta_comparison(
                    before_case,
                    after_case,
                    title="Quick-look before/after parameter delta",
                    display_interpolation=cfg.display_interpolation,
                    interpolation=cfg.display_interpolation_method,
                ),
                "Stage 8.7 optional before/after parameter-delta diagnostic. "
                + QUICKLOOK_CAVEAT,
            )
        )

    output = write_quicklook_outputs(
        config=cfg,
        output_root=output_root,
        metric_rows=metric_rows,
        figure_records=figures,
        run_id=rid,
        generated_at_utc=generated,
        extra_config={
            "config_summary": config_summary_frame(cfg).to_dict(orient="records"),
            "qa_summary": {
                "rows_written": len(metric_rows),
                "figures_requested": len(figures),
                "quicklook_registry_status": figure_registry.classify_path(
                    "outputs/figures/quicklook/four_condition_phase_comparison_phase_masks.png"
                ).status,
            },
        },
    )
    output.update(
        {
            "config_summary": config_summary_frame(cfg),
            "slm": slm_preview,
            "ideal": ideal_preview,
            "conical": conical_preview,
            "lab": lab_preview,
            "through_sample": through_preview,
            "material_proxy": material_preview,
            "four_condition_phase_cases": four_condition_cases,
        }
    )
    return output


__all__ = [
    "ALLOWED_COMPUTATIONAL_MODES",
    "FLATTENED_PHASE_CONVENTION",
    "FIRST_ORDER_SANITY_FAILED_TEXT",
    "QUICKLOOK_CAVEAT",
    "QUICKLOOK_SCHEMA_VERSION",
    "VISUAL_SANITY_FAILED_TEXT",
    "FOUR_CONDITION_PHASE_COMPARISON_PRESETS",
    "QuicklookConfig",
    "QuicklookComparison",
    "QuicklookPreview",
    "QuicklookSweep",
    "build_four_condition_phase_configs",
    "beam_centre_offset_um",
    "beam_visual_sanity_metrics",
    "centre_intensity_fraction",
    "compare_configs",
    "config_summary_frame",
    "config_to_jsonable",
    "default_config",
    "default_quicklook_config",
    "default_config_path",
    "expected_ring_radius_error",
    "first_order_selection_sanity",
    "known_good_visual_config",
    "load_config",
    "load_quicklook_config",
    "make_twin_config",
    "output_directories",
    "plot_config_comparison",
    "plot_four_condition_phase_comparison",
    "plot_ideal_preview",
    "plot_lab_preview",
    "plot_material_proxy_preview",
    "plot_parameter_sweep_preview",
    "plot_slm_preview",
    "plot_through_sample_preview",
    "quick_preview_config",
    "radial_symmetry_score",
    "resolve_config",
    "ringness_score",
    "run_conical_axicon_preview",
    "run_gaussian_reference_preview",
    "run_ideal_beam_preview",
    "run_lab_realistic_preview",
    "run_material_proxy_preview",
    "run_parameter_sweep_preview",
    "run_four_condition_phase_comparison",
    "run_quicklook",
    "run_quicklook_pipeline",
    "run_slm_phase_preview",
    "run_through_sample_preview",
    "run_visual_sanity_reference_cases",
    "save_default_config",
    "save_four_condition_phase_comparison",
    "save_quicklook_config",
    "with_updates",
    "write_quicklook_outputs",
    "xz_structure_score",
]
