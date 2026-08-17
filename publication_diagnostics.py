from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bessel_twin_core as bt
from vbb_study import setup_study, vbb_style, vbb_vector
from vbb_study.publication.tables import (
    SCALAR_OUTPUT_SCHEMA_VERSION,
    SCALAR_SUMMARY_COLUMNS,
    annotate_scalar_row,
    ordered_row,
    propagation_power_label as _canonical_power_label,
)


# Phase 7 note: SOURCE_SCHEMA_VERSION now aliases the canonical version
# from vbb_study.publication.tables so all scalars use one version string.
SOURCE_SCHEMA_VERSION = SCALAR_OUTPUT_SCHEMA_VERSION
SCALAR_ENGINE_VERSION = "structured_beam_atlas_v2"


DEFAULT_SHORTLIST = [
    {"label": "ell0_core3_L150", "ell": 0, "D_um": 3.0, "L_um": 150.0, "Ein_uJ": 10.0},
    {"label": "ell3_core3_L150", "ell": 3, "D_um": 3.0, "L_um": 150.0, "Ein_uJ": 10.0},
    {"label": "ell5_core4_L200", "ell": 5, "D_um": 4.0, "L_um": 200.0, "Ein_uJ": 20.0},
]


def new_csv_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_commit_or_empty() -> str:
    commit = setup_study.code_version(Path(__file__).resolve().parent)
    return str(commit or "")


def add_csv_metadata(
    df: pd.DataFrame,
    *,
    preset: str | None = None,
    run_id: str | None = None,
    generated_at_utc: str | None = None,
    engine_version: str = SCALAR_ENGINE_VERSION,
    source_schema_version: str = SOURCE_SCHEMA_VERSION,
    git_commit: str | None = None,
) -> pd.DataFrame:
    """Return a copy of a CSV-bound table with stale-output guard columns."""

    out = df.copy()
    metadata = [
        ("run_id", run_id or new_csv_run_id()),
        ("engine_version", engine_version),
        ("git_commit", _git_commit_or_empty() if git_commit is None else str(git_commit)),
        ("preset", "" if preset is None else str(preset)),
        ("generated_at_utc", generated_at_utc or datetime.now(timezone.utc).isoformat()),
        ("source_schema_version", source_schema_version),
    ]
    for index, (column, value) in enumerate(metadata):
        if column in out.columns:
            out[column] = value
        else:
            out.insert(index, column, value)
    return out


def _normalise(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    peak = float(np.max(arr)) if arr.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(arr)
    return arr / peak


def _sqrt_normalise(data: np.ndarray) -> np.ndarray:
    return np.sqrt(_normalise(data))


def _vector_hardware_label(vector_hardware: Any, *, default: str = "") -> str:
    if vector_hardware is None:
        return default
    if isinstance(vector_hardware, str):
        return vector_hardware.lower().strip() or default
    return "custom"


def _xy_extent_um(grid: Dict[str, Any]) -> list[float]:
    x = np.asarray(grid["x"], dtype=float) / bt.um
    if "y" in grid:
        y = np.asarray(grid["y"], dtype=float) / bt.um
    else:
        y = x
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _xy_extent_mm(grid: Dict[str, Any]) -> list[float]:
    x = np.asarray(grid["x"], dtype=float) / bt.mm
    if "y" in grid:
        y = np.asarray(grid["y"], dtype=float) / bt.mm
    else:
        y = x
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _xz_extent_um(volume: Dict[str, Any]) -> list[float]:
    z = np.asarray(volume["z"], dtype=float) / bt.um
    x = np.asarray(volume["crop_grid"]["x"], dtype=float) / bt.um
    return [float(z[0]), float(z[-1]), float(x[0]), float(x[-1])]


def _z_relative_to_peak_um(volume: Dict[str, Any]) -> np.ndarray:
    z = np.asarray(volume["z"], dtype=float) / bt.um
    if z.size == 0:
        return z
    peak_index = int(volume.get("peak_index", int(np.argmax(np.asarray(volume.get("peak", []), dtype=float)))))
    peak_index = int(np.clip(peak_index, 0, len(z) - 1))
    return z - float(z[peak_index])


def _representative_plane_index(result: Dict[str, Any]) -> int:
    volume = result["volume"]
    z_um = np.asarray(volume["z"], dtype=float) / bt.um
    if z_um.size == 0:
        return 0
    metrics = result.get("metrics", {})
    peak_idx = int(np.clip(int(volume.get("peak_index", 0)), 0, len(z_um) - 1))
    surface_z = float(metrics.get("surface_z_um", np.nan))
    if np.isfinite(surface_z) and float(z_um[0]) <= surface_z <= float(z_um[-1]):
        return int(np.argmin(np.abs(z_um - surface_z)))
    zone_start = float(metrics.get("canonical_zone_start_um", metrics.get("zone_start_um", np.nan)))
    zone_end = float(metrics.get("canonical_zone_end_um", metrics.get("zone_end_um", np.nan)))
    if np.isfinite(zone_start) and np.isfinite(zone_end) and zone_end > zone_start:
        return int(np.argmin(np.abs(z_um - 0.5 * (zone_start + zone_end))))
    return peak_idx


def _crop_complex_plane(U: np.ndarray, grid: Dict[str, Any], crop_pixels: int) -> tuple[np.ndarray, Dict[str, Any]]:
    field = np.asarray(U, dtype=complex)
    height, width = field.shape
    size = max(1, min(int(crop_pixels), height, width))
    y0 = max(0, (height - size) // 2)
    x0 = max(0, (width - size) // 2)
    field_crop = field[y0 : y0 + size, x0 : x0 + size]

    x = np.asarray(grid["x"], dtype=float)[x0 : x0 + size]
    y = np.asarray(grid.get("y", grid["x"]), dtype=float)[y0 : y0 + size]
    X, Y = np.meshgrid(x, y, indexing="xy")
    dx = float(grid.get("dx", np.median(np.diff(x)))) if len(x) > 1 else float(grid.get("dx", 1.0))
    crop_grid = {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "dx": dx,
        "N": int(size),
        "R": np.hypot(X, Y),
        "Phi": np.arctan2(Y, X),
    }
    return field_crop, crop_grid


def _propagate_complex_plane(
    U0: np.ndarray,
    grid: Dict[str, Any],
    wavelength_m: float,
    z_m: float,
    *,
    n_medium: float,
    method: str,
    propagation_config: bt.PropagationConfig,
) -> tuple[np.ndarray | None, Dict[str, Any] | None]:
    method_key = str(method).lower().strip()
    if method_key != "bl_asm":
        return None, None
    propagator = bt.make_bl_asm_propagator(
        U0,
        grid,
        wavelength_m,
        n_medium=n_medium,
        bandlimit=True,
    )
    return np.asarray(propagator(float(z_m)), dtype=complex), grid


def _build_vector_diagnostic_case(
    U_plus0: np.ndarray,
    grid_plus: Dict[str, Any],
    U_minus0: np.ndarray,
    grid_minus: Dict[str, Any],
    z_m: float,
    design: bt.BeamDesign,
    config: bt.TwinConfig,
    mode: str,
    *,
    vector_program: str = "paper_seed",
    vector_hardware: Any | None = None,
    post_optics: str | Sequence[Any] | None = "paper_qwp",
    vector_method: str = "A",
    future_element: str = "none",
) -> Dict[str, Any] | None:
    plane_plus, plane_grid = _propagate_complex_plane(
        U_plus0,
        grid_plus,
        config.laser.wavelength_m,
        z_m,
        n_medium=config.material.refractive_index,
        method=config.propagation.method,
        propagation_config=config.propagation,
    )
    plane_minus, _ = _propagate_complex_plane(
        U_minus0,
        grid_minus,
        config.laser.wavelength_m,
        z_m,
        n_medium=config.material.refractive_index,
        method=config.propagation.method,
        propagation_config=config.propagation,
    )
    if plane_plus is None or plane_minus is None or plane_grid is None:
        return None

    crop_pixels = min(int(config.grid.crop_pixels), plane_plus.shape[0], plane_plus.shape[1])
    plus_crop, crop_grid = _crop_complex_plane(plane_plus, plane_grid, crop_pixels)
    minus_crop, _ = _crop_complex_plane(plane_minus, plane_grid, crop_pixels)
    program_key = str(vector_program).lower().strip()
    if program_key == "case1_lab":
        return vbb_vector.build_actual_lab_vector_case(
            crop_grid,
            ell=design.ell,
            kr_m_inv=design.kr_sample_m_inv,
            waist_m=design.w0_sample_m,
            target=mode,
            method=vector_method,
            future_element=future_element,
            carrier_lpmm=config.slm.carrier_lpmm,
            z_m=0.0,
            wavelength_m=config.laser.wavelength_m,
            n_medium=config.material.refractive_index,
        )
    if program_key == "slm_encoded":
        return vbb_vector.build_slm_encoded_vector_mode(
            crop_grid,
            plus_crop,
            minus_crop,
            design.ell,
            design.kr_sample_m_inv,
            mode,
            hardware=vector_hardware,
            post_optics=post_optics,
            metadata={"z_um": float(z_m / bt.um)},
        )
    return vbb_vector.build_vector_mode_from_seed(
        crop_grid,
        vbb_vector.VectorField(Ex=plus_crop, Ey=minus_crop),
        design.ell,
        design.kr_sample_m_inv,
        mode,
        post_optics=post_optics,
        metadata={"z_um": float(z_m / bt.um)},
    )


def publication_output_tree(base: str | Path | None = None, folder_name: str = "publication_study") -> Dict[str, Path]:
    """Return the publication output folders inside this study workspace.

    I default to `outputs` so running notebooks from the repo
    root or from the notebook folder writes to the same visible place.
    """

    if base is None:
        base = Path(__file__).resolve().parent / "outputs"
    roots = bt.ensure_output_tree(base)
    out = {
        "base": Path(base),
        "figures": roots["figures"] / folder_name,
        "csv": roots["csv"] / folder_name,
        "json": roots["json"] / folder_name,
        "holograms": roots["holograms"] / folder_name,
    }
    for key in ("figures", "csv", "json", "holograms"):
        out[key].mkdir(parents=True, exist_ok=True)
    return out


def config_from_shortlist_item(base_config: bt.TwinConfig, item: Dict[str, Any]) -> bt.TwinConfig:
    return replace(
        base_config,
        target=replace(
            base_config.target,
            ell=int(item["ell"]),
            target_core_diameter_m=float(item["D_um"]) * bt.um,
            target_bessel_length_m=float(item["L_um"]) * bt.um,
        ),
        energy=replace(base_config.energy, pulse_energy_in_J=float(item["Ein_uJ"]) * bt.uJ),
    )


def energy_budget_dataframe(config: bt.TwinConfig) -> pd.DataFrame:
    input_uJ = float(config.energy.pulse_energy_in_J / bt.uJ)
    study_kind = str(getattr(config, "study_kind", "beam_to_surface")).lower().strip()
    stages = [
        ("Input pulse", 1.0),
        ("Pre-SLM optics", float(config.energy.pre_slm_transmission)),
        ("SLM reflectivity", float(config.energy.slm_reflectivity)),
        ("First-order efficiency", float(config.energy.first_order_efficiency)),
        ("Relay transmission", float(config.energy.relay_transmission)),
        ("Focusing transmission", float(config.energy.focusing_transmission)),
        ("User transmission", float(config.energy.user_extra_transmission)),
    ]
    if study_kind != "beam_to_surface":
        stages.insert(-1, ("Sample surface", float(config.energy.sample_surface_transmission)))
    rows = []
    cumulative = 1.0
    for index, (stage, factor) in enumerate(stages):
        cumulative *= factor
        rows.append(
            {
                "stage_order": index,
                "stage": stage,
                "incremental_transmission": factor,
                "cumulative_transmission": cumulative,
                "pulse_energy_uJ": input_uJ * cumulative,
            }
        )
    return pd.DataFrame(rows)


def metrics_summary_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    metrics = result["metrics"]
    keys = [
        "case_id",
        "path",
        "ell",
        "target_core_diameter_um",
        "target_scale_definition",
        "target_equivalent_l0_core_diameter_um",
        "equivalent_l0_first_zero_radius_um",
        "equivalent_l0_first_zero_diameter_um",
        "vortex_main_ring_radius_um",
        "vortex_main_ring_diameter_um",
        "target_bessel_length_um",
        "gamma_slm_deg",
        "magnification_to_sample",
        "ring_radius_um",
        "ring_diameter_um",
        "core_radius_um",
        "core_radius_definition",
        "core_hwhm_radius_um",
        "core_hwhm_diameter_um",
        "core_first_zero_radius_um",
        "core_first_zero_diameter_um",
        "feature_radius_um",
        "feature_diameter_um",
        "ring_width_um",
        "propagation_method",
        "output_dx_at_peak_um",
        "output_dx_min_um",
        "output_dx_max_um",
        "sas_retained_power_fraction_min",
        "sas_z_limit_min_um",
        "sas_z_over_limit_max",
        "sas_output_magnification_at_peak",
        "bessel_zone_um",
        "canonical_zone_um",
        "canonical_zone_start_um",
        "canonical_zone_end_um",
        "zone_start_um",
        "zone_end_um",
        "bessel_region_um",
        "strict_bessel_region_um",
        "strict_bessel_region_start_um",
        "strict_bessel_region_end_um",
        "bessel_region_start_um",
        "bessel_region_end_um",
        "bessel_region_definition",
        "feature_power_zone_um",
        "radius_stability_zone_um",
        "ring_or_core_radius_drift_fraction_max_in_region",
        "peak_fluence_J_cm2",
        "core_or_ring_peak_fluence_J_cm2",
        "side_lobe_peak_fluence_J_cm2",
        "side_to_core_peak_ratio",
        "pulse_energy_at_sample_uJ",
        "incubated_threshold_J_cm2",
        "line_proxy_threshold_length_um",
        "line_proxy_threshold_width_um",
        "first_order_selected_fraction",
        "qa_status",
        "propagation_power_drift_fraction",
        "propagation_power_label",
        "phase_sampling_label",
        "focal_sampling_label",
        "axial_sampling_label",
        "focal_kt_nyquist_over_kr",
        "focal_samples_per_radial_period",
        "axial_dz_um",
        "hardware_reachable",
        "lab_realizable",
        "simulation_only",
        "requires_element",
        "uses_waveplates",
        "encoded_power_fraction",
        "vector_method",
        "future_element",
    ]
    rows = [{"metric": key, "value": metrics[key]} for key in keys if key in metrics]
    return pd.DataFrame(rows)


def build_case_bundle(
    config: bt.TwinConfig | None = None,
    preset: str = "fast",
    path: bt.PathKind = "realistic",
    case_id: str = "publication_case",
    z_values_m: Sequence[float] | None = None,
) -> Dict[str, Any]:
    config = bt.default_config(preset) if config is None else config
    result = bt.run_case(config, preset=preset, path=path, case_id=case_id, z_values_m=z_values_m)
    plane = np.asarray(result["volume"]["planes"]["peak"], dtype=float)
    crop_grid = result["volume"]["crop_grid"]
    radial = bt.extract_radial_metrics(plane, crop_grid, result["design"].ell, result["design"].kr_sample_m_inv)
    fluence_xy = bt.fluence_from_intensity(plane, crop_grid["dx"], config.energy.pulse_energy_at_sample_J)
    fluence_xz = bt.line_fluence_proxy_xz(
        result["volume"]["xz"],
        crop_grid["dx"],
        config.energy.pulse_energy_at_sample_J,
    )
    return {
        "config": config,
        "preset": preset,
        "case_id": case_id,
        "result": result,
        "radial": radial,
        "fluence_xy": fluence_xy,
        "fluence_xz": fluence_xz,
        "energy_budget": energy_budget_dataframe(config),
        "summary": metrics_summary_dataframe(result),
    }


def build_shortlist_bundles(
    shortlist: Sequence[Dict[str, Any]],
    base_config: bt.TwinConfig,
    preset: str = "fast",
    path: bt.PathKind = "realistic",
) -> list[Dict[str, Any]]:
    bundles = []
    for item in shortlist:
        cfg = config_from_shortlist_item(base_config, item)
        bundle = build_case_bundle(cfg, preset=preset, path=path, case_id=str(item["label"]))
        bundle["spec"] = dict(item)
        bundle["input_energy_uJ"] = float(item["Ein_uJ"])
        bundles.append(bundle)
    return bundles


def _combine_additive_volumes(*volumes: Dict[str, Any]) -> Dict[str, Any]:
    """Combine volumes whose total intensity adds plane by plane."""

    if not volumes:
        raise ValueError("At least one volume is required.")
    ref = volumes[0]
    stack = np.zeros_like(np.asarray(ref["intensity_stack"], dtype=float))
    xz = np.zeros_like(np.asarray(ref["xz"], dtype=float))
    total_power = np.zeros_like(np.asarray(ref["total_power"], dtype=float))
    for volume in volumes:
        stack = stack + np.asarray(volume["intensity_stack"], dtype=float)
        xz = xz + np.asarray(volume["xz"], dtype=float)
        total_power = total_power + np.asarray(volume["total_power"], dtype=float)

    peak = np.max(stack, axis=(1, 2))
    h = int(stack.shape[1] // 2)
    onaxis = stack[:, h, h]
    peak_index = int(np.argmax(peak))
    planes = {
        "start": stack[0].astype(np.float32),
        "middle": stack[len(stack) // 2].astype(np.float32),
        "peak": stack[peak_index].astype(np.float32),
        "end": stack[-1].astype(np.float32),
    }
    return {
        "z": np.asarray(ref["z"], dtype=float),
        "xz": xz.astype(np.float32),
        "intensity_stack": stack.astype(np.float32),
        "peak": peak.astype(float),
        "onaxis": onaxis.astype(float),
        "total_power": total_power.astype(float),
        "planes": planes,
        "peak_index": peak_index,
        "crop_grid": ref["crop_grid"],
        "propagation_method": str(ref.get("propagation_method", "bl_asm")),
        "output_dx_m": np.asarray(ref.get("output_dx_m", []), dtype=float),
        "retained_power_fraction": np.asarray(ref.get("retained_power_fraction", []), dtype=float),
        "native_grids": ref.get("native_grids", []),
        "sas_metadata": ref.get("sas_metadata", []),
    }


def build_vector_case_bundle(
    config: bt.TwinConfig | None = None,
    preset: str = "fast",
    case_id: str = "stage5_vector_total",
    mode: str = "radial",
    z_values_m: Sequence[float] | None = None,
    model: str = "ideal",
    vector_hardware: Any | None = None,
    post_optics: str | Sequence[Any] | None = None,
    vector_method: str = "A",
    future_element: str = "none",
) -> Dict[str, Any]:
    """Build a Stage 5 vector-total bundle on the same Stage 2 metric schema."""

    config = bt.default_config(preset) if config is None else config
    design = bt.compute_design_from_targets(config.laser, config.target, config.material)
    if z_values_m is None:
        z_values = np.linspace(0.0, 1.30 * design.target_bessel_length_m, int(config.grid.axial_points))
    else:
        z_values = np.asarray(z_values_m, dtype=float)
        if z_values.ndim != 1 or z_values.size == 0:
            raise ValueError("z_values_m must be a non-empty 1D sequence.")

    model_key = str(model).lower().strip()
    component_results: Dict[str, Any] = {}

    if model_key in {"ideal", "vector_ideal"}:
        component_source = "ideal"
        vector_program = "paper_seed"
        path = "vector_ideal"
        resolved_vector_hardware: Any | None = "paper_seed"
        resolved_post_optics: str | Sequence[Any] | None = "paper_qwp" if post_optics is None else post_optics
    elif model_key in {"lab_realistic", "realistic", "lab"}:
        component_source = "realistic"
        vector_program = "case1_lab"
        path = "vector_case1_lab"
        resolved_vector_hardware = "case1_same_axis_no_waveplates"
        resolved_post_optics = "none"
    elif model_key in {"slm_encoded", "slm_program", "vector_slm_encoded"}:
        component_source = "realistic"
        vector_program = "case1_lab"
        path = "vector_case1_lab_method_b"
        resolved_vector_hardware = "case1_same_axis_no_waveplates"
        resolved_post_optics = "none"
        vector_method = "B"
    elif model_key in {"ideal_slm_encoded", "ideal_program", "analytic_slm_encoded"}:
        component_source = "ideal"
        vector_program = "slm_encoded"
        path = "vector_ideal_slm_encoded"
        resolved_vector_hardware = "slm_only_symmetric" if vector_hardware is None else vector_hardware
        slm_program = vbb_vector.resolve_slm_encoder_program(
            mode,
            hardware=resolved_vector_hardware,
            post_optics=post_optics,
        )
        resolved_post_optics = slm_program.post_optics
    else:
        raise ValueError(
            "model must be 'ideal', 'lab_realistic', 'slm_encoded', or 'ideal_slm_encoded'."
        )

    post_optics_label = (
        str(resolved_post_optics).lower().strip()
        if isinstance(resolved_post_optics, str)
        else "custom"
    )
    vector_hardware_label = _vector_hardware_label(
        resolved_vector_hardware,
        default="slm_only_symmetric" if vector_program == "slm_encoded" else "paper_seed",
    )

    if vector_program == "slm_encoded":
        slm_program = vbb_vector.resolve_slm_encoder_program(
            mode,
            hardware=resolved_vector_hardware,
            post_optics=resolved_post_optics,
        )
        post_optics_label = slm_program.post_optics_label
        vector_hardware_label = slm_program.hardware
    if vector_program == "case1_lab":
        post_optics_label = "none"
        vector_hardware_label = "case1_same_axis_no_waveplates"

    if component_source == "ideal":
        grid = bt.make_xy_grid(config.grid.ideal_N, config.grid.ideal_dx_m)
        crop_pixels = min(int(config.grid.crop_pixels), int(config.grid.ideal_N))
        design_plus = replace(design, ell=abs(int(design.ell)))
        design_minus = replace(design, ell=-abs(int(design.ell)))
        U_plus = bt.build_conical_axicon_field_ideal(grid, design_plus, config.laser)
        U_minus = bt.build_conical_axicon_field_ideal(grid, design_minus, config.laser)

        volume_plus = bt.propagate_volume(
            U_plus,
            grid,
            config.laser.wavelength_m,
            z_values,
            n_medium=config.material.refractive_index,
            crop_pixels=crop_pixels,
            bandlimit=True,
            method=config.propagation.method,
            propagation_config=config.propagation,
        )
        volume_minus = bt.propagate_volume(
            U_minus,
            grid,
            config.laser.wavelength_m,
            z_values,
            n_medium=config.material.refractive_index,
            crop_pixels=crop_pixels,
            bandlimit=True,
            method=config.propagation.method,
            propagation_config=config.propagation,
        )
        volume = _combine_additive_volumes(volume_plus, volume_minus)
        focal_grid = grid
        first_order_selected_fraction = 1.0
        z_zone_centre_m = 0.0
        interface_zernike_fit: Dict[str, Any] = {}
    else:
        cfg_plus = replace(config, target=replace(config.target, ell=abs(int(config.target.ell))))
        cfg_minus = replace(config, target=replace(config.target, ell=-abs(int(config.target.ell))))
        design_plus = bt.compute_design_from_targets(cfg_plus.laser, cfg_plus.target, cfg_plus.material)
        design_minus = bt.compute_design_from_targets(cfg_minus.laser, cfg_minus.target, cfg_minus.material)
        plus_result = bt.realistic_slm_to_sample(cfg_plus, design_plus, z_values_m=z_values)
        minus_result = bt.realistic_slm_to_sample(cfg_minus, design_minus, z_values_m=z_values)
        volume = _combine_additive_volumes(plus_result["volume"], minus_result["volume"])
        focal_grid = plus_result["focal_grid"]
        first_order_selected_fraction = float(
            np.mean(
                [
                    float(plus_result.get("first_order_selected_fraction", 1.0)),
                    float(minus_result.get("first_order_selected_fraction", 1.0)),
                ]
            )
        )
        z_zone_centre_m = float(
            np.mean(
                [
                    float(plus_result.get("z_zone_centre_m", 0.0)),
                    float(minus_result.get("z_zone_centre_m", 0.0)),
                ]
            )
        )
        interface_zernike_fit = dict(plus_result.get("interface_zernike_fit", {}))
        component_results = {"plus": plus_result, "minus": minus_result}
        U_plus = np.asarray(plus_result["U_focus"], dtype=complex)
        U_minus = np.asarray(minus_result["U_focus"], dtype=complex)
        grid = plus_result["focal_grid"]

    result = {
        "path": path,
        "design": design,
        "focal_grid": focal_grid,
        "volume": volume,
        "first_order_selected_fraction": first_order_selected_fraction,
        "z_zone_centre_m": z_zone_centre_m,
        "interface_zernike_fit": interface_zernike_fit,
        "vector_mode": str(mode),
        "vector_model": model_key,
        "vector_program": vector_program,
        "vector_post_optics": post_optics_label,
        "vector_encoder_hardware": vector_hardware_label,
        "vector_method": str(vector_method).upper(),
        "future_element": str(future_element).lower().strip(),
    }
    metrics = bt.extract_vortex_safe_metrics(result, config)
    qa = bt.sampling_report(config, design, result)
    metrics.update(
        {
            "study": "single_case",
            "case_id": case_id,
            "path": path,
            "material": config.material.name,
            "write_depth_um": config.material.write_depth_m / bt.um,
            "NA": config.objective.NA,
            "phase_bits": config.slm.phase_bits,
            "fill_factor": config.slm.fill_factor,
            "blaze_period_px": config.slm.blaze_period_px,
            "beam_radius_on_slm_mm": config.laser.beam_radius_on_slm_m / bt.mm,
            "hardware_reachable": bt._hardware_reachable(config, design),
            "vector_mode": str(mode),
            "vector_model": model_key,
            "vector_program": vector_program,
            "vector_post_optics": post_optics_label,
            "vector_encoder_hardware": vector_hardware_label,
            "vector_method": str(vector_method).upper(),
            "future_element": str(future_element).lower().strip(),
            **qa,
        }
    )
    result["metrics"] = metrics
    result["sampling_report"] = qa

    diagnostic_idx = _representative_plane_index(result)
    diagnostic_z_m = float(np.asarray(volume["z"], dtype=float)[diagnostic_idx]) if np.asarray(volume["z"]).size else 0.0
    vector_diagnostic = _build_vector_diagnostic_case(
        U_plus,
        grid,
        U_minus,
        grid,
        diagnostic_z_m,
        design,
        config,
        mode,
        vector_program=vector_program,
        vector_hardware=resolved_vector_hardware,
        post_optics=resolved_post_optics,
        vector_method=vector_method,
        future_element=future_element,
    )
    result["vector_diagnostic"] = vector_diagnostic
    if vector_diagnostic is not None:
        result["lab_realizable"] = bool(vector_diagnostic.get("lab_realizable", False))
        result["requires_element"] = vector_diagnostic.get("requires_element")
        result["uses_waveplates"] = bool(vector_diagnostic.get("uses_waveplates", vector_program != "case1_lab"))
        result["simulation_only"] = bool(vector_diagnostic.get("simulation_only", False) or result["uses_waveplates"])
        metrics.update(
            {
                "lab_realizable": result["lab_realizable"],
                "simulation_only": result["simulation_only"],
                "requires_element": result["requires_element"],
                "uses_waveplates": result["uses_waveplates"],
                "encoded_power_fraction": float(vector_diagnostic.get("encoded_power_fraction", np.nan)),
            }
        )

    plane = np.asarray(volume["planes"]["peak"], dtype=float)
    crop_grid = volume["crop_grid"]
    radial = bt.extract_radial_metrics(plane, crop_grid, design.ell, design.kr_sample_m_inv)
    fluence_xy = bt.fluence_from_intensity(plane, crop_grid["dx"], config.energy.pulse_energy_at_sample_J)
    fluence_xz = bt.line_fluence_proxy_xz(volume["xz"], crop_grid["dx"], config.energy.pulse_energy_at_sample_J)
    return {
        "config": config,
        "preset": preset,
        "case_id": case_id,
        "path": path,
        "comparison_label": case_id,
        "comparison_group": "vector",
        "vector_mode": str(mode),
        "vector_model": model_key,
        "vector_program": vector_program,
        "vector_post_optics": post_optics_label,
        "vector_encoder_hardware": vector_hardware_label,
        "vector_method": str(vector_method).upper(),
        "future_element": str(future_element).lower().strip(),
        "result": result,
        "vector_diagnostic": vector_diagnostic,
        "component_results": component_results,
        "radial": radial,
        "fluence_xy": fluence_xy,
        "fluence_xz": fluence_xz,
        "energy_budget": energy_budget_dataframe(config),
        "summary": metrics_summary_dataframe(result),
    }


def build_fidelity_ladder_bundles(
    base_config: bt.TwinConfig | None = None,
    preset: str = "fast",
    *,
    vector_mode: str = "radial",
    vector_model: str = "ideal",
    include_lab_vector: bool = False,
    include_slm_encoded_vector: bool = False,
    include_device: bool = False,
    include_corrected: bool = False,
) -> list[Dict[str, Any]]:
    """Build the consolidated Stage 6 modelling-fidelity ladder.

    The full interface-pre-corrected realistic branch is excluded by default
    because the current SLM carrier/filter model clips most of its first-order
    power, making it a diagnostic branch rather than a fair ladder comparison.
    """

    base = bt.default_config(preset) if base_config is None else base_config
    scalar_variants = [
        (
            "analytic_sample_ideal",
            "analytic ideal",
            replace(base, apply_interface=False, correct_interface=False),
            "ideal",
        ),
        ("lab_realistic", "lab realistic scalar", base, "realistic"),
    ]

    if include_device:
        scalar_variants.insert(
            1,
            (
                "ideal_slm_device",
                "ideal SLM device",
                replace(
                    base,
                    apply_interface=False,
                    correct_interface=False,
                    include_blaze=False,
                    include_first_order_isolation=False,
                    include_quantization=False,
                    include_fill_factor=False,
                    include_active_aperture=False,
                ),
                "realistic",
            ),
        )

    if include_corrected:
        scalar_variants.append(
            ("lab_realistic_corrected", "lab + correction", replace(base, correct_interface=True), "realistic")
        )

    design = bt.compute_design_from_targets(base.laser, base.target, base.material)
    shared_z_max_m = max(2.0 * float(base.grid.axial_range_m), 1.30 * float(design.target_bessel_length_m))
    for case_id, _, cfg, path in scalar_variants:
        probe = bt.run_case(cfg, preset=preset, path=path, case_id=f"{case_id}_z_probe")
        shared_z_max_m = max(shared_z_max_m, float(np.asarray(probe["volume"]["z"], dtype=float)[-1]))
    shared_z_values = np.linspace(0.0, shared_z_max_m, int(base.grid.axial_points))

    bundles: list[Dict[str, Any]] = []
    for order, (case_id, label, cfg, path) in enumerate(scalar_variants):
        bundle = build_case_bundle(cfg, preset=preset, path=path, case_id=case_id, z_values_m=shared_z_values)
        bundle["comparison_label"] = label
        bundle["comparison_order"] = order
        bundle["comparison_group"] = "scalar"
        bundles.append(bundle)

    vector_specs: list[tuple[str, str, str, str | Sequence[Any] | None]] = []
    vector_model_key = str(vector_model).lower().strip()
    if vector_model_key in {"lab_realistic", "realistic", "lab"}:
        vector_specs.append(("vector_stage5_total", "actual Case-1 H-shaped + V-reference", "lab_realistic", "none"))
    elif vector_model_key in {"slm_encoded", "slm_program", "vector_slm_encoded"}:
        vector_specs.append(("vector_stage6_slm_encoded", "actual Case-1 Method B H-shaped + V-reference", "slm_encoded", "none"))
    elif vector_model_key in {"ideal_slm_encoded", "ideal_program", "analytic_slm_encoded"}:
        vector_specs.append(("vector_stage5_total", "vector ideal SLM-encoded", "ideal_slm_encoded", "none"))
    else:
        vector_specs.append(("vector_stage5_total", "vector ideal benchmark", vector_model_key, "paper_qwp"))
        if include_lab_vector:
            vector_specs.append(("vector_stage6_lab_realistic", "actual Case-1 H-shaped + V-reference", "lab_realistic", "none"))
        if include_slm_encoded_vector:
            vector_specs.append(("vector_stage6_slm_encoded", "actual Case-1 Method B H-shaped + V-reference", "slm_encoded", "none"))

    for case_id, label_prefix, model_name, post_optics_name in vector_specs:
        vector_bundle = build_vector_case_bundle(
            base,
            preset=preset,
            case_id=case_id,
            mode=vector_mode,
            z_values_m=shared_z_values,
            model=model_name,
            post_optics=post_optics_name,
        )
        vector_bundle["comparison_label"] = f"{label_prefix} ({vector_mode})"
        vector_bundle["comparison_order"] = len(bundles)
        vector_bundle["comparison_group"] = "vector"
        bundles.append(vector_bundle)
    return bundles


def interface_depth_audit_dataframe(
    base_config: bt.TwinConfig | None = None,
    preset: str = "fast",
    *,
    write_depths_um: Sequence[float] = (50.0, 150.0, 300.0, 600.0),
    measured_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Compare scalar interface-depth predictions against measured depth data when available."""

    base = bt.default_config(preset) if base_config is None else base_config
    measured_df: pd.DataFrame | None = None
    measured_note = "No measured depth CSV provided."
    depth_column: str | None = None
    peak_column: str | None = None

    if measured_csv is not None and Path(measured_csv).exists():
        measured_df = pd.read_csv(measured_csv)
        lower_map = {str(col).lower(): str(col) for col in measured_df.columns}
        for candidate in ("write_depth_um", "depth_um", "nominal_write_depth_um"):
            if candidate in lower_map:
                depth_column = lower_map[candidate]
                break
        for candidate in ("measured_peak_z_um", "measured_focal_depth_um", "measured_focus_z_um"):
            if candidate in lower_map:
                peak_column = lower_map[candidate]
                break
        if depth_column and peak_column:
            measured_note = f"Using measured depth columns '{depth_column}' and '{peak_column}'."
        else:
            measured_note = "Measured CSV found, but no measured peak-depth columns are populated under the current schema."
    elif measured_csv is not None:
        measured_note = f"Measured depth CSV not found: {Path(measured_csv)}"

    rows: list[Dict[str, Any]] = []
    for depth_um in write_depths_um:
        depth_m = float(depth_um) * bt.um
        audit_span_m = max(
            float(base.grid.axial_range_m),
            1.25 * depth_m,
            1.30 * float(base.target.target_bessel_length_m),
        )
        audit_grid = replace(
            base.grid,
            axial_range_m=audit_span_m,
            coarse_scan_factor=max(float(base.grid.coarse_scan_factor), 2.5),
            coarse_scan_points=max(int(base.grid.coarse_scan_points), 21),
        )
        cfg = replace(base, material=replace(base.material, write_depth_m=depth_m), grid=audit_grid)
        bundle = build_case_bundle(cfg, preset=preset, path="realistic", case_id=f"scalar_depth_audit_{depth_um:g}")
        result = bundle["result"]
        metrics = result["metrics"]
        z_values = np.asarray(result["volume"]["z"], dtype=float)
        peak_idx = int(np.clip(int(result["volume"].get("peak_index", 0)), 0, max(len(z_values) - 1, 0)))
        row = {
            "write_depth_um": float(depth_um),
            "predicted_peak_z_um": float(z_values[peak_idx] / bt.um) if z_values.size else np.nan,
            "predicted_zone_start_um": float(metrics.get("zone_start_um", np.nan)),
            "predicted_zone_end_um": float(metrics.get("zone_end_um", np.nan)),
            "predicted_zone_centre_um": float(result.get("z_zone_centre_m", np.nan) / bt.um),
            "first_order_selected_fraction": float(metrics.get("first_order_selected_fraction", np.nan)),
            "measured_peak_z_um": np.nan,
            "depth_residual_um": np.nan,
            "has_measured_depth": False,
        }
        if measured_df is not None and depth_column and peak_column:
            subset = measured_df[[depth_column, peak_column]].dropna()
            if not subset.empty:
                depth_values = subset[depth_column].astype(float).to_numpy()
                match_index = np.where(np.isclose(depth_values, float(depth_um), atol=1.0e-9))[0]
                if match_index.size:
                    measured_peak = float(subset.iloc[int(match_index[0])][peak_column])
                    row["measured_peak_z_um"] = measured_peak
                    row["depth_residual_um"] = row["predicted_peak_z_um"] - measured_peak
                    row["has_measured_depth"] = True
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["audit_status"] = np.where(frame["has_measured_depth"], "measured_compare", "prediction_only")
    frame.attrs["measured_depth_note"] = measured_note
    return frame


def fidelity_ladder_summary_dataframe(bundles: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Return one tidy comparison table for the Stage 6 fidelity ladder."""

    rows = []
    keep = [
        "case_id",
        "path",
        "vector_model",
        "vector_program",
        "vector_post_optics",
        "vector_encoder_hardware",
        "vector_method",
        "future_element",
        "lab_realizable",
        "simulation_only",
        "uses_waveplates",
        "encoded_power_fraction",
        "propagation_method",
        "ring_radius_um",
        "feature_radius_um",
        "feature_diameter_um",
        "bessel_zone_um",
        "canonical_zone_um",
        "zone_start_um",
        "zone_end_um",
        "bessel_region_um",
        "strict_bessel_region_um",
        "bessel_region_start_um",
        "bessel_region_end_um",
        "peak_fluence_J_cm2",
        "side_to_core_peak_ratio",
        "first_order_selected_fraction",
        "qa_status",
    ]
    for bundle in bundles:
        metrics = bundle["result"]["metrics"]
        row = {key: metrics.get(key) for key in keep}
        row["comparison_label"] = bundle.get("comparison_label", bundle["case_id"])
        row["comparison_group"] = bundle.get("comparison_group", "scalar")
        row["comparison_order"] = bundle.get("comparison_order", 0)
        row["vector_mode"] = bundle.get("vector_mode", metrics.get("vector_mode", ""))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("comparison_order").reset_index(drop=True)


def fidelity_delta_table(
    summary: pd.DataFrame,
    *,
    reference_case: str = "analytic_sample_ideal",
) -> pd.DataFrame:
    """Compute the realism-penalty delta table relative to the analytic ideal."""

    baseline = summary.loc[summary["case_id"] == reference_case]
    if baseline.empty:
        raise ValueError(f"Reference case '{reference_case}' not found in summary.")
    ref = baseline.iloc[0]
    specs = [
        ("canonical_zone_um", "canonical_zone_shift_um", "canonical_zone_shift_pct"),
        ("strict_bessel_region_um", "strict_bessel_region_shift_um", "strict_bessel_region_shift_pct"),
        ("ring_radius_um", "ring_radius_shift_um", "ring_radius_shift_pct"),
        ("peak_fluence_J_cm2", "peak_fluence_shift_J_cm2", "peak_fluence_shift_pct"),
        ("side_to_core_peak_ratio", "side_to_core_shift", "side_to_core_shift_pct"),
    ]
    rows = []
    for _, item in summary.iterrows():
        row = {
            "case_id": item["case_id"],
            "comparison_label": item["comparison_label"],
            "comparison_group": item["comparison_group"],
            "qa_status": item.get("qa_status"),
        }
        for metric, abs_name, pct_name in specs:
            current = float(item[metric])
            base = float(ref[metric])
            row[abs_name] = current - base
            row[pct_name] = 100.0 * (current - base) / (abs(base) + bt.EPS)
        row["zone_shift_um"] = row["canonical_zone_shift_um"]
        row["zone_shift_pct"] = row["canonical_zone_shift_pct"]
        row["bessel_region_shift_um"] = row["strict_bessel_region_shift_um"]
        row["bessel_region_shift_pct"] = row["strict_bessel_region_shift_pct"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fidelity_ladder(
    bundles: Sequence[Dict[str, Any]],
    *,
    title: str | None = None,
) -> plt.Figure:
    """Plot matched XY peak planes and forward-only XZ sections for the fidelity ladder."""

    if not bundles:
        raise ValueError("At least one bundle is required.")
    vbb_style.apply_style()

    ordered = sorted(bundles, key=lambda bundle: int(bundle.get("comparison_order", 0)))
    fig, axes = plt.subplots(2, len(ordered), figsize=(3.55 * len(ordered), 6.8), constrained_layout=True)
    if len(ordered) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    xy_half = min(
        float(np.max(np.abs(np.asarray(bundle["result"]["volume"]["crop_grid"]["x"], dtype=float) / bt.um)))
        for bundle in ordered
    )
    z_max_um = max(float(np.asarray(bundle["result"]["volume"]["z"], dtype=float)[-1] / bt.um) for bundle in ordered)
    xy_artist = None
    xz_artist = None
    for col, bundle in enumerate(ordered):
        result = bundle["result"]
        volume = result["volume"]
        metrics = result["metrics"]
        plane_idx = _representative_plane_index(result)
        zone_start = float(metrics.get("zone_start_um", np.nan))
        zone_end = float(metrics.get("zone_end_um", np.nan))
        plane = np.asarray(volume["intensity_stack"][plane_idx], dtype=float)

        xy_artist = axes[0, col].imshow(
            vbb_style.display_scale(plane, gamma=0.45),
            origin="lower",
            extent=_xy_extent_um(volume["crop_grid"]),
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, col].set_title(bundle.get("comparison_label", bundle["case_id"]))
        axes[0, col].set_xlim(-xy_half, xy_half)
        axes[0, col].set_ylim(-xy_half, xy_half)
        axes[0, col].set_xlabel("x [um, sample plane]")
        axes[0, col].set_ylabel("y [um, sample plane]")
        axes[0, col].grid(False)

        xz_artist = axes[1, col].imshow(
            vbb_style.display_scale(np.asarray(volume["xz"], dtype=float), gamma=0.45),
            origin="lower",
            aspect="auto",
            extent=_xz_extent_um(volume),
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        if np.isfinite(zone_start) and np.isfinite(zone_end) and zone_end > zone_start:
            axes[1, col].axvspan(zone_start, zone_end, color="#f2c14e", alpha=0.12)
            axes[1, col].axvline(zone_start, color="#a56600", linestyle="--", linewidth=1.0)
            axes[1, col].axvline(zone_end, color="#a56600", linestyle="--", linewidth=1.0)
        axes[1, col].set_xlim(0.0, z_max_um)
        axes[1, col].set_ylim(-xy_half, xy_half)
        axes[1, col].set_xlabel("z [um, sample plane]")
        axes[1, col].set_ylabel("x [um, sample plane]")
        axes[1, col].grid(False)

    if xy_artist is not None:
        cbar_xy = fig.colorbar(xy_artist, ax=axes[0, :], shrink=0.92, pad=0.01)
        cbar_xy.set_label("display intensity [a.u.]")
    if xz_artist is not None:
        cbar_xz = fig.colorbar(xz_artist, ax=axes[1, :], shrink=0.92, pad=0.01)
        cbar_xz.set_label("display intensity [a.u.]")
    fig.suptitle(title or "Stage 6 fidelity ladder", fontsize=12)
    return fig


def plot_phase_mask_comparison(
    realistic_bundle: Dict[str, Any],
    corrected_bundle: Dict[str, Any],
    *,
    title: str | None = None,
) -> plt.Figure:
    """Plot matched device phase masks for the lab and corrected variants."""

    vbb_style.apply_style()
    phase_rows = []
    for bundle in (realistic_bundle, corrected_bundle):
        slm_field = bundle["result"].get("slm_field")
        if slm_field is None:
            raise ValueError("Phase-mask comparison requires realistic bundles with slm_field data.")
        phase = np.asarray(slm_field.get("phase_wrapped", slm_field["phase"]), dtype=float)
        grid = slm_field.get("rect_grid", slm_field["grid"])
        phase_rows.append((bundle.get("comparison_label", bundle["case_id"]), phase, grid))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), constrained_layout=True)
    artist = None
    for axis, (label, phase, grid) in zip(axes, phase_rows):
        extent = [
            float(np.asarray(grid["x"], dtype=float)[0] / bt.mm),
            float(np.asarray(grid["x"], dtype=float)[-1] / bt.mm),
            float(np.asarray(grid.get("y", grid["x"]), dtype=float)[0] / bt.mm),
            float(np.asarray(grid.get("y", grid["x"]), dtype=float)[-1] / bt.mm),
        ]
        artist = axis.imshow(phase, origin="lower", extent=extent, cmap=vbb_style.PHASE_CMAP)
        axis.set_title(label)
        axis.set_xlabel("x [mm, SLM plane]")
        axis.set_ylabel("y [mm, SLM plane]")
        axis.grid(False)
    if artist is not None:
        cbar = fig.colorbar(artist, ax=axes, shrink=0.92, pad=0.02)
        cbar.set_label("wrapped phase [rad]")
    fig.suptitle(title or "Device phase masks: lab realistic versus corrected", fontsize=12)
    return fig


def plot_fidelity_delta_table(
    delta_table: pd.DataFrame,
    *,
    title: str | None = None,
) -> plt.Figure:
    """Render the Stage 6 realism-penalty table as a compact figure."""

    vbb_style.apply_style()
    display = delta_table[
        [
            "comparison_label",
            "canonical_zone_shift_pct",
            "ring_radius_shift_pct",
            "peak_fluence_shift_pct",
            "side_to_core_shift_pct",
        ]
    ].copy()
    display.columns = [
        "variant",
        "canonical zone shift [%]",
        "ring-radius shift [%]",
        "peak-fluence shift [%]",
        "side/core shift [%]",
    ]
    for column in display.columns[1:]:
        display[column] = display[column].map(lambda value: f"{float(value):+.2f}")

    fig_height = max(2.6, 0.52 * (len(display) + 2))
    fig, ax = plt.subplots(1, 1, figsize=(10.6, fig_height), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#fafafa")
        cell.set_edgecolor("#d9d9d9")
    ax.set_title(title or "Stage 6 realism penalty relative to the analytic ideal")
    return fig


def shortlist_summary_dataframe(bundles: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    keep = [
        "case_id",
        "path",
        "ell",
        "target_core_diameter_um",
        "target_scale_definition",
        "target_equivalent_l0_core_diameter_um",
        "equivalent_l0_first_zero_radius_um",
        "equivalent_l0_first_zero_diameter_um",
        "vortex_main_ring_radius_um",
        "vortex_main_ring_diameter_um",
        "target_bessel_length_um",
        "gamma_slm_deg",
        "magnification_to_sample",
        "ring_radius_um",
        "ring_diameter_um",
        "core_radius_um",
        "core_radius_definition",
        "core_hwhm_radius_um",
        "core_hwhm_diameter_um",
        "core_first_zero_radius_um",
        "core_first_zero_diameter_um",
        "feature_radius_um",
        "feature_diameter_um",
        "ring_width_um",
        "propagation_method",
        "output_dx_at_peak_um",
        "sas_retained_power_fraction_min",
        "sas_z_over_limit_max",
        "pulse_energy_at_sample_uJ",
        "peak_fluence_J_cm2",
        "incubated_threshold_J_cm2",
        "side_to_core_peak_ratio",
        "bessel_zone_um",
        "canonical_zone_um",
        "canonical_zone_start_um",
        "canonical_zone_end_um",
        "bessel_region_um",
        "strict_bessel_region_um",
        "strict_bessel_region_start_um",
        "strict_bessel_region_end_um",
        "bessel_region_start_um",
        "bessel_region_end_um",
        "bessel_region_definition",
        "line_proxy_threshold_length_um",
        "qa_status",
        "propagation_power_drift_fraction",
        "propagation_power_label",
        "phase_sampling_label",
        "focal_sampling_label",
        "axial_sampling_label",
        "focal_kt_nyquist_over_kr",
    ]
    for bundle in bundles:
        metrics = bundle["result"]["metrics"]
        row = {key: metrics.get(key) for key in keep}
        if "input_energy_uJ" in bundle:
            row["input_energy_uJ"] = bundle["input_energy_uJ"]
        # Stamp with canonical schema metadata if not already present.
        annotate_scalar_row(row, qa_status=str(metrics.get("qa_status", "exploratory")))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_heatmap(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str,
    cbar_label: str,
    cmap: str = vbb_style.INTENSITY_CMAP,
) -> plt.Figure:
    pivot = df.pivot(index=index, columns=columns, values=values).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)), [str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(cbar_label)
    return fig


def thin_obstacle_mask(
    grid: Dict[str, Any],
    kind: str = "disk",
    radius_m: float | None = None,
    width_m: float | None = None,
    height_m: float | None = None,
    offset_x_m: float = 0.0,
    offset_y_m: float = 0.0,
) -> np.ndarray:
    X = np.asarray(grid["X"], dtype=float) - float(offset_x_m)
    Y = np.asarray(grid["Y"], dtype=float) - float(offset_y_m)
    dx = float(grid["dx"])
    key = str(kind).lower().strip()
    if key == "disk":
        radius = max(float(radius_m if radius_m is not None else 1.5 * bt.um), dx)
        blocked = X**2 + Y**2 <= radius**2
    elif key == "strip":
        width = max(float(width_m if width_m is not None else 2.0 * bt.um), dx)
        blocked = np.abs(X) <= 0.5 * width
    elif key == "square":
        width = max(float(width_m if width_m is not None else 2.0 * bt.um), dx)
        height = max(float(height_m if height_m is not None else width), dx)
        blocked = (np.abs(X) <= 0.5 * width) & (np.abs(Y) <= 0.5 * height)
    elif key == "half_plane":
        blocked = X >= 0.0
    else:
        raise ValueError(f"Unsupported obstacle kind: {kind}")
    return (~blocked).astype(float)


def build_self_healing_bundle(
    config: bt.TwinConfig | None = None,
    preset: str = "fast",
    path: bt.PathKind = "ideal",
    case_id: str = "self_healing_case",
    obstacle_kind: str = "disk",
    obstacle_z_m: float | None = None,
    obstacle_radius_m: float | None = None,
    obstacle_width_m: float | None = None,
    obstacle_height_m: float | None = None,
    offset_x_m: float = 0.0,
    offset_y_m: float = 0.0,
    span_after_m: float | None = None,
    axial_points: int | None = None,
) -> Dict[str, Any]:
    config = bt.default_config(preset) if config is None else config
    design = bt.compute_design_from_targets(config.laser, config.target, config.material)

    if path == "ideal":
        grid = bt.make_xy_grid(config.grid.ideal_N, config.grid.ideal_dx_m)
        U_start = bt.build_conical_axicon_field_ideal(grid, design, config.laser)
        prep = None
    else:
        prep = bt.realistic_slm_to_sample(config, design)
        grid = prep["focal_grid"]
        U_start = prep["U_focus"]

    obstacle_z_m = float(0.35 * design.target_bessel_length_m if obstacle_z_m is None else obstacle_z_m)
    span_after_m = float(1.2 * design.target_bessel_length_m if span_after_m is None else span_after_m)
    axial_points = int(config.grid.axial_points if axial_points is None else axial_points)
    z_relative = np.linspace(0.0, span_after_m, axial_points)

    propagator = bt.make_bl_asm_propagator(
        U_start,
        grid,
        config.laser.wavelength_m,
        n_medium=config.material.refractive_index,
        bandlimit=True,
        include_evanescent=True,
    )
    U_at_obstacle = propagator(obstacle_z_m)
    mask = thin_obstacle_mask(
        grid,
        kind=obstacle_kind,
        radius_m=obstacle_radius_m,
        width_m=obstacle_width_m,
        height_m=obstacle_height_m,
        offset_x_m=offset_x_m,
        offset_y_m=offset_y_m,
    )
    U_obstructed = U_at_obstacle * mask

    reference_volume = bt.propagate_volume(
        U_at_obstacle,
        grid,
        config.laser.wavelength_m,
        z_relative,
        n_medium=config.material.refractive_index,
        crop_pixels=config.grid.crop_pixels,
        bandlimit=True,
    )
    obstructed_volume = bt.propagate_volume(
        U_obstructed,
        grid,
        config.laser.wavelength_m,
        z_relative,
        n_medium=config.material.refractive_index,
        crop_pixels=config.grid.crop_pixels,
        bandlimit=True,
    )
    peak_recovery = np.divide(
        np.asarray(obstructed_volume["peak"], dtype=float),
        np.asarray(reference_volume["peak"], dtype=float) + bt.EPS,
    )
    onaxis_recovery = np.divide(
        np.asarray(obstructed_volume["onaxis"], dtype=float),
        np.asarray(reference_volume["onaxis"], dtype=float) + bt.EPS,
    )
    metrics = {
        "case_id": case_id,
        "path": path,
        "obstacle_kind": obstacle_kind,
        "obstacle_z_um": obstacle_z_m / bt.um,
        "blocked_fraction": float(1.0 - np.mean(mask)),
        "peak_recovery_end": float(peak_recovery[-1]),
        "peak_recovery_max": float(np.max(peak_recovery)),
        "peak_recovery_mean": float(np.mean(peak_recovery)),
        "onaxis_recovery_end": float(onaxis_recovery[-1]),
        "target_bessel_length_um": design.target_bessel_length_m / bt.um,
        "ell": int(design.ell),
    }
    return {
        "config": config,
        "preset": preset,
        "case_id": case_id,
        "path": path,
        "design": design,
        "grid": grid,
        "prep": prep,
        "obstacle_kind": obstacle_kind,
        "obstacle_z_m": obstacle_z_m,
        "z_relative": z_relative,
        "mask": mask,
        "reference_plane": np.abs(U_at_obstacle) ** 2,
        "obstructed_plane": np.abs(U_obstructed) ** 2,
        "reference_volume": reference_volume,
        "obstructed_volume": obstructed_volume,
        "peak_recovery": peak_recovery,
        "onaxis_recovery": onaxis_recovery,
        "metrics": metrics,
    }


def self_healing_summary_dataframe(bundles: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([bundle["metrics"] for bundle in bundles])


def plot_self_healing_diagnostics(bundle: Dict[str, Any], title: str | None = None) -> plt.Figure:
    reference_volume = bundle["reference_volume"]
    obstructed_volume = bundle["obstructed_volume"]
    z_um = np.asarray(bundle["z_relative"], dtype=float) / bt.um
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ax[0, 0].imshow(
        _sqrt_normalise(bundle["reference_plane"]),
        origin="lower",
        extent=_xy_extent_um(bundle["grid"]),
        cmap="inferno",
    )
    ax[0, 0].set_title("Reference plane at obstacle")
    ax[0, 0].set_xlabel("x [um, sample plane]")
    ax[0, 0].set_ylabel("y [um, sample plane]")

    ax[0, 1].imshow(
        _sqrt_normalise(bundle["obstructed_plane"]),
        origin="lower",
        extent=_xy_extent_um(bundle["grid"]),
        cmap="inferno",
    )
    ax[0, 1].set_title("Obstructed plane")
    ax[0, 1].set_xlabel("x [um, sample plane]")
    ax[0, 1].set_ylabel("y [um, sample plane]")

    ax[0, 2].imshow(
        _sqrt_normalise(obstructed_volume["xz"]),
        origin="lower",
        aspect="auto",
        extent=[float(z_um[0]), float(z_um[-1]), float(obstructed_volume["crop_grid"]["x"][0] / bt.um), float(obstructed_volume["crop_grid"]["x"][-1] / bt.um)],
        cmap="inferno",
    )
    ax[0, 2].set_title("Obstructed XZ after obstacle")
    ax[0, 2].set_xlabel("z after obstacle [um, sample plane]")
    ax[0, 2].set_ylabel("x [um, sample plane]")

    ax[1, 0].plot(z_um, _normalise(reference_volume["peak"]), label="reference peak")
    ax[1, 0].plot(z_um, _normalise(obstructed_volume["peak"]), label="obstructed peak")
    ax[1, 0].plot(z_um, _normalise(reference_volume["onaxis"]), "--", label="reference on axis")
    ax[1, 0].plot(z_um, _normalise(obstructed_volume["onaxis"]), "--", label="obstructed on axis")
    ax[1, 0].set_title("Axial intensity traces")
    ax[1, 0].set_xlabel("z after obstacle [um, sample plane]")
    ax[1, 0].set_ylabel("normalised intensity")
    ax[1, 0].legend(frameon=False)

    ax[1, 1].plot(z_um, bundle["peak_recovery"], label="peak recovery")
    ax[1, 1].plot(z_um, bundle["onaxis_recovery"], label="on-axis recovery")
    ax[1, 1].axhline(1.0, color="0.5", linewidth=0.8)
    ax[1, 1].set_title("Recovery ratio")
    ax[1, 1].set_xlabel("z after obstacle [um, sample plane]")
    ax[1, 1].set_ylabel("obstructed / reference")
    ax[1, 1].legend(frameon=False)

    ax[1, 2].imshow(
        _sqrt_normalise(obstructed_volume["planes"]["end"]),
        origin="lower",
        extent=_xy_extent_um(obstructed_volume["crop_grid"]),
        cmap="inferno",
    )
    ax[1, 2].set_title("End plane after recovery")
    ax[1, 2].set_xlabel("x [um, sample plane]")
    ax[1, 2].set_ylabel("y [um, sample plane]")

    fig.suptitle(
        title
        or f"{bundle['case_id']} | {bundle['obstacle_kind']} obstacle at {bundle['metrics']['obstacle_z_um']:.1f} um",
        fontsize=13,
    )
    return fig


def export_case_report_artifacts(
    bundle: Dict[str, Any],
    figure_dir: str | Path,
    csv_dir: str | Path,
) -> Dict[str, Path]:
    figure_root = Path(figure_dir)
    csv_root = Path(csv_dir)
    figure_root.mkdir(parents=True, exist_ok=True)
    csv_root.mkdir(parents=True, exist_ok=True)
    safe_case = str(bundle["case_id"]).replace(" ", "_")

    summary_path = csv_root / f"{safe_case}_summary.csv"
    energy_path = csv_root / f"{safe_case}_energy_budget.csv"
    diag_path = figure_root / f"{safe_case}_diagnostics.png"
    montage_path = figure_root / f"{safe_case}_plane_montage.png"

    run_id = new_csv_run_id()
    add_csv_metadata(bundle["summary"], preset=bundle.get("preset"), run_id=run_id).to_csv(summary_path, index=False)
    add_csv_metadata(bundle["energy_budget"], preset=bundle.get("preset"), run_id=run_id).to_csv(energy_path, index=False)

    fig1 = plot_case_diagnostics(bundle)
    fig1.savefig(diag_path, dpi=220, bbox_inches="tight")
    plt.close(fig1)
    fig2 = plot_axial_plane_montage(bundle)
    fig2.savefig(montage_path, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    return {
        "summary_csv": summary_path,
        "energy_budget_csv": energy_path,
        "diagnostics_png": diag_path,
        "plane_montage_png": montage_path,
    }


def plot_case_diagnostics(bundle: Dict[str, Any], title: str | None = None) -> plt.Figure:
    result = bundle["result"]
    volume = result["volume"]
    radial = bundle["radial"]
    config = bundle["config"]
    metrics = result["metrics"]

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    if "slm_field" in result:
        slm_field = result["slm_field"]
        phase = np.asarray(slm_field.get("phase_wrapped", slm_field["phase"]), dtype=float)
        phase_grid = slm_field.get("rect_grid", slm_field["grid"])
        ax[0, 0].imshow(phase, origin="lower", extent=_xy_extent_mm(phase_grid), cmap="twilight")
        ax[0, 0].set_title("SLM wrapped phase")
        ax[0, 0].set_xlabel("x [mm, SLM plane]")
        ax[0, 0].set_ylabel("y [mm, SLM plane]")
    else:
        ax[0, 0].axis("off")
        ax[0, 0].set_title("Ideal path has no device phase")

    representative_idx = _representative_plane_index(result)
    z_um = np.asarray(volume["z"], dtype=float) / bt.um
    representative_z_um = float(z_um[representative_idx]) if z_um.size else float("nan")
    if "intensity_stack" in volume and len(volume["intensity_stack"]) > representative_idx:
        peak_plane = np.asarray(volume["intensity_stack"][representative_idx], dtype=float)
    else:
        peak_plane = np.asarray(volume["planes"]["peak"], dtype=float)
    ax[0, 1].imshow(
        _sqrt_normalise(peak_plane),
        origin="lower",
        extent=_xy_extent_um(volume["crop_grid"]),
        cmap="inferno",
    )
    ax[0, 1].set_title(f"Surface/zone plane XY, z={representative_z_um:.1f} um")
    ax[0, 1].set_xlabel("x [um, surface-in-air plane]")
    ax[0, 1].set_ylabel("y [um, surface-in-air plane]")

    ax[0, 2].imshow(
        _sqrt_normalise(volume["xz"]),
        origin="lower",
        aspect="auto",
        extent=_xz_extent_um(volume),
        cmap="inferno",
    )
    ax[0, 2].set_title("Surface-in-air XZ, sqrt(I)")
    ax[0, 2].set_xlabel("z [um, surface-in-air plane]")
    ax[0, 2].set_ylabel("x [um, surface-in-air plane]")

    peak_trace = _normalise(np.asarray(volume["peak"], dtype=float))
    onaxis_trace = _normalise(np.asarray(volume["onaxis"], dtype=float))
    ax[1, 0].plot(z_um, peak_trace, label="peak in plane")
    ax[1, 0].plot(z_um, onaxis_trace, "--", label="on axis")
    ax[1, 0].axhline(0.5, color="0.5", linewidth=0.8)
    region_start = float(metrics.get("canonical_zone_start_um", metrics.get("bessel_region_start_um", np.nan)))
    region_end = float(metrics.get("canonical_zone_end_um", metrics.get("bessel_region_end_um", np.nan)))
    if np.isfinite(region_start) and np.isfinite(region_end) and region_end > region_start:
        ax[1, 0].axvspan(
            region_start,
            region_end,
            color="#009E73",
            alpha=0.14,
            label="canonical FWHM zone",
        )
    surface_z_um = float(metrics.get("surface_z_um", np.nan))
    if np.isfinite(surface_z_um):
        ax[1, 0].axvline(surface_z_um, color="tab:purple", linewidth=0.9, linestyle="-.", label="surface plane")
    if np.isfinite(metrics.get("zone_start_um", np.nan)):
        ax[1, 0].axvline(metrics["zone_start_um"], color="tab:green", linewidth=0.8, label="peak FWHM zone")
    if np.isfinite(metrics.get("zone_end_um", np.nan)):
        ax[1, 0].axvline(metrics["zone_end_um"], color="tab:green", linewidth=0.8)
    ax[1, 0].set_title("Axial observables")
    ax[1, 0].set_xlabel("z [um, surface-in-air plane]")
    ax[1, 0].set_ylabel("normalised intensity")
    ax[1, 0].legend(frameon=False)

    radius_um = np.asarray(radial["r_profile_m"], dtype=float) / bt.um
    raw_profile = np.asarray(radial["radial_profile_norm"], dtype=float)
    smooth_profile = np.asarray(radial["radial_profile_smooth"], dtype=float)
    ax[1, 1].plot(radius_um, raw_profile, alpha=0.35, label="raw")
    ax[1, 1].plot(radius_um, smooth_profile, linewidth=2.0, label="smoothed")
    ax[1, 1].axvline(radial["feature_radius_m"] / bt.um, color="tab:red", linestyle="--", label="feature")
    ax[1, 1].axvline(radial["r_half_inner_m"] / bt.um, color="tab:blue", linestyle=":")
    ax[1, 1].axvline(radial["r_half_outer_m"] / bt.um, color="tab:blue", linestyle=":")
    ax[1, 1].set_title("Radial profile")
    ax[1, 1].set_xlabel("r [um, surface-in-air plane]")
    ax[1, 1].set_ylabel("normalised intensity")
    ax[1, 1].legend(frameon=False)

    budget = bundle["energy_budget"]
    ax[1, 2].plot(budget["stage_order"], budget["pulse_energy_uJ"], marker="o")
    ax[1, 2].set_xticks(budget["stage_order"], budget["stage"], rotation=30, ha="right")
    ax[1, 2].set_title("Energy budget")
    ax[1, 2].set_ylabel("pulse energy [uJ]")
    ax[1, 2].grid(alpha=0.25)

    fig.suptitle(
        title
        or f"{bundle['case_id']} | {result['path']} | ell={metrics['ell']} | zone={metrics.get('canonical_zone_um', metrics['bessel_zone_um']):.1f} um",
        fontsize=13,
    )
    return fig


def plot_axial_plane_montage(
    bundle: Dict[str, Any],
    plane_keys: Sequence[str] = ("start", "middle", "peak", "end"),
    title: str | None = None,
) -> plt.Figure:
    result = bundle["result"]
    volume = result["volume"]
    z_um = np.asarray(volume["z"], dtype=float) / bt.um
    plane_z = {
        "start": float(z_um[0]),
        "middle": float(z_um[len(z_um) // 2]),
        "peak": float(z_um[int(volume["peak_index"])]),
        "end": float(z_um[-1]),
    }
    available = [key for key in plane_keys if key in volume["planes"]]
    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 3.8), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]
    extent = _xy_extent_um(volume["crop_grid"])
    for axis, key in zip(axes, available):
        axis.imshow(_sqrt_normalise(volume["planes"][key]), origin="lower", extent=extent, cmap="inferno")
        axis.set_title(f"{key} plane\nz={plane_z[key]:.1f} um")
        axis.set_xlabel("x [um, sample plane]")
        axis.set_ylabel("y [um, sample plane]")
    fig.suptitle(title or f"{bundle['case_id']} plane evolution", fontsize=13)
    return fig


__all__ = [
    "DEFAULT_SHORTLIST",
    "SCALAR_ENGINE_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "add_csv_metadata",
    "build_case_bundle",
    "build_fidelity_ladder_bundles",
    "build_self_healing_bundle",
    "build_shortlist_bundles",
    "build_vector_case_bundle",
    "config_from_shortlist_item",
    "energy_budget_dataframe",
    "export_case_report_artifacts",
    "fidelity_delta_table",
    "fidelity_ladder_summary_dataframe",
    "interface_depth_audit_dataframe",
    "metrics_summary_dataframe",
    "new_csv_run_id",
    "plot_axial_plane_montage",
    "plot_case_diagnostics",
    "plot_fidelity_delta_table",
    "plot_fidelity_ladder",
    "plot_metric_heatmap",
    "plot_phase_mask_comparison",
    "plot_self_healing_diagnostics",
    "publication_output_tree",
    "self_healing_summary_dataframe",
    "shortlist_summary_dataframe",
    "thin_obstacle_mask",
]
