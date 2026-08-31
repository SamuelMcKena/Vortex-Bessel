"""Physics-separated q=20 correction benchmark v3.

v2 established that lateral axicon displacement is identifiable from the beam
trajectory, but also showed that a Gaussian beam-radius error is *not*
identifiable from independently normalized transverse morphology.  That is the
expected physics: for an axicon/Bessel beam, propagation distance maps to input
radius (rho ~= z tan(alpha)), so the Gaussian illumination radius is carried
primarily by the longitudinal Bessel intensity envelope.

v3 therefore estimates physical quantities from observables that retain the
information each quantity actually controls:

* input Gaussian radius: longitudinal high-intensity Bessel envelope versus z;
  only one global normalization is applied across the complete scan;
* Fourier-iris radius: azimuthally averaged transverse morphology;
* lateral alignment: centroid trajectory through z.

The same Miao retrieval is then applied both directly and after physical-system
compensation.  An exact field-ratio phase remains an oracle diagnostic only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import benchmark_q20_miao_vs_digital_twin as v1  # noqa: E402
import benchmark_q20_method_physics_v2 as v2  # noqa: E402
from vbb_study.digital_twin.hierarchical_physical_fit import hierarchical_physical_fit  # noqa: E402
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT = ROOT / "outputs" / "poster" / "q20_method_comparison_v3"
Z_FIT_M = np.linspace(28e-3, 138e-3, 15)
Z_DISPLAY_M = np.linspace(28e-3, 138e-3, 111)
GRID_N = 512
WINDOW_M = 8.0e-3
NOISE_SIGMA = 0.0015
SEED = 2031
FIT_CROP_HALF_M = 1.65e-3
EPS = np.finfo(float).tiny

# Keep imported helpers on one numerical/physical contract.
v1.GRID_N = GRID_N
v1.WINDOW_M = WINDOW_M
v1.Z_FIT_M = Z_FIT_M
v1.Z_DISPLAY_M = Z_DISPLAY_M
v1.NOISE_SIGMA = NOISE_SIGMA
v1.FIT_CROP_HALF_M = FIT_CROP_HALF_M
v1.METRIC_RADIUS_M = v2.METRIC_RADIUS_M
v2.Z_FIT_M = Z_FIT_M
v2.Z_DISPLAY_M = Z_DISPLAY_M


@dataclass(frozen=True)
class OneParameterFit:
    family: str
    values: tuple[float, ...]
    costs: tuple[float, ...]
    best_value: float
    best_cost: float
    nominal_cost: float

    @property
    def improvement_fraction(self) -> float:
        return float((self.nominal_cost-self.best_cost)/max(self.nominal_cost, EPS))

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "values": list(self.values),
            "costs": list(self.costs),
            "best_value": self.best_value,
            "best_cost": self.best_cost,
            "nominal_cost": self.nominal_cost,
            "improvement_fraction": self.improvement_fraction,
        }


def _top_intensity_envelope(stack: np.ndarray, *, top_fraction: float = 0.012) -> np.ndarray:
    """Return a robust Bessel brightness envelope, preserving relative z power.

    A small upper fraction of pixels is averaged in every plane.  Unlike the
    transverse morphology metrics, planes are *not* independently normalized.
    This quantity follows the bright q=20 Bessel rings while being less sensitive
    than a single peak pixel to camera noise and sub-pixel ring placement.

    A single global normalization removes unknown overall laser/camera gain but
    preserves the longitudinal envelope.  Experimental use therefore requires
    fixed exposure/gain over the z scan, which is already how a correction stack
    should be acquired.
    """
    a = np.maximum(np.asarray(stack, float), 0.0)
    n = a.shape[-1]*a.shape[-2]
    k = max(8, int(round(float(top_fraction)*n)))
    env = []
    for p in a:
        flat = p.ravel()
        # partition avoids sorting the complete plane.
        hi = np.partition(flat, flat.size-k)[-k:]
        env.append(float(np.mean(hi)))
    env = np.asarray(env, float)
    return env/max(float(np.max(env)), EPS)


def longitudinal_envelope_error(model: np.ndarray, data: np.ndarray) -> float:
    """RMSE between globally normalized Bessel brightness envelopes versus z."""
    m = _top_intensity_envelope(model)
    d = _top_intensity_envelope(data)
    return float(np.sqrt(np.mean((m-d)**2)))


def _fit_single_registry_parameter(target: np.ndarray, simulator, initial: SystemErrorConfig,
                                   family: str, registry: dict) -> tuple[SystemErrorConfig, OneParameterFit]:
    entry = registry[family]
    vals = tuple(float(v) for v in entry["values"])
    costs = []
    configs = []
    for value in vals:
        cfg = v1.apply_registry_family(initial, family, value, registry=registry)
        configs.append(cfg)
        costs.append(longitudinal_envelope_error(simulator(cfg), target))
    costs_a = np.asarray(costs, float)
    ib = int(np.argmin(costs_a))
    # Registry values include the nominal value 1.0 for beam radius.
    inom = int(np.argmin(np.abs(np.asarray(vals)-1.0)))
    fit = OneParameterFit(
        family=family,
        values=vals,
        costs=tuple(float(x) for x in costs),
        best_value=vals[ib],
        best_cost=float(costs_a[ib]),
        nominal_cost=float(costs_a[inom]),
    )
    # Require a small but real improvement; otherwise keep nominal rather than
    # converting noise into a physical diagnosis.
    return (configs[ib] if fit.improvement_fraction >= 0.01 else initial), fit


def _selected_from_hierarchical(fit) -> dict[str, float]:
    out = {}
    for step in fit.steps:
        if step.accepted and step.selected_family is not None:
            out[str(step.selected_family)] = float(step.selected_value)
    return out


def build(out: Path = OUT) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    registry = system_sweep_registry()
    nominal = SystemErrorConfig()

    nominal_route = v1._route(nominal, None)
    residual = v1._residual_phase(nominal_route["grid"])
    truth = v1.apply_registry_family(nominal, "beam_radius_scale", 0.85, registry=registry)
    truth = v1.apply_registry_family(truth, "axicon_lateral_decentre_x", 250e-6, registry=registry)

    ideal_route = v1._route(nominal, None)
    distorted_route = v1._route(truth, residual)
    distorted_clean = v1._propagate(distorted_route, Z_FIT_M)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=distorted_clean.shape)
    distorted = v1._add_noise(distorted_clean, noise)

    x = np.asarray(ideal_route["grid"]["x"], float)
    ids = np.flatnonzero(np.abs(x) <= FIT_CROP_HALF_M)
    target = distorted[:, ids[:, None], ids]
    simulator = v1.PhysicalSimulator(ids)

    # 1) Beam radius from the longitudinal Bessel envelope.  This retains the
    # annulus-to-z illumination information that plane-wise normalization erased.
    beam_cfg, beam_fit = _fit_single_registry_parameter(
        target, simulator, nominal, "beam_radius_scale", registry
    )

    # 2) Aperture radius from centered, azimuthally averaged transverse shape.
    iris_fit = hierarchical_physical_fit(
        target_stack=target,
        simulate_config=simulator,
        initial_config=beam_cfg,
        families=("fourf_iris_radius_scale",),
        registry=registry,
        max_stages=1,
        min_improvement_fraction=0.004,
        loss_fn=v2.axisymmetric_shape_error,
    )

    # 3) Lateral optical errors from beam trajectory versus z.
    lateral_fit = hierarchical_physical_fit(
        target_stack=target,
        simulate_config=simulator,
        initial_config=iris_fit.final_config,
        families=v2.LATERAL_PARAMETERS,
        registry=registry,
        max_stages=1,
        min_improvement_fraction=0.004,
        loss_fn=v2.lateral_trajectory_error,
    )
    estimated = lateral_fit.final_config

    compensated = v1._compensate_physical_parameters(truth, estimated)
    compensated_route = v1._route(compensated, residual)
    compensated_clean = v1._propagate(compensated_route, Z_FIT_M)
    compensated_noisy = v1._add_noise(compensated_clean, noise)

    miao_phase, miao_diag = v2.miao_correction_calibrated_axis(distorted_route, distorted)
    hybrid_phase, hybrid_diag = v2.miao_correction_calibrated_axis(compensated_route, compensated_noisy)

    compensated_nores = v1._route(compensated, None)
    oracle_phase, oracle_valid = v2._oracle_phase(compensated_route, compensated_nores)

    ideal_disp = v1._propagate(ideal_route, Z_DISPLAY_M)
    distorted_disp = v1._propagate(distorted_route, Z_DISPLAY_M)
    miao_disp = v1._propagate(distorted_route, Z_DISPLAY_M, miao_phase)
    hybrid_disp = v1._propagate(compensated_route, Z_DISPLAY_M, hybrid_phase)
    oracle_disp = v1._propagate(compensated_route, Z_DISPLAY_M, oracle_phase)
    metrics = v2._metric_rows(
        Z_DISPLAY_M, ideal_disp, distorted_disp, miao_disp, hybrid_disp, oracle_disp,
        ideal_route["grid"],
    )
    metrics.to_csv(out/"comparison_metrics_vs_z.csv", index=False)

    selected = {}
    if beam_cfg != nominal:
        selected["beam_radius_scale"] = float(beam_fit.best_value)
    selected.update(_selected_from_hierarchical(iris_fit))
    selected.update(_selected_from_hierarchical(lateral_fit))

    mvalid = oracle_valid & (np.abs(miao_phase) > 0)
    hvalid = oracle_valid & (np.abs(hybrid_phase) > 0)
    summary = {
        "study": "q20 physical-observable fit followed by Miao residual-phase retrieval v3",
        "truth": {
            "beam_radius_scale": 0.85,
            "axicon_lateral_decentre_x_m": 250e-6,
            "residual_phase": "0.42 cos(2theta)+0.24 sin(3theta)+0.12 cos(5theta)",
        },
        "z_sampling": {
            "fit_planes": len(Z_FIT_M),
            "z_min_mm": float(Z_FIT_M[0]*1e3),
            "z_max_mm": float(Z_FIT_M[-1]*1e3),
        },
        "physical_parameter_estimation": {
            "selected": selected,
            "beam_radius_fit": beam_fit.as_dict(),
            "fourier_iris_fit": iris_fit.as_dict(),
            "lateral_fit": lateral_fit.as_dict(),
            "method": (
                "beam radius from longitudinal Bessel brightness envelope; "
                "iris radius from azimuthally averaged transverse morphology; "
                "lateral errors from centroid trajectory"
            ),
        },
        "miao_only": {
            **miao_diag,
            "phase_rms_to_oracle_rad": v2._circular_phase_rms(miao_phase, oracle_phase, mvalid),
        },
        "digital_twin_plus_miao": {
            **hybrid_diag,
            "phase_rms_to_oracle_rad": v2._circular_phase_rms(hybrid_phase, oracle_phase, hvalid),
        },
    }
    for key, col in (
        ("aberrated", "distorted"),
        ("miao_only", "miao_only"),
        ("digital_twin_plus_miao", "digital_twin_plus_miao"),
        ("oracle_phase_only", "oracle_phase_only"),
    ):
        summary.setdefault(key, {})
        summary[key].update({
            "mean_pearson_r": float(metrics[f"{col}_pearson_r"].mean()),
            "mean_nrmse": float(metrics[f"{col}_nrmse"].mean()),
        })
    summary["hybrid_minus_miao"] = {
        "mean_pearson_r": summary["digital_twin_plus_miao"]["mean_pearson_r"] - summary["miao_only"]["mean_pearson_r"],
        "mean_nrmse": summary["digital_twin_plus_miao"]["mean_nrmse"] - summary["miao_only"]["mean_nrmse"],
    }
    summary["hybrid_minus_aberrated"] = {
        "mean_pearson_r": summary["digital_twin_plus_miao"]["mean_pearson_r"] - summary["aberrated"]["mean_pearson_r"],
        "mean_nrmse": summary["digital_twin_plus_miao"]["mean_nrmse"] - summary["aberrated"]["mean_nrmse"],
    }

    np.save(out/"miao_correction_rad.npy", miao_phase.astype(np.float32))
    np.save(out/"hybrid_correction_rad.npy", hybrid_phase.astype(np.float32))
    np.save(out/"oracle_correction_rad.npy", oracle_phase.astype(np.float32))
    np.save(out/"target_longitudinal_envelope.npy", _top_intensity_envelope(target).astype(np.float32))

    # Reuse the v2 diagnostic layout for now; poster styling comes only after the
    # physics benchmark is settled.
    png, pdf, preview = v2._build_diagnostic_figure(
        out, ideal_route["grid"], metrics, summary,
        ideal_disp, distorted_disp, miao_disp, hybrid_disp, oracle_disp,
    )
    # Give v3 assets distinct names.
    p3 = out/"q20_method_physics_v3.png"
    d3 = out/"q20_method_physics_v3.pdf"
    r3 = out/"q20_method_physics_v3.preview.jpg"
    Path(png).replace(p3); Path(pdf).replace(d3); Path(preview).replace(r3)
    with Image.open(p3) as im:
        summary["assets"] = {
            "png": str(p3), "pdf": str(d3), "preview": str(r3),
            "pixel_size": list(im.size), "dpi": list(im.info.get("dpi", (0, 0))),
        }
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    build()
