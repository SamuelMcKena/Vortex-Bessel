"""Extended Virtual Lab experiment builder and iterative SLM-only recovery.

This module builds on :mod:`labcontrol.virtual_lab` without changing the live
hardware provider path.  It adds three things needed for useful offline control
experiments:

* explicit, combinable hidden perturbations of the incident Gaussian, low-order
  wavefront and physical axicon;
* a measurement-plan helper with configurable z-plane count/repeats and an
  explicit frame budget;
* an outer auto-convergence loop which repeatedly runs the existing verified
  sensorless SLM optimiser until improvement stalls or the budget is exhausted.

The optimiser still receives intensity frames and public commands only.  Hidden
truth remains inside the virtual bench.  First-order SLM steering (tip/tilt) is
part of the default correction authority because some apparent alignment/beam
walk errors are deliberately allowed to be compensated optically before any
mechanical Alignment Assist is considered.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from labcontrol.devices.base import ProviderError
from labcontrol.metrics import MetricEngine
from labcontrol.state import ExperimentState, utc_now
from labcontrol.virtual_lab import (
    AlignmentCommand,
    BlindCorrectionResult,
    BlindCorrectionRunner,
    DataOrigin,
    EPS,
    OperatingMode,
    ScenarioKind,
    TWOPI,
    VirtualBenchEngine,
    VirtualScenarioHandle,
    _HiddenScenario,
    _phase_sample_to_model,
    _scenario_from_seed,
    _truth_hash,
)
from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, gaussian_input_field
from vbb_study.digital_twin.vortex_system_route import AxiconError, physical_axicon_on_own_plane
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


@dataclass(frozen=True)
class ManualPerturbationSpec:
    """Combinable hidden perturbations applied before/at the physical elements.

    Units are deliberately operator-friendly.  A curvature radius of 0 means a
    collimated axis (infinite radius).  Zernike values are RMS waves and use the
    same named basis already used by the repository digital twin.
    """

    radius_x_scale: float = 1.0
    radius_y_scale: float = 1.0
    beam_decentre_x_um: float = 0.0
    beam_decentre_y_um: float = 0.0
    beam_pointing_x_mrad: float = 0.0
    beam_pointing_y_mrad: float = 0.0
    curvature_radius_x_m: float = 0.0
    curvature_radius_y_m: float = 0.0

    defocus_waves: float = 0.0
    astigmatism_x_waves: float = 0.0
    astigmatism_y_waves: float = 0.0
    coma_x_waves: float = 0.0
    coma_y_waves: float = 0.0
    spherical_waves: float = 0.0

    axicon_decentre_x_um: float = 0.0
    axicon_decentre_y_um: float = 0.0

    def validate(self) -> None:
        if self.radius_x_scale <= 0 or self.radius_y_scale <= 0:
            raise ValueError("Input-beam radius scales must be positive.")
        if abs(self.beam_pointing_x_mrad) > 1000 or abs(self.beam_pointing_y_mrad) > 1000:
            raise ValueError("Input pointing is outside the supported virtual-lab range.")
        for value in (self.curvature_radius_x_m, self.curvature_radius_y_m):
            if value != 0.0 and abs(value) < 1e-6:
                raise ValueError("Finite curvature radius is too close to zero.")

    def zernike_map(self) -> dict[str, float]:
        values = {
            "defocus": self.defocus_waves,
            "astigmatism_x": self.astigmatism_x_waves,
            "astigmatism_y": self.astigmatism_y_waves,
            "coma_x": self.coma_x_waves,
            "coma_y": self.coma_y_waves,
            "spherical": self.spherical_waves,
        }
        return {key: float(value) for key, value in values.items() if abs(float(value)) > 0.0}


@dataclass(frozen=True)
class _ExtendedHiddenScenario:
    input_zernike_waves_rms: Mapping[str, float]
    beam_decentre_m: tuple[float, float]
    beam_pointing_rad: tuple[float, float]
    axicon_decentre_m: tuple[float, float]
    radius_x_scale: float = 1.0
    radius_y_scale: float = 1.0
    curvature_radius_x_m: float = math.inf
    curvature_radius_y_m: float = math.inf

    def primitive(self) -> dict[str, Any]:
        return {
            "input_zernike_waves_rms": dict(self.input_zernike_waves_rms),
            "beam_decentre_m": list(self.beam_decentre_m),
            "beam_pointing_rad": list(self.beam_pointing_rad),
            "axicon_decentre_m": list(self.axicon_decentre_m),
            "radius_x_scale": float(self.radius_x_scale),
            "radius_y_scale": float(self.radius_y_scale),
            "curvature_radius_x_m": (
                None if not np.isfinite(self.curvature_radius_x_m) else float(self.curvature_radius_x_m)
            ),
            "curvature_radius_y_m": (
                None if not np.isfinite(self.curvature_radius_y_m) else float(self.curvature_radius_y_m)
            ),
        }


def _extended_from_base(hidden: _HiddenScenario) -> _ExtendedHiddenScenario:
    return _ExtendedHiddenScenario(
        input_zernike_waves_rms=dict(hidden.input_zernike_waves_rms),
        beam_decentre_m=tuple(hidden.beam_decentre_m),
        beam_pointing_rad=tuple(hidden.beam_pointing_rad),
        axicon_decentre_m=tuple(hidden.axicon_decentre_m),
    )


class ExperimentalVirtualBenchEngine(VirtualBenchEngine):
    """Virtual bench with explicit input-beam perturbation controls."""

    def __init__(self, *args, canonical_beam_radius_mm: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.canonical_beam_radius_mm = float(canonical_beam_radius_mm)
        self._hidden = _extended_from_base(self._hidden)

    def set_canonical_beam_radius_mm(self, value: float) -> None:
        value = float(value)
        if not (0.05 <= value <= 20.0):
            raise ValueError("Canonical beam radius must be between 0.05 and 20 mm.")
        with self._lock:
            self.canonical_beam_radius_mm = value

    def generate_scenario(self, kind: str | ScenarioKind, seed: int) -> VirtualScenarioHandle:
        scenario_kind = kind if isinstance(kind, ScenarioKind) else ScenarioKind(str(kind))
        base = _scenario_from_seed(scenario_kind, int(seed))
        hidden = _extended_from_base(base)

        # Use already-supported GaussianBeamError degrees of freedom to make the
        # mixed/stress cases exercise amplitude/curvature mismatch as well as
        # simple pointing/decentre.  These remain modest and deterministic.
        if scenario_kind in {ScenarioKind.MIXED, ScenarioKind.STRESS_TEST}:
            rng = np.random.default_rng(int(seed) + 9173)
            scale_sigma = 0.035 if scenario_kind is ScenarioKind.MIXED else 0.08
            rx = max(0.70, 1.0 + float(rng.normal(0.0, scale_sigma)))
            ry = max(0.70, 1.0 + float(rng.normal(0.0, scale_sigma)))
            curvature_scale = 4.0 if scenario_kind is ScenarioKind.MIXED else 2.0
            curv_x = float(rng.choice([-1.0, 1.0]) * rng.uniform(curvature_scale, curvature_scale * 2.0))
            curv_y = float(rng.choice([-1.0, 1.0]) * rng.uniform(curvature_scale, curvature_scale * 2.0))
            hidden = _ExtendedHiddenScenario(
                **{**hidden.__dict__, "radius_x_scale": rx, "radius_y_scale": ry,
                   "curvature_radius_x_m": curv_x, "curvature_radius_y_m": curv_y}
            )

        handle = VirtualScenarioHandle(
            scenario_id=f"virtual-{scenario_kind.value.lower()}-seed-{int(seed)}",
            kind=scenario_kind.value,
            seed=int(seed),
            truth_hash=_truth_hash(hidden),
            created_utc=utc_now(),
        )
        with self._lock:
            self._hidden = hidden
            self._scenario = handle
            self._revealed = False
            self._alignment = AlignmentCommand()
        return handle

    def apply_manual_perturbations(
        self,
        spec: ManualPerturbationSpec,
        *,
        seed: int = 0,
        label: str = "MANUAL",
    ) -> VirtualScenarioHandle:
        spec.validate()
        curvature_x = math.inf if spec.curvature_radius_x_m == 0.0 else float(spec.curvature_radius_x_m)
        curvature_y = math.inf if spec.curvature_radius_y_m == 0.0 else float(spec.curvature_radius_y_m)
        hidden = _ExtendedHiddenScenario(
            input_zernike_waves_rms=spec.zernike_map(),
            beam_decentre_m=(spec.beam_decentre_x_um * 1e-6, spec.beam_decentre_y_um * 1e-6),
            beam_pointing_rad=(spec.beam_pointing_x_mrad * 1e-3, spec.beam_pointing_y_mrad * 1e-3),
            axicon_decentre_m=(spec.axicon_decentre_x_um * 1e-6, spec.axicon_decentre_y_um * 1e-6),
            radius_x_scale=float(spec.radius_x_scale),
            radius_y_scale=float(spec.radius_y_scale),
            curvature_radius_x_m=curvature_x,
            curvature_radius_y_m=curvature_y,
        )
        handle = VirtualScenarioHandle(
            scenario_id=f"virtual-{label.lower()}-seed-{int(seed)}",
            kind=str(label),
            seed=int(seed),
            truth_hash=_truth_hash(hidden),
            created_utc=utc_now(),
        )
        with self._lock:
            self._hidden = hidden
            self._scenario = handle
            self._revealed = False
            self._alignment = AlignmentCommand()
        return handle

    def _forward_complex_field(
        self,
        state: ExperimentState,
        *,
        z_mm: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Same virtual route as the base engine, with full Gaussian input errors."""

        with self._lock:
            hidden = self._hidden
            scenario = self._scenario
            alignment = self._alignment
            optical = {k: v.copy() for k, v in self._optical_cast_phase.items()}
            hashes = dict(self._cast_hashes)
            geometry = self.geometry
            beam_radius_mm = float(self.canonical_beam_radius_mm)

        missing = [name for name in ("SLM1", "SLM2") if name not in optical]
        if missing:
            raise ProviderError(
                "Virtual bench has no cast phase for " + ", ".join(missing)
                + ". Cast both virtual SLMs before acquiring a frame."
            )

        grid = self._grid()
        wavelength_m = float(state.system.wavelength_nm) * 1e-9
        beam_error = GaussianBeamError(
            radius_x_scale=float(getattr(hidden, "radius_x_scale", 1.0)),
            radius_y_scale=float(getattr(hidden, "radius_y_scale", 1.0)),
            decentre_m=tuple(hidden.beam_decentre_m),
            pointing_rad=tuple(hidden.beam_pointing_rad),
            curvature_radius_x_m=float(getattr(hidden, "curvature_radius_x_m", math.inf)),
            curvature_radius_y_m=float(getattr(hidden, "curvature_radius_y_m", math.inf)),
        )
        field, beam_meta = gaussian_input_field(
            grid,
            wavelength_m=wavelength_m,
            canonical_radius_m=beam_radius_mm * 1e-3,
            error=beam_error,
        )

        pupil_radius_m = max(0.5e-3, 0.5 * float(state.slm1.phase.pupil_diameter_mm) * 1e-3)
        hidden_phase = self._hidden_input_phase(grid, wavelength_m, pupil_radius_m)
        field = np.asarray(field, dtype=np.complex128) * np.exp(1j * hidden_phase)

        phi1 = _phase_sample_to_model(optical["SLM1"], state.slm1, grid)
        field = field * np.exp(1j * phi1)

        d12_m = float(geometry.slm1_to_slm2_mm) * 1e-3
        if abs(d12_m) > 1e-15:
            field = angular_spectrum_propagate_bl(
                field, grid, wavelength_m, d12_m,
                n_medium=1.0, bandlimit=True, include_evanescent=True,
            )

        phi2 = _phase_sample_to_model(optical["SLM2"], state.slm2, grid)
        field = np.asarray(field, dtype=np.complex128) * np.exp(1j * phi2)

        effective_axicon_decentre = (
            float(hidden.axicon_decentre_m[0]) + float(alignment.axicon_x_um) * 1e-6,
            float(hidden.axicon_decentre_m[1]) + float(alignment.axicon_y_um) * 1e-6,
        )
        axicon_t, axicon_meta = physical_axicon_on_own_plane(
            grid,
            wavelength_m=wavelength_m,
            base_angle_rad=math.radians(float(geometry.axicon_model_base_angle_deg)),
            refractive_index=float(geometry.axicon_refractive_index),
            external_index=float(geometry.axicon_external_index),
            error=AxiconError(decentre_m=effective_axicon_decentre),
        )
        post_axicon = np.asarray(field, dtype=np.complex128) * axicon_t

        z_m = float(z_mm) * 1e-3
        if abs(z_m) > 1e-15:
            observed = angular_spectrum_propagate_bl(
                post_axicon, grid, wavelength_m, z_m,
                n_medium=1.0, bandlimit=True, include_evanescent=True,
            )
        else:
            observed = post_axicon

        radial_period_m = TWOPI / max(abs(float(axicon_meta["exact_kr_m_inv"])), EPS)
        sampling_per_radial_period = radial_period_m / max(float(grid["dx"]), EPS)
        warnings: list[str] = []
        if sampling_per_radial_period < 3.0:
            warnings.append(
                f"Axicon radial phase period is sampled by only {sampling_per_radial_period:.2f} pixels; "
                "use Validation quality before interpreting fine structure."
            )
        warnings.append(
            "4F order selection is an ideal selected-order surrogate because the physical pinhole "
            "axial position/full relay geometry are not yet bench-bound."
        )
        warnings.append(
            "The reported '20° Thorlabs axicon' is not yet mapped to the model base-angle convention; "
            f"using {geometry.axicon_model_base_angle_deg:g}° ({geometry.axicon_model_angle_source})."
        )

        metadata = {
            "scenario_id": scenario.scenario_id,
            "scenario_kind": scenario.kind,
            "scenario_seed": scenario.seed,
            "scenario_truth_hash": scenario.truth_hash,
            "data_origin": DataOrigin.SIMULATED.value,
            "simulation_quality": self.quality,
            "grid_n": self.grid_n,
            "simulation_window_mm": geometry.simulation_window_mm,
            "model_pixel_size_um": self.pixel_size_um,
            "canonical_beam_radius_mm": beam_radius_mm,
            "wavelength_nm": float(state.system.wavelength_nm),
            "z_mm": float(z_mm),
            "z_reference": geometry.camera_z_reference,
            "relay_mode": geometry.relay_mode,
            "geometry_status": "PARTIALLY_BOUND_NOT_BENCH_CALIBRATED",
            "slm1_to_slm2_mm": geometry.slm1_to_slm2_mm,
            "slm1_to_slm2_source": geometry.slm1_to_slm2_source,
            "lens_to_lens_separation_mm": geometry.lens_to_lens_separation_mm,
            "pinhole_axial_position_mm": geometry.pinhole_axial_position_mm,
            "axicon_reported_label": geometry.axicon_reported_label,
            "axicon_model": axicon_meta,
            "beam_model": beam_meta,
            "cast_phase_hashes": hashes,
            "alignment_command": asdict(alignment),
            "warnings": warnings,
            "hidden_truth_exposed": False,
        }
        return np.asarray(observed, dtype=np.complex128), metadata


def random_manual_perturbation(seed: int, *, severity: float = 1.0) -> ManualPerturbationSpec:
    """Generate a deterministic multi-fault manual specification for UI use."""

    severity = max(0.0, float(severity))
    rng = np.random.default_rng(int(seed))
    z = rng.normal(0.0, 0.11 * severity, 6)
    scales = np.clip(1.0 + rng.normal(0.0, 0.045 * severity, 2), 0.70, 1.30)
    return ManualPerturbationSpec(
        radius_x_scale=float(scales[0]),
        radius_y_scale=float(scales[1]),
        beam_decentre_x_um=float(rng.normal(0.0, 90.0 * severity)),
        beam_decentre_y_um=float(rng.normal(0.0, 90.0 * severity)),
        beam_pointing_x_mrad=float(rng.normal(0.0, 0.10 * severity)),
        beam_pointing_y_mrad=float(rng.normal(0.0, 0.10 * severity)),
        curvature_radius_x_m=float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 7.0)) if severity else 0.0,
        curvature_radius_y_m=float(rng.choice([-1.0, 1.0]) * rng.uniform(3.0, 7.0)) if severity else 0.0,
        defocus_waves=float(z[0]),
        astigmatism_x_waves=float(z[1]),
        astigmatism_y_waves=float(z[2]),
        coma_x_waves=float(z[3]),
        coma_y_waves=float(z[4]),
        spherical_waves=float(z[5]),
        axicon_decentre_x_um=float(rng.normal(0.0, 45.0 * severity)),
        axicon_decentre_y_um=float(rng.normal(0.0, 45.0 * severity)),
    )


def build_even_z_plan(start_mm: float, stop_mm: float, count: int) -> tuple[float, ...]:
    count = int(count)
    if count < 2:
        raise ValueError("At least two z planes are required.")
    if count > 101:
        raise ValueError("Virtual z-plan count is capped at 101 planes.")
    return tuple(float(v) for v in np.linspace(float(start_mm), float(stop_mm), count))


def expand_measurement_plan(z_plan_mm: Sequence[float], repeats_per_plane: int) -> tuple[float, ...]:
    repeats = int(repeats_per_plane)
    if repeats < 1 or repeats > 20:
        raise ValueError("repeats_per_plane must be between 1 and 20.")
    values = tuple(float(z) for z in z_plan_mm)
    if len(values) < 2:
        raise ValueError("At least two z planes are required.")
    return tuple(z for z in values for _ in range(repeats))


DEFAULT_SLM_CORRECTION_PARAMETERS = (
    # Alignment-equivalent optical authority first: try to remove pointing/axis
    # walk with programmable phase before declaring a mechanical move necessary.
    "tip_x",
    "tip_y",
    "defocus",
    "astig_x",
    "astig_xy",
    "coma_x",
    "coma_y",
    "spherical",
)


def estimated_blind_cycle_frames(
    z_plane_count: int,
    repeats_per_plane: int,
    parameter_count: int,
    target_count: int,
) -> int:
    """Exact current runner frame count for one single-pass correction cycle.

    Each mode uses 2 controls + 4 candidates + 1 fresh verification = 7 z
    stacks.  BlindCorrectionRunner additionally measures an initial and final
    stack for the cycle.
    """

    stacks = 2 + 7 * int(parameter_count) * int(target_count)
    return stacks * int(z_plane_count) * int(repeats_per_plane)


@dataclass
class AutoConvergeResult:
    initial_objective: float
    final_objective: float
    improvement_fraction: float
    cycles: list[dict[str, Any]]
    total_frames: int
    stop_reason: str
    converged: bool
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "improvement_fraction": self.improvement_fraction,
            "cycles": list(self.cycles),
            "total_frames": int(self.total_frames),
            "stop_reason": self.stop_reason,
            "converged": bool(self.converged),
            "cancelled": bool(self.cancelled),
        }


class AutoConvergeRunner:
    """Repeat verified SLM-only correction cycles until improvement stalls."""

    def __init__(self, controller: Any, *, metric_engine: MetricEngine | None = None):
        self.controller = controller
        self.metric_engine = metric_engine or MetricEngine()

    def run(
        self,
        z_plan_mm: Sequence[float],
        *,
        repeats_per_plane: int = 1,
        targets: Sequence[str] = ("SLM1", "SLM2"),
        parameters: Sequence[str] = DEFAULT_SLM_CORRECTION_PARAMETERS,
        initial_probe_amplitude_waves: float = 0.15,
        max_cycles: int = 5,
        convergence_fraction: float = 0.005,
        patience: int = 2,
        max_frames: int = 5000,
        seed: int = 42,
        frame_callback: Callable[[Any], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AutoConvergeResult:
        if self.controller.operating_mode is not OperatingMode.VIRTUAL_LAB:
            raise RuntimeError("Auto-converge correction requires VIRTUAL LAB mode.")

        base_z = tuple(float(z) for z in z_plan_mm)
        expanded_z = expand_measurement_plan(base_z, repeats_per_plane)
        target_names = tuple(str(name).upper() for name in targets)
        mode_parameters = tuple(str(p) for p in parameters)
        if not mode_parameters:
            raise ValueError("Select at least one SLM correction parameter.")
        max_cycles = max(1, int(max_cycles))
        patience = max(1, int(patience))
        max_frames = max(1, int(max_frames))
        convergence_fraction = max(0.0, float(convergence_fraction))

        frames_per_cycle = estimated_blind_cycle_frames(
            len(base_z), repeats_per_plane, len(mode_parameters), len(target_names)
        )
        if frames_per_cycle > max_frames:
            raise ValueError(
                f"One correction cycle requires about {frames_per_cycle} frames, "
                f"exceeding the max-frame budget of {max_frames}."
            )

        cycles: list[dict[str, Any]] = []
        total_frames = 0
        initial_objective: float | None = None
        current_objective: float | None = None
        low_improvement_streak = 0
        stop_reason = "max_cycles"
        was_cancelled = False

        for cycle_index in range(max_cycles):
            if cancelled is not None and cancelled():
                stop_reason = "cancelled"
                was_cancelled = True
                break
            if total_frames + frames_per_cycle > max_frames:
                stop_reason = "frame_budget"
                break

            probe = abs(float(initial_probe_amplitude_waves)) / (2.0 ** cycle_index)
            if progress_callback is not None:
                progress_callback(
                    {
                        "kind": "cycle_start",
                        "cycle": cycle_index + 1,
                        "max_cycles": max_cycles,
                        "probe_amplitude_waves": probe,
                        "estimated_cycle_frames": frames_per_cycle,
                        "remaining_frame_budget": max_frames - total_frames,
                    }
                )

            runner = BlindCorrectionRunner(self.controller, metric_engine=self.metric_engine)
            result: BlindCorrectionResult = runner.run(
                expanded_z,
                targets=target_names,
                parameters=mode_parameters,
                probe_amplitude_waves=probe,
                passes=1,
                seed=int(seed) + cycle_index * 101,
                frame_callback=frame_callback,
                progress_callback=progress_callback,
                cancelled=cancelled,
            )
            total_frames += frames_per_cycle
            if initial_objective is None:
                initial_objective = float(result.initial_objective)
            previous = float(result.initial_objective)
            current_objective = float(result.final_objective)
            cycle_improvement = (previous - current_objective) / max(abs(previous), EPS)
            cycles.append(
                {
                    "cycle": cycle_index + 1,
                    "probe_amplitude_waves": probe,
                    "objective_before": previous,
                    "objective_after": current_objective,
                    "fractional_improvement": cycle_improvement,
                    "accepted_runs": result.accepted_runs,
                    "estimated_frames": frames_per_cycle,
                }
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "kind": "cycle_complete",
                        "cycle": cycle_index + 1,
                        "objective_before": previous,
                        "objective_after": current_objective,
                        "fractional_improvement": cycle_improvement,
                        "total_frames": total_frames,
                    }
                )

            if result.cancelled:
                stop_reason = "cancelled"
                was_cancelled = True
                break

            if cycle_improvement < convergence_fraction:
                low_improvement_streak += 1
            else:
                low_improvement_streak = 0
            if low_improvement_streak >= patience:
                stop_reason = "converged_low_improvement"
                break

        if initial_objective is None or current_objective is None:
            raise RuntimeError("Auto-converge stopped before completing a correction cycle.")

        overall = (initial_objective - current_objective) / max(abs(initial_objective), EPS)
        return AutoConvergeResult(
            initial_objective=initial_objective,
            final_objective=current_objective,
            improvement_fraction=float(overall),
            cycles=cycles,
            total_frames=total_frames,
            stop_reason=stop_reason,
            converged=(stop_reason == "converged_low_improvement"),
            cancelled=was_cancelled,
        )
