"""Virtual optical bench adapters for blind, intensity-only correction development.

This module deliberately reuses the repository's physical optics primitives.  It
does not manufacture Gentec .BMG files and it never claims simulated arrays are
measurements.  The GUI receives the same CameraFrame contract used by the real
camera provider.

The current laboratory geometry is only partially bound.  Reported values are
kept separate from model placeholders so an offline correction study can run
without turning an assumption into "measured bench geometry".
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from slm_lab_control.config import AppConfig
from labcontrol.devices.base import DeviceStatus, ProviderError
from labcontrol.devices.camera import CameraFrame, CameraProvider
from labcontrol.devices.slm import SlmProvider
from labcontrol.devices.stage import MoveRequest, StageProvider
from labcontrol.metrics import BeamMetrics, MetricEngine
from labcontrol.phase_service import PhaseService, phase_sha256
from labcontrol.sensorless import SensorlessOptimiser
from labcontrol.state import ConnectionState, ExperimentState, utc_now

from vbb_study.digital_twin.vortex_beam_slm_errors import (
    GaussianBeamError,
    gaussian_input_field,
)
from vbb_study.digital_twin.vortex_system_route import (
    AxiconError,
    physical_axicon_on_own_plane,
)
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_wavefront_errors import unit_rms_zernike
from labcontrol.relay_4f import apply_fourier_plane_aperture, rotate_180
from labcontrol.sample_plane import SamplePlaneScale, bench_ring_radius_um, sample_plane_scale
from labcontrol.axicon_propagation import (
    BEAMAGE_PIXEL_UM,
    BEAMAGE_PIXELS,
    DEFAULT_MIN_SAMPLES_PER_PERIOD,
    MEASURED_BENCH_AXICON_K_PERP_M_INV,
    MEASURED_BENCH_AXICON_SOURCE,
    AxiconPropagationPlan,
    build_resolved_axicon_spectrum,
    detector_intensity,
    plan_axicon_propagation,
    resolved_field_at_z,
)
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


TWOPI = 2.0 * np.pi
EPS = 1e-30


def _finite_beam_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Represent collimated axes without non-standard JSON infinities."""
    result = dict(metadata)
    for axis in ("x", "y"):
        key = f"beam_curvature_radius_{axis}_m"
        value = result.get(key)
        collimated = value is not None and not math.isfinite(float(value))
        if collimated:
            result[key] = None
        result[f"beam_{axis}_collimated"] = bool(collimated)
    return result


class OperatingMode(str, Enum):
    LIVE_LAB = "LIVE LAB"
    RECORDED_LAB = "RECORDED LAB"
    VIRTUAL_LAB = "VIRTUAL LAB"


class DataOrigin(str, Enum):
    LIVE_MEASURED = "live_measured"
    RECORDED_MEASURED = "recorded_measured"
    SIMULATED = "simulated"


class ScenarioKind(str, Enum):
    NOMINAL = "NOMINAL"
    LOW_ORDER_WAVEFRONT = "LOW_ORDER_WAVEFRONT"
    ALIGNMENT = "ALIGNMENT"
    MIXED = "MIXED"
    STRESS_TEST = "STRESS_TEST"


@dataclass(frozen=True)
class VirtualLabGeometry:
    """Known bench facts and explicitly labelled modelling placeholders."""

    wavelength_nm: float = 1030.0
    # 10 mm matches the bench digital twin (FIT_WINDOW_M) and holds the annulus
    # of the measured bench axicon well past the end of its Bessel zone.
    # Exactly the Beamage-4M sensor (2048 x 5.5 µm): every virtual camera pixel
    # is then a true 5.5 µm pixel, so sensor aliasing of the ~6.5 µm Bessel
    # intensity fringes appears as it does on the real camera.
    simulation_window_mm: float = BEAMAGE_PIXELS * BEAMAGE_PIXEL_UM / 1000.0
    preview_grid_n: int = 256
    validation_grid_n: int = 512

    # Present GUI/HEDS phase geometry.
    slm_pixel_pitch_um: float = 8.0

    # The existing CSLM route contains 0.040 mm, but its own documentation marks
    # it as diagnostic/placeholder geometry.  Keep it editable and labelled.
    slm1_to_slm2_mm: float = 0.040
    slm1_to_slm2_source: str = "repo_placeholder_unmeasured"

    # Operator-reported layout (2026-09-17):
    #   SLM -300- L1(f=300) -300- aperture -300- L2(f=300) -300- axicon.
    # Every spacing equals a focal length, so this *is* a symmetric 4F: unit
    # magnification, 180-degree image rotation, stop in the shared focal plane.
    lens1_focal_length_mm: float = 300.0
    lens2_focal_length_mm: float = 300.0
    slm_to_lens1_mm: float = 300.0
    lens_to_lens_separation_mm: float = 600.0
    lens2_to_axicon_mm: float = 300.0
    relay_magnification: float = 1.0
    # A symmetric 4F images the SLM onto the axicon rotated by 180 degrees, so
    # odd errors (tilt, coma, decentre) reach the axicon with the opposite sign.
    relay_inverts_image: bool = True
    relay_geometry_source: str = "operator_reported_20260917_symmetric_4f_300mm_spacings"
    pinhole_role: str = "+1 diffraction-order selection"
    pinhole_axial_position_mm: float | None = 300.0
    # Stop diameter in the shared focal plane.  None = ideal order selection
    # with no spatial filtering; the bench aperture has not been measured.
    fourier_aperture_diameter_mm: float | None = None
    fourier_aperture_source: str = "not_measured_default_ideal_order_select"

    # The optic is reported as a "20 degree Thorlabs axicon".  The repository's
    # current exact-refractive model requires a *base angle*.  Until the part
    # number / convention is bound, retain the old 2 deg model value only as an
    # explicit simulation placeholder.
    axicon_reported_label: str = "Thorlabs 20° axicon"
    # The axicon is set, as in the bench digital twin, by its measured transverse
    # wavenumber.  The "20°" label is not a model base angle: this k_perp implies
    # ~9.9° in the exact refractive convention.  Set k_perp to None to drive the
    # model by base angle instead.
    axicon_k_perp_m_inv: float | None = MEASURED_BENCH_AXICON_K_PERP_M_INV
    axicon_k_perp_source: str = MEASURED_BENCH_AXICON_SOURCE
    axicon_model_base_angle_deg: float = 2.0
    axicon_model_angle_source: str = "used only when axicon_k_perp_m_inv is None"
    axicon_refractive_index: float = 1.458
    axicon_external_index: float = 1.0
    axicon_min_samples_per_period: float = DEFAULT_MIN_SAMPLES_PER_PERIOD
    max_propagation_grid_n: int = 4096

    # Optional demagnifying objective after the axicon; nothing like it is on
    # the bench today.  M = 1/N for a 1:N objective.  The post-axicon pattern
    # scales exactly (see labcontrol/sample_plane.py), so this changes the
    # reported lengths and the z mapping, never the simulated field.
    objective_demagnification: float = 1.0
    objective_source: str = "not_installed_default_unity"

    # The physical camera reference is not yet established.  Virtual z is
    # therefore relative to the post-axicon model plane.
    camera_z_reference: str = "virtual post-axicon plane; physical camera z=0 UNCALIBRATED"

    # The relay distances are now bench-reported, so the model applies the 4F's
    # unit magnification and image rotation.  Only the stop diameter is unknown.
    relay_mode: str = "symmetric_4f_unit_magnification_order_select"

    def quality_grid(self, quality: str) -> int:
        q = str(quality).strip().lower()
        if q == "preview":
            return int(self.preview_grid_n)
        if q == "validation":
            return int(self.validation_grid_n)
        raise ValueError("quality must be 'preview' or 'validation'")

    def public_summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["geometry_status"] = "PARTIALLY_BOUND_NOT_BENCH_CALIBRATED"
        payload["known_fact_notes"] = [
            "Both relay lenses reported as 300 mm focal length.",
            "Operator-reported spacings SLM-300-L1-300-aperture-300-L2-300-axicon: a symmetric 4F, "
            "so the axicon sees the SLM field at unit magnification, rotated by 180 degrees.",
            "The aperture selects +1 in the shared focal plane; its diameter is not measured, so no "
            "spatial filtering is applied unless one is entered.",
            "Physical optic reported as a Thorlabs 20° axicon; modelled by its measured k_perp from the "
            "BeamGage q=20 z-scan rather than by the manufacturer label.",
        ]
        payload["claim_boundary"] = (
            "Offline optical-control model only. Unknown distances/angle conventions remain "
            "explicit; no absolute current-bench prediction is claimed."
        )
        return payload


@dataclass(frozen=True)
class VirtualScenarioHandle:
    scenario_id: str
    kind: str
    seed: int
    truth_hash: str
    created_utc: str


@dataclass(frozen=True)
class _HiddenScenario:
    input_zernike_waves_rms: Mapping[str, float] = field(default_factory=dict)
    beam_decentre_m: tuple[float, float] = (0.0, 0.0)
    beam_pointing_rad: tuple[float, float] = (0.0, 0.0)
    axicon_decentre_m: tuple[float, float] = (0.0, 0.0)

    def primitive(self) -> dict[str, Any]:
        return {
            "input_zernike_waves_rms": dict(self.input_zernike_waves_rms),
            "beam_decentre_m": list(self.beam_decentre_m),
            "beam_pointing_rad": list(self.beam_pointing_rad),
            "axicon_decentre_m": list(self.axicon_decentre_m),
        }


@dataclass(frozen=True)
class AlignmentCommand:
    """User/optimizer-visible virtual mechanical actuator commands."""

    axicon_x_um: float = 0.0
    axicon_y_um: float = 0.0


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total: float
    terms: Mapping[str, float]
    family: str

    def to_dict(self) -> dict[str, Any]:
        return {"total": float(self.total), "family": self.family, "terms": dict(self.terms)}


@dataclass(frozen=True)
class VirtualStackResult:
    z_mm: tuple[float, ...]
    frames: tuple[CameraFrame, ...]
    metrics: tuple[BeamMetrics, ...]
    objective: ObjectiveBreakdown

    def summary(self) -> dict[str, Any]:
        return {
            "z_mm": list(self.z_mm),
            "frame_ids": [f.frame_id for f in self.frames],
            "objective": self.objective.to_dict(),
            "data_origin": DataOrigin.SIMULATED.value,
        }


def _truth_hash(hidden: _HiddenScenario) -> str:
    blob = json.dumps(hidden.primitive(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _scenario_from_seed(kind: ScenarioKind, seed: int) -> _HiddenScenario:
    rng = np.random.default_rng(int(seed))
    if kind is ScenarioKind.NOMINAL:
        return _HiddenScenario()

    zernikes: dict[str, float] = {}
    beam_decentre = (0.0, 0.0)
    beam_pointing = (0.0, 0.0)
    axicon_decentre = (0.0, 0.0)

    if kind in {ScenarioKind.LOW_ORDER_WAVEFRONT, ScenarioKind.MIXED, ScenarioKind.STRESS_TEST}:
        names = ("astigmatism_x", "astigmatism_y", "coma_x", "coma_y", "defocus", "spherical")
        scale = 0.14 if kind is ScenarioKind.LOW_ORDER_WAVEFRONT else 0.11
        if kind is ScenarioKind.STRESS_TEST:
            scale = 0.22
        raw = rng.normal(0.0, scale, len(names))
        # Ensure the blind example is non-trivial while remaining low-order.
        raw[np.argmax(np.abs(raw))] += math.copysign(0.10, raw[np.argmax(np.abs(raw))] or 1.0)
        zernikes = {name: float(value) for name, value in zip(names, raw)}

    if kind in {ScenarioKind.ALIGNMENT, ScenarioKind.MIXED, ScenarioKind.STRESS_TEST}:
        dec_scale = 0.10e-3 if kind is not ScenarioKind.STRESS_TEST else 0.18e-3
        point_scale = 0.10e-3 if kind is not ScenarioKind.STRESS_TEST else 0.20e-3
        ax_scale = 0.05e-3 if kind is not ScenarioKind.STRESS_TEST else 0.10e-3
        beam_decentre = tuple(float(v) for v in rng.normal(0.0, dec_scale, 2))
        beam_pointing = tuple(float(v) for v in rng.normal(0.0, point_scale, 2))
        axicon_decentre = tuple(float(v) for v in rng.normal(0.0, ax_scale, 2))

    return _HiddenScenario(
        input_zernike_waves_rms=zernikes,
        beam_decentre_m=beam_decentre,
        beam_pointing_rad=beam_pointing,
        axicon_decentre_m=axicon_decentre,
    )


def _phase_sample_to_model(
    phase_rad: np.ndarray,
    slm_state: Any,
    grid: Mapping[str, Any],
) -> np.ndarray:
    """Nearest native-pixel sampling of an already composed SLM phase.

    The simulation grid is much coarser than the native panel.  Sampling the
    *wrapped phase* by nearest pixel avoids interpolation across a 2π branch cut.
    The virtual model therefore consumes the exact phase engine's result rather
    than reimplementing vortex/Zernike/retrieved-correction semantics.
    """

    arr = np.asarray(phase_rad, dtype=np.float64)
    cfg = slm_state.phase
    pitch_m = float(cfg.geometry.pixel_pitch_m)
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid["x"], dtype=float)
    ix = np.rint(x / pitch_m + float(cfg.center_x_px)).astype(int)
    iy = np.rint(y / pitch_m + float(cfg.center_y_px)).astype(int)
    valid_x = (ix >= 0) & (ix < arr.shape[1])
    valid_y = (iy >= 0) & (iy < arr.shape[0])
    ix_clip = np.clip(ix, 0, arr.shape[1] - 1)
    iy_clip = np.clip(iy, 0, arr.shape[0] - 1)
    out = arr[np.ix_(iy_clip, ix_clip)].copy()
    valid = np.outer(valid_y, valid_x)
    # Outside the finite SLM panel there is no additional phase command.
    out[~valid] = 0.0
    return out


class VirtualBenchEngine:
    """Stateful hidden optical bench used by the virtual providers.

    The hidden scenario is intentionally private.  Observation methods return
    intensity and public provenance only.  ``reveal_truth`` is a deliberate,
    post-run operator action and is never called by the correction engine.
    """

    def __init__(
        self,
        geometry: VirtualLabGeometry | None = None,
        *,
        quality: str = "preview",
        realistic_camera: bool = False,
    ):
        self.geometry = geometry or VirtualLabGeometry()
        self.quality = str(quality).lower()
        self.realistic_camera = bool(realistic_camera)
        self._lock = threading.RLock()
        self._hidden = _HiddenScenario()
        self._scenario = VirtualScenarioHandle(
            scenario_id="virtual-nominal-seed-0",
            kind=ScenarioKind.NOMINAL.value,
            seed=0,
            truth_hash=_truth_hash(self._hidden),
            created_utc=utc_now(),
        )
        self._revealed = False
        self._full_cast_phase: dict[str, np.ndarray] = {}
        self._optical_cast_phase: dict[str, np.ndarray] = {}
        self._cast_slm_states: dict[str, Any] = {}
        self._cast_hashes: dict[str, str] = {}
        self._alignment = AlignmentCommand()
        self._frame_counter = 0
        self._optical_cache_token = 0
        self._cached_model_intensity: tuple[tuple[Any, ...], np.ndarray, dict[str, Any]] | None = None
        # One frozen post-axicon spectrum per optical state: every camera plane
        # is then a single inverse FFT instead of a full relay + axicon rebuild.
        self._axicon_spectrum_cache: tuple[tuple[Any, ...], Any, dict[str, Any]] | None = None
        # The SLM relay does not change when only the camera moves.
        self._relay_cache: tuple[tuple[Any, ...], np.ndarray, dict[str, Any]] | None = None
        # Noise-free intensity on the resolved model grid, before camera pixels.
        # Only kept while a view asks for it; ~64 MB for the 4096² bench grid.
        self.keep_native_intensity = False
        self._native_cache: tuple[tuple[Any, ...], np.ndarray] | None = None

    def _invalidate_optics(self) -> None:
        # Called with the bench lock held whenever a physical/SLM input changes.
        self._optical_cache_token += 1
        self._cached_model_intensity = None
        self._axicon_spectrum_cache = None
        self._relay_cache = None
        self._native_cache = None

    @property
    def scenario(self) -> VirtualScenarioHandle:
        return self._scenario

    @property
    def grid_n(self) -> int:
        return self.geometry.quality_grid(self.quality)

    @property
    def pixel_size_um(self) -> float:
        return float(self.geometry.simulation_window_mm) * 1000.0 / float(self.grid_n)

    @property
    def shape_yx(self) -> tuple[int, int]:
        return (self.grid_n, self.grid_n)

    def set_quality(self, quality: str) -> None:
        self.geometry.quality_grid(quality)
        with self._lock:
            self.quality = str(quality).lower()
            self._invalidate_optics()

    def set_geometry(self, **updates: Any) -> None:
        with self._lock:
            self.geometry = replace(self.geometry, **updates)
            self._invalidate_optics()

    def generate_scenario(self, kind: str | ScenarioKind, seed: int) -> VirtualScenarioHandle:
        scenario_kind = kind if isinstance(kind, ScenarioKind) else ScenarioKind(str(kind))
        hidden = _scenario_from_seed(scenario_kind, int(seed))
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
            self._invalidate_optics()
        return handle

    def reset_scenario(self) -> VirtualScenarioHandle:
        return self.generate_scenario(ScenarioKind.NOMINAL, 0)

    def reveal_truth(self) -> dict[str, Any]:
        with self._lock:
            self._revealed = True
            return {
                "scenario": asdict(self._scenario),
                "hidden_truth": self._hidden.primitive(),
                "alignment_command": asdict(self._alignment),
                "warning": (
                    "Simulation truth only. Compensation performance does not imply unique "
                    "physical parameter identification on the real bench."
                ),
            }

    def public_scenario_record(self) -> dict[str, Any]:
        return asdict(self._scenario)

    def set_alignment_command(self, *, axicon_x_um: float, axicon_y_um: float) -> None:
        with self._lock:
            self._alignment = AlignmentCommand(float(axicon_x_um), float(axicon_y_um))
            self._invalidate_optics()

    @property
    def alignment_command(self) -> AlignmentCommand:
        return self._alignment

    def set_cast_phase(
        self,
        name: str,
        full_phase_rad: np.ndarray,
        optical_phase_rad: np.ndarray,
        *,
        full_phase_hash: str | None = None,
        slm_state: Any | None = None,
    ) -> None:
        key = str(name).upper()
        if key not in {"SLM1", "SLM2"}:
            raise ValueError("name must be SLM1 or SLM2")
        full = np.asarray(full_phase_rad, dtype=np.float64)
        optical = np.asarray(optical_phase_rad, dtype=np.float64)
        if full.shape != optical.shape:
            raise ValueError("full and selected-order virtual phases must have the same shape")
        with self._lock:
            self._full_cast_phase[key] = full.copy()
            self._optical_cast_phase[key] = optical.copy()
            if slm_state is not None:
                self._cast_slm_states[key] = copy.deepcopy(slm_state)
            self._cast_hashes[key] = full_phase_hash or phase_sha256(full)
            self._invalidate_optics()

    def blank(self, name: str, shape: tuple[int, int]) -> None:
        zeros = np.zeros(shape, dtype=np.float64)
        self.set_cast_phase(name, zeros, zeros)

    def cast_hash(self, name: str) -> str | None:
        return self._cast_hashes.get(str(name).upper())

    def cast_vortex_charge(self) -> int:
        """Charge encoded in the last virtual casts, not uncast GUI edits."""
        with self._lock:
            return sum(
                int(slm.phase.vortex_charge)
                for slm in self._cast_slm_states.values()
                if slm.phase.switches.vortex
            )

    def clear_cast_phases(self) -> None:
        with self._lock:
            self._full_cast_phase.clear()
            self._optical_cast_phase.clear()
            self._cast_slm_states.clear()
            self._cast_hashes.clear()
            self._invalidate_optics()

    def _model_intensity(self, state: ExperimentState, z_mm: float) -> tuple[np.ndarray, dict[str, Any]]:
        """Cache the propagated field, not camera noise, between live frames."""
        with self._lock:
            key = (
                self._optical_cache_token,
                self.grid_n,
                float(z_mm),
                float(state.system.wavelength_nm),
            )
            cached = self._cached_model_intensity
            if cached is not None and cached[0] == key:
                return cached[1], copy.deepcopy(cached[2])

        started = time.perf_counter()
        self._axicon_stage_rebuilt = False
        field, metadata = self._forward_complex_field(state, z_mm=float(z_mm))
        elapsed = time.perf_counter() - started
        if self._axicon_stage_rebuilt:
            # A new optical state (every optimiser probe) pays for relay, axicon
            # and spectrum; the GUI uses this to give honest run-time estimates.
            self.last_full_rebuild_s = elapsed
        # Area-integrate the resolved intensity over each camera pixel.
        intensity = detector_intensity(field, int(metadata.get("propagation_integration_factor", 1)))
        intensity = np.asarray(intensity, dtype=np.float64)
        native = detector_intensity(field, 1) if self.keep_native_intensity else None
        with self._lock:
            if key[0] == self._optical_cache_token:
                self._cached_model_intensity = (key, intensity, copy.deepcopy(metadata))
                if native is not None:
                    self._native_cache = (key, native)
        return intensity, metadata

    def native_intensity(self, state: ExperimentState, z_mm: float) -> tuple[np.ndarray, float]:
        """Noise-free resolved-model intensity for a plane, and its size in camera pixels.

        This is what the optics produce before any camera: the Bessel intensity
        fringes fully resolved.  Comparing it with the camera frame shows what
        5.5 µm pixels and detector noise do to the same light.
        """

        with self._lock:
            key = (
                self._optical_cache_token,
                self.grid_n,
                float(z_mm),
                float(state.system.wavelength_nm),
            )
            cached = self._native_cache
            if cached is not None and cached[0] == key:
                native = cached[1]
                return native, float(self.detector_n) / float(native.shape[0])
        field, _metadata = self._forward_complex_field(state, z_mm=float(z_mm))
        native = detector_intensity(field, 1)
        with self._lock:
            if key[0] == self._optical_cache_token:
                self._native_cache = (key, native)
        return native, float(self.detector_n) / float(native.shape[0])

    def _grid(self) -> dict[str, Any]:
        n = self.grid_n
        width_m = float(self.geometry.simulation_window_mm) * 1e-3
        return make_xy_grid(n, width_m / n)

    def _apply_4f_relay(
        self,
        field: np.ndarray,
        grid: Mapping[str, Any],
        *,
        wavelength_m: float,
        geometry: "VirtualLabGeometry",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Carry the SLM field through the bench 4F onto the axicon.

        The relay is 1:1, so the field is not rescaled.  The stop in the shared
        focal plane low-passes it when a diameter is entered, and the symmetric
        4F hands the axicon a 180-degree rotated image of the SLM plane.
        """

        meta: dict[str, Any] = {
            "relay_magnification": float(geometry.relay_magnification),
            "relay_image_rotation_deg": 180.0 if geometry.relay_inverts_image else 0.0,
        }
        diameter_mm = geometry.fourier_aperture_diameter_mm
        if diameter_mm:
            field, aperture_meta = apply_fourier_plane_aperture(
                field,
                dx_m=float(grid["dx"]),
                wavelength_m=float(wavelength_m),
                focal_length_m=float(geometry.lens1_focal_length_mm) * 1e-3,
                diameter_m=float(diameter_mm) * 1e-3,
            )
            meta.update(aperture_meta)
        else:
            meta["fourier_aperture_diameter_mm"] = None
        if geometry.relay_inverts_image:
            field = rotate_180(field)
        return field, meta

    def _hidden_input_phase(
        self,
        grid: Mapping[str, Any],
        wavelength_m: float,
        pupil_radius_m: float,
    ) -> np.ndarray:
        phase = np.zeros((self.grid_n, self.grid_n), dtype=np.float64)
        for name, waves in self._hidden.input_zernike_waves_rms.items():
            phase += (
                TWOPI
                * float(waves)
                * unit_rms_zernike(
                    name,
                    grid,
                    pupil_radius_m=float(pupil_radius_m),
                )
            )
        return phase

    @property
    def detector_n(self) -> int:
        """Camera pixels across the window; the relay grid unless a camera model says otherwise."""
        return int(self.grid_n)

    def axicon_plan(self, *, grid_n: int | None = None, geometry: "VirtualLabGeometry | None" = None,
                    beam_radius_mm: float | None = None, wavelength_nm: float | None = None) -> AxiconPropagationPlan:
        """What the axicon stage will compute on a (possibly proposed) grid/geometry."""
        geometry = self.geometry if geometry is None else geometry
        return plan_axicon_propagation(
            geometry,
            wavelength_m=float(wavelength_nm if wavelength_nm is not None else geometry.wavelength_nm) * 1e-9,
            relay_grid_n=int(self.grid_n if grid_n is None else grid_n),
            detector_n=self._detector_n_for(int(self.grid_n if grid_n is None else grid_n)),
            beam_radius_mm=float(
                beam_radius_mm if beam_radius_mm is not None else getattr(self, "canonical_beam_radius_mm", 2.0)
            ),
        )

    def peak_model_intensity(self, state: ExperimentState, z_mm: float) -> float:
        """Brightest model intensity at this plane, before camera gain or noise.

        Exposure arithmetic needs the true peak: a clipped frame no longer
        carries it, and detector noise would bias a frame-based estimate.
        """

        intensity, _metadata = self._model_intensity(state, float(z_mm))
        return float(np.max(np.asarray(intensity)))

    def sample_plane(
        self,
        *,
        charge: int = 0,
        geometry: "VirtualLabGeometry | None" = None,
        wavelength_nm: float | None = None,
    ) -> SamplePlaneScale:
        """Where this bench's pattern lands if a 1:N objective follows the axicon.

        Pure bookkeeping: the post-axicon pattern is scale invariant, so the
        simulated frame is already the sample-plane pattern in new units.
        """

        geometry = self.geometry if geometry is None else geometry
        plan = self.axicon_plan(geometry=geometry)
        ring_um = bench_ring_radius_um(k_perp_m_inv=plan.k_perp_m_inv, charge=charge) if plan.valid else None
        return sample_plane_scale(
            demagnification=float(geometry.objective_demagnification),
            camera_pixel_um=float(self.pixel_size_um),
            native_pixel_um=float(self.pixel_size_um) / max(1, int(plan.integration_factor or 1)),
            ring_radius_um=ring_um,
            bessel_length_mm=plan.bessel_zone_mm if plan.valid else None,
            k_perp_m_inv=plan.k_perp_m_inv if plan.valid else None,
            wavelength_nm=float(wavelength_nm if wavelength_nm else geometry.wavelength_nm),
        )

    def _cached_relay(self, key: tuple[Any, ...], compute: Callable[[], tuple[np.ndarray, dict[str, Any]]]):
        with self._lock:
            cached = self._relay_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        field, beam_meta = compute()
        with self._lock:
            if key[0] == self._optical_cache_token:
                self._relay_cache = (key, field, beam_meta)
        return field, beam_meta

    def _detector_n_for(self, relay_grid_n: int) -> int:
        return int(self.detector_n) if relay_grid_n == int(self.grid_n) else int(relay_grid_n)

    def _observe_after_axicon(
        self,
        field_on_axicon: np.ndarray,
        *,
        geometry: "VirtualLabGeometry",
        wavelength_m: float,
        decentre_m: tuple[float, float],
        z_mm: float,
        beam_radius_mm: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Axicon + free-space stage, ported from the bench digital twin's multirate route."""

        relay_n = int(np.asarray(field_on_axicon).shape[0])
        plan = plan_axicon_propagation(
            geometry, wavelength_m=wavelength_m, relay_grid_n=relay_n, beam_radius_mm=beam_radius_mm,
            detector_n=self._detector_n_for(relay_n),
        )
        if not plan.valid:
            raise ProviderError(plan.message)
        z_m = float(z_mm) * 1e-3
        # One band limit serves every plane up to z_support.  Rounding up to the
        # next 10 mm keeps a camera walked in small steps on the same spectrum.
        z_support_m = max(0.04, math.ceil((abs(z_m) + 1e-9) / 0.01) * 0.01)
        key = (
            self._optical_cache_token,
            relay_n,
            plan.propagation_grid_n,
            float(wavelength_m),
            round(float(plan.k_perp_m_inv), 3),
            tuple(round(float(v), 12) for v in decentre_m),
            z_support_m,
        )
        with self._lock:
            cached = self._axicon_spectrum_cache
        if cached is not None and cached[0] == key:
            spectrum, stage_meta = cached[1], cached[2]
        else:
            window_m = float(geometry.simulation_window_mm) * 1e-3
            self._axicon_stage_rebuilt = True
            try:
                spectrum = build_resolved_axicon_spectrum(
                    field_on_axicon,
                    window_m=window_m,
                    fine_n=int(plan.propagation_grid_n),
                    wavelength_m=wavelength_m,
                    k_perp_m_inv=float(plan.k_perp_m_inv),
                    decentre_m=tuple(float(v) for v in decentre_m),
                    z_support_m=z_support_m,
                    minimum_retained_spectral_power=0.98,
                )
            except RuntimeError as exc:
                raise ProviderError(
                    f"The camera at z={z_mm:g} mm is too far for the {geometry.simulation_window_mm:g} mm model "
                    "window: the light cone leaves the window before it reaches the camera. Move the camera "
                    f"closer. ({exc})"
                ) from exc
            axicon_meta = {
                "base_angle_rad": math.radians(float(plan.base_angle_deg)),
                "refractive_index": float(geometry.axicon_refractive_index),
                "external_index": float(geometry.axicon_external_index),
                "exact_kr_m_inv": float(plan.k_perp_m_inv),
                "decentre_m": tuple(float(v) for v in decentre_m),
                "tip_model": "sharp",
                "transmission": "exp(-i k_perp r) on the resolved axicon grid",
            }
            stage_meta = {
                "axicon_propagation_method": (
                    "multirate fixed-window Fourier handoff + fixed-support angular spectrum "
                    "(ported from real_bmg_digital_twin_correction.propagate_route)"
                ),
                "axicon_plan": plan.as_dict(),
                "axicon_model": axicon_meta,
                "axicon_k_perp_source": geometry.axicon_k_perp_source
                if geometry.axicon_k_perp_m_inv is not None else "model base angle",
                "retained_spectral_power_fraction": float(spectrum.retained_spectral_power_fraction),
                "detector_pixel_um": float(geometry.simulation_window_mm) * 1000.0 / self._detector_n_for(relay_n),
                "z_support_mm": z_support_m * 1e3,
            }
            with self._lock:
                if key[0] == self._optical_cache_token:
                    self._axicon_spectrum_cache = (key, spectrum, stage_meta)
        observed = resolved_field_at_z(spectrum, z_m)
        metadata = dict(stage_meta)
        metadata["propagation_integration_factor"] = int(plan.propagation_grid_n // self._detector_n_for(relay_n))
        metadata["propagation_grid_n"] = int(plan.propagation_grid_n)
        return np.asarray(observed, dtype=np.complex128), metadata

    def _forward_complex_field(
        self,
        state: ExperimentState,
        *,
        z_mm: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        with self._lock:
            hidden = self._hidden
            scenario = self._scenario
            alignment = self._alignment
            optical = {k: v.copy() for k, v in self._optical_cast_phase.items()}
            cast_states = {k: copy.deepcopy(v) for k, v in self._cast_slm_states.items()}
            hashes = dict(self._cast_hashes)
            geometry = self.geometry

        missing = [name for name in ("SLM1", "SLM2") if name not in optical]
        if missing:
            raise ProviderError(
                "Virtual bench has no cast phase for "
                + ", ".join(missing)
                + ". Cast both virtual SLMs before acquiring a frame."
            )

        wavelength_m = float(state.system.wavelength_nm) * 1e-9
        with self._lock:
            relay_token = self._optical_cache_token
        relay_key = (relay_token, self.grid_n, wavelength_m, 2.0)

        def compute_relay():
            grid = self._grid()
            beam_radius_m = 2.0e-3  # existing repo assumption; retained as uncalibrated model input
            beam_error = GaussianBeamError(
                decentre_m=tuple(hidden.beam_decentre_m),
                pointing_rad=tuple(hidden.beam_pointing_rad),
            )
            field, beam_meta = gaussian_input_field(
                grid,
                wavelength_m=wavelength_m,
                canonical_radius_m=beam_radius_m,
                error=beam_error,
            )
            beam_meta = _finite_beam_metadata(beam_meta)

            pupil_radius_m = max(
                0.5e-3,
                0.5 * float(cast_states.get("SLM1", state.slm1).phase.pupil_diameter_mm) * 1e-3,
            )
            hidden_phase = self._hidden_input_phase(grid, wavelength_m, pupil_radius_m)
            field = np.asarray(field, dtype=np.complex128) * np.exp(1j * hidden_phase)

            phi1 = _phase_sample_to_model(optical["SLM1"], cast_states.get("SLM1", state.slm1), grid)
            field = field * np.exp(1j * phi1)

            d12_m = float(geometry.slm1_to_slm2_mm) * 1e-3
            if abs(d12_m) > 1e-15:
                field = angular_spectrum_propagate_bl(
                    field,
                    grid,
                    wavelength_m,
                    d12_m,
                    n_medium=1.0,
                    bandlimit=True,
                    include_evanescent=True,
                )

            phi2 = _phase_sample_to_model(optical["SLM2"], cast_states.get("SLM2", state.slm2), grid)
            field = np.asarray(field, dtype=np.complex128) * np.exp(1j * phi2)
            field, relay_meta = self._apply_4f_relay(
                field, grid, wavelength_m=wavelength_m, geometry=geometry
            )
            beam_meta.update(relay_meta)
            return field, beam_meta

        field, beam_meta = self._cached_relay(relay_key, compute_relay)

        # Deliberate claim boundary: the known 300 mm lens-centre separation and
        # unknown pinhole axial position are not forced into an ideal 4F formula.
        # VirtualSlmProvider has removed locked blaze terms from the optical phase,
        # so this field is the selected-order surrogate handoff.
        effective_axicon_decentre = (
            float(hidden.axicon_decentre_m[0]) + float(alignment.axicon_x_um) * 1e-6,
            float(hidden.axicon_decentre_m[1]) + float(alignment.axicon_y_um) * 1e-6,
        )
        observed, stage_meta = self._observe_after_axicon(
            field,
            geometry=geometry,
            wavelength_m=wavelength_m,
            decentre_m=effective_axicon_decentre,
            z_mm=float(z_mm),
            beam_radius_mm=2.0,
        )
        axicon_meta = stage_meta["axicon_model"]
        warnings: list[str] = [
            "4F relay modelled from the operator-reported 300 mm spacings: unit magnification and a "
            "180-degree image rotation. Order selection is ideal unless a stop diameter is entered; "
            "lens aberrations and the stop's centring are not modelled.",
        ]

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
            "wavelength_nm": float(state.system.wavelength_nm),
            "z_mm": float(z_mm),
            "z_reference": geometry.camera_z_reference,
            "relay_mode": geometry.relay_mode,
            "geometry_status": "PARTIALLY_BOUND_NOT_BENCH_CALIBRATED",
            "slm1_to_slm2_mm": geometry.slm1_to_slm2_mm,
            "slm1_to_slm2_source": geometry.slm1_to_slm2_source,
            "lens_to_lens_separation_mm": geometry.lens_to_lens_separation_mm,
            "pinhole_axial_position_mm": geometry.pinhole_axial_position_mm,
            "relay_magnification": geometry.relay_magnification,
            "relay_image_rotation_deg": 180.0 if geometry.relay_inverts_image else 0.0,
            "fourier_aperture_diameter_mm": geometry.fourier_aperture_diameter_mm,
            "objective_demagnification": geometry.objective_demagnification,
            "axicon_reported_label": geometry.axicon_reported_label,
            "axicon_model": axicon_meta,
            "beam_model": beam_meta,
            "cast_phase_hashes": hashes,
            "alignment_command": asdict(alignment),
            "warnings": warnings,
            "hidden_truth_exposed": False,
            **stage_meta,
        }
        return np.asarray(observed, dtype=np.complex128), metadata

    def acquire_intensity(
        self,
        state: ExperimentState,
        *,
        z_mm: float,
        exposure_us: float = 1000.0,
        gain: float = 0.0,
        full_scale: float = 4095.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        intensity, metadata = self._model_intensity(state, float(z_mm))
        from .virtual_radiometry import linear_camera_signal
        camera, radiometry = linear_camera_signal(
            intensity, exposure_us=exposure_us, gain=gain,
        )
        metadata.update(radiometry)

        if self.realistic_camera:
            # Deterministic per-frame noise: seeded from scenario + frame counter,
            # never from hidden values leaked through metadata.
            with self._lock:
                counter = self._frame_counter
            rng = np.random.default_rng(int(self._scenario.seed) * 1000003 + counter)
            background = 12.0
            read_sigma = 2.0
            shot = rng.normal(0.0, np.sqrt(np.maximum(camera, 0.0)) * 0.20)
            power_scale = max(0.0, 1.0 + float(rng.normal(0.0, 0.004)))
            camera = camera * power_scale + background + shot + rng.normal(0.0, read_sigma, camera.shape)
            metadata["virtual_camera_model"] = "background + shot-like noise + read noise + 0.4% rms source-power fluctuation"
            metadata["virtual_power_scale"] = power_scale
        else:
            metadata["virtual_camera_model"] = "clean deterministic intensity sampling"

        camera = np.clip(camera, 0.0, float(full_scale)).astype(np.float64)
        metadata.update(
            {
                "quantitative_valid": True,
                "quantitative_scope": "simulated numerical intensity only",
                "full_scale": float(full_scale),
                "exposure_semantics": "relative virtual scaling; not Beamage radiometric calibration",
            }
        )
        with self._lock:
            self._frame_counter += 1
        return camera, metadata


class VirtualCameraProvider(CameraProvider):
    name = "virtual"
    implementation_status = "SIMULATED"
    declared_data_kind = "SYNTHETIC"
    supports_configuration = True

    def __init__(
        self,
        engine: VirtualBenchEngine,
        state_supplier: Callable[[], ExperimentState],
        *,
        exposure_us: float = 1000.0,
        gain: float = 0.0,
        full_scale: float = 4095.0,
    ):
        self.engine = engine
        self.state_supplier = state_supplier
        self._exposure_us = float(exposure_us)
        self._gain = float(gain)
        self._full_scale = float(full_scale)
        self._connected = False
        self._running = False
        self._counter = 0
        self.device_serial = "VIRTUAL-DIGITAL-TWIN"

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> DeviceStatus:
        self._connected = True
        return DeviceStatus(
            self.name,
            ConnectionState.CONNECTED,
            self.implementation_status,
            "Virtual optical camera connected to the repository digital twin.",
        )

    def disconnect(self) -> DeviceStatus:
        self._running = False
        self._connected = False
        return DeviceStatus(
            self.name,
            ConnectionState.DISCONNECTED,
            self.implementation_status,
            "Virtual camera disconnected.",
        )

    def start(self) -> None:
        if not self._connected:
            raise ProviderError("Connect the virtual camera before starting acquisition.")
        self._running = True

    def stop(self) -> None:
        self._running = False

    def configure(self, *, exposure_us: float, gain: float) -> None:
        self._exposure_us = float(exposure_us)
        self._gain = float(gain)

    # The virtual camera derives z from the state it is handed, so the frame's
    # z is not an independent observation of the camera position.
    reports_independent_z = False

    def acquire_frame(self, *, fresh: bool = True, timeout_s: float = 2.0) -> CameraFrame:
        if not self._connected:
            raise ProviderError("Virtual camera is disconnected.")
        state = self.state_supplier()
        z_mm = 0.0 if state.camera.current_z_mm is None else float(state.camera.current_z_mm)
        data, metadata = self.engine.acquire_intensity(
            state,
            z_mm=z_mm,
            exposure_us=self._exposure_us,
            gain=self._gain,
            full_scale=self._full_scale,
        )
        self._counter += 1
        metadata.update(
            {
                "operating_mode": OperatingMode.VIRTUAL_LAB.value,
                "provider": self.name,
                "frame_sequence": self._counter,
            }
        )
        return CameraFrame(
            data=data,
            frame_id=f"virtual-{self.engine.scenario.scenario_id}-{self._counter:06d}",
            timestamp_utc=utc_now(),
            provider=self.name,
            exposure_us=self._exposure_us,
            gain=self._gain,
            full_scale=self._full_scale,
            z_mm=z_mm,
            data_kind="SYNTHETIC",
            metadata=metadata,
        )


class VirtualSlmProvider(SlmProvider):
    """Dry-run SLM provider whose cast target is the VirtualBenchEngine only."""

    def __init__(
        self,
        engine: VirtualBenchEngine,
        state_supplier: Callable[[], ExperimentState],
        *,
        phase_service: PhaseService | None = None,
    ):
        self.engine = engine
        self.state_supplier = state_supplier
        self.phase_service = phase_service or PhaseService()
        self._connected = False
        self._messages: dict[str, str] = {
            "SLM1": "virtual SLM not connected",
            "SLM2": "virtual SLM not connected",
        }

    def connect(self, config: AppConfig) -> list[str]:
        self._connected = True
        for name in self._messages:
            self._messages[name] = "VIRTUAL CONNECTED — no HEDS handle opened"
        return [
            "Virtual SLM provider connected.",
            "SLM1 virtual target ready; physical HEDS untouched.",
            "SLM2 virtual target ready; physical HEDS untouched.",
        ]

    def _selected_order_phase(self, name: str) -> np.ndarray:
        # The present relay geometry is incomplete.  The virtual bench therefore
        # models a selected-order handoff.  Use the same authoritative phase
        # composer with blaze disabled on a COPY of the config.  PhaseService
        # validates ExperimentState and re-enables the hardware-locked carrier;
        # using it here accidentally propagated the blaze on both virtual SLMs.
        # The full hardware-command phase remains untouched and provenance-linked.
        from slm_lab_control.phase import compose_phase

        state = self.state_supplier()
        phase_cfg = copy.deepcopy(getattr(state, name.lower()).phase)
        phase_cfg.switches.blaze = False
        return np.asarray(compose_phase(phase_cfg).phase_rad, dtype=np.float64)

    def cast(
        self,
        name: str,
        result: Any,
        transfer_mode: str,
        log_png: Path,
    ) -> str:
        if not self._connected:
            raise ProviderError("Connect virtual SLMs before casting.")
        key = str(name).upper()
        optical_phase = self._selected_order_phase(key)
        full_phase = np.asarray(result.phase_rad, dtype=np.float64)
        self.engine.set_cast_phase(
            key,
            full_phase,
            optical_phase,
            full_phase_hash=phase_sha256(full_phase),
            slm_state=getattr(self.state_supplier(), key.lower()),
        )
        message = (
            f"{key}: VIRTUAL CAST to digital twin; no HEDS command issued "
            f"(relay={self.engine.geometry.relay_mode})."
        )
        self._messages[key] = message
        return message

    def blank(self, config: AppConfig, names: Iterable[str], output_folder: Path) -> list[str]:
        if not self._connected:
            raise ProviderError("Connect virtual SLMs before blanking.")
        messages = []
        for raw in names:
            name = str(raw).upper()
            phase = config.slm1 if name == "SLM1" else config.slm2
            self.engine.blank(name, (phase.geometry.height_px, phase.geometry.width_px))
            message = f"{name}: VIRTUAL BLANK; physical HEDS untouched."
            self._messages[name] = message
            messages.append(message)
        return messages

    def close(self) -> str:
        self._connected = False
        self.engine.clear_cast_phases()
        return "Virtual SLM provider closed; no physical hardware was addressed."

    def disconnect(self) -> str:
        return self.close()

    def status(self, name: str) -> DeviceStatus:
        key = str(name).upper()
        return DeviceStatus(
            "virtual-slm",
            ConnectionState.CONNECTED if self._connected else ConnectionState.DISCONNECTED,
            "SIMULATED",
            self._messages.get(key, "virtual SLM"),
        )


class RecordedReadOnlySlmProvider(SlmProvider):
    """Safety barrier: historical camera data cannot respond to new SLM commands."""

    def __init__(self):
        self._connected = False

    def connect(self, config: AppConfig) -> list[str]:
        self._connected = True
        return ["Recorded Lab is read-only; historical SLM state may be inspected but not recast."]

    def cast(self, name: str, result: Any, transfer_mode: str, log_png: Path) -> str:
        raise ProviderError("RECORDED LAB is read-only; a historical dataset cannot respond to a new SLM cast.")

    def blank(self, config: AppConfig, names: Iterable[str], output_folder: Path) -> list[str]:
        raise ProviderError("RECORDED LAB is read-only; blanking hardware is disabled.")

    def close(self) -> str:
        self._connected = False
        return "Recorded Lab read-only SLM provider closed."

    def disconnect(self) -> str:
        return self.close()

    def status(self, name: str) -> DeviceStatus:
        return DeviceStatus(
            "recorded-read-only",
            ConnectionState.CONNECTED if self._connected else ConnectionState.DISCONNECTED,
            "REPLAY",
            "READ ONLY — recorded laboratory state",
        )


class VirtualStageProvider(StageProvider):
    name = "virtual-stage"

    def __init__(self):
        self._position: float | None = None

    def connect(self) -> DeviceStatus:
        return DeviceStatus(
            self.name,
            ConnectionState.CONNECTED,
            "SIMULATED",
            "Virtual z stage ready.",
        )

    def request_move(self, z_mm: float) -> MoveRequest:
        self._position = float(z_mm)
        return MoveRequest(
            target_z_mm=self._position,
            prompt=f"Virtual camera moved to z={self._position:.3f} mm.",
            requires_operator_confirmation=False,
        )

    def confirm_position(self, z_mm: float) -> float:
        self._position = float(z_mm)
        return self._position

    @property
    def position_mm(self) -> float | None:
        return self._position


def objective_from_metrics(
    metrics: Sequence[BeamMetrics],
    *,
    frame_shape_yx: tuple[int, int],
) -> ObjectiveBreakdown:
    if not metrics:
        raise ValueError("At least one analysed plane is required.")

    family = metrics[0].family
    if any(item.family != family for item in metrics):
        raise ValueError("All z-stack planes must use the same analysis family.")

    centres = np.asarray([item.centre_yx_px for item in metrics], dtype=float)
    centre_mean = np.mean(centres, axis=0)
    centre_rms_px = float(np.sqrt(np.mean(np.sum((centres - centre_mean) ** 2, axis=1))))

    edge = float(np.mean([float(item.values.get("edge_power_fraction") or 0.0) for item in metrics]))
    saturation = float(np.mean([float(item.values.get("saturation_fraction") or 0.0) for item in metrics]))

    if family == "vortex_bessel":
        radii = np.asarray(
            [float(item.values.get("principal_ring_radius_px") or 0.0) for item in metrics],
            dtype=float,
        )
        scale = max(float(np.mean(radii[radii > 0])) if np.any(radii > 0) else 1.0, 1.0)
        walk = centre_rms_px / scale
        ring_cv = float(np.std(radii) / max(abs(float(np.mean(radii))), EPS))
        eccentricity = float(np.mean([float(item.values.get("ring_eccentricity") or 0.0) for item in metrics]))
        azimuthal_cv = float(np.mean([float(item.values.get("azimuthal_cv") or 0.0) for item in metrics]))
        dark_core = float(np.mean([float(item.values.get("dark_core_fraction") or 0.0) for item in metrics]))
        terms = {
            "beam_walk": walk,
            "ring_radius_cv": ring_cv,
            "ring_eccentricity": eccentricity,
            "azimuthal_cv": azimuthal_cv,
            "dark_core_fraction": dark_core,
            "edge_power_fraction": edge,
            "saturation_fraction": saturation,
        }
        total = (
            3.0 * walk
            + 1.0 * ring_cv
            + 1.5 * eccentricity
            + 1.5 * azimuthal_cv
            + 0.5 * dark_core
            + 4.0 * edge
            + 4.0 * saturation
        )
    else:
        widths = np.asarray(
            [
                float(item.values.get("fwhm_major_px") or item.values.get("width_x_sigma_px") or 1.0)
                for item in metrics
            ],
            dtype=float,
        )
        scale = max(float(np.mean(widths)), 1.0)
        walk = centre_rms_px / scale
        width_cv = float(np.std(widths) / max(abs(float(np.mean(widths))), EPS))
        ellipticity_error = float(
            np.mean([abs(float(item.values.get("ellipticity") or 1.0) - 1.0) for item in metrics])
        )
        terms = {
            "beam_walk": walk,
            "width_cv": width_cv,
            "ellipticity_error": ellipticity_error,
            "edge_power_fraction": edge,
            "saturation_fraction": saturation,
        }
        total = 3.0 * walk + 1.0 * width_cv + 1.5 * ellipticity_error + 4.0 * edge + 4.0 * saturation

    if not np.isfinite(total):
        total = 1e9
    return ObjectiveBreakdown(float(total), terms, family)


def capture_virtual_z_stack(
    controller: Any,
    z_plan_mm: Sequence[float],
    *,
    metric_engine: MetricEngine | None = None,
    frame_callback: Callable[[CameraFrame], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> VirtualStackResult:
    """Acquire a multi-plane stack through the same CameraFrame provider route."""

    z_values = tuple(float(z) for z in z_plan_mm)
    if len(z_values) < 2:
        raise ValueError("A blind propagation objective requires at least two z planes.")
    metrics_engine = metric_engine or MetricEngine()

    if getattr(controller, "operating_mode", None) != OperatingMode.VIRTUAL_LAB:
        raise RuntimeError("Virtual z-stack acquisition requires VIRTUAL LAB mode.")

    if controller.camera_provider is None:
        raise RuntimeError("Virtual camera provider is not installed.")
    if not controller.camera_provider.connected:
        controller.connect_camera()
    controller.start_camera()

    frames: list[CameraFrame] = []
    metrics: list[BeamMetrics] = []
    for index, z_mm in enumerate(z_values, 1):
        if cancelled is not None and cancelled():
            raise InterruptedError("Virtual z-stack cancelled.")
        controller.move_stage(z_mm)
        frame = controller.acquire_frame(fresh=True)
        analysed = metrics_engine.analyse(frame, controller.store.snapshot())
        frames.append(frame)
        metrics.append(analysed)
        if frame_callback is not None:
            frame_callback(frame)
        if progress_callback is not None:
            progress_callback(
                {
                    "kind": "z_plane",
                    "index": index,
                    "count": len(z_values),
                    "z_mm": z_mm,
                    "frame_id": frame.frame_id,
                }
            )

    objective = objective_from_metrics(metrics, frame_shape_yx=frames[0].shape_yx)
    return VirtualStackResult(z_values, tuple(frames), tuple(metrics), objective)


@dataclass
class BlindCorrectionResult:
    initial_objective: float
    final_objective: float
    improvement_fraction: float
    accepted_runs: list[dict[str, Any]]
    final_stack: VirtualStackResult
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "improvement_fraction": self.improvement_fraction,
            "accepted_runs": self.accepted_runs,
            "final_stack": self.final_stack.summary(),
            "cancelled": self.cancelled,
        }


class BlindCorrectionRunner:
    """Sequential sensorless modal probing using intensity stacks only."""

    DEFAULT_PARAMETERS = (
        "defocus",
        "astig_x",
        "astig_xy",
        "coma_x",
        "coma_y",
        "spherical",
    )

    def __init__(self, controller: Any, *, metric_engine: MetricEngine | None = None):
        self.controller = controller
        self.metric_engine = metric_engine or MetricEngine()

    def _stack(
        self,
        z_plan_mm: Sequence[float],
        *,
        frame_callback: Callable[[CameraFrame], None] | None,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        cancelled: Callable[[], bool] | None,
    ) -> VirtualStackResult:
        return capture_virtual_z_stack(
            self.controller,
            z_plan_mm,
            metric_engine=self.metric_engine,
            frame_callback=frame_callback,
            progress_callback=progress_callback,
            cancelled=cancelled,
        )

    def run(
        self,
        z_plan_mm: Sequence[float],
        *,
        targets: Sequence[str] = ("SLM2",),
        parameters: Sequence[str] | None = None,
        probe_amplitude_waves: float = 0.15,
        passes: int = 2,
        seed: int = 42,
        frame_callback: Callable[[CameraFrame], None] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> BlindCorrectionResult:
        if self.controller.operating_mode is not OperatingMode.VIRTUAL_LAB:
            raise RuntimeError("Blind virtual correction requires VIRTUAL LAB mode.")
        target_names = tuple(str(name).upper() for name in targets)
        if not target_names or any(name not in {"SLM1", "SLM2"} for name in target_names):
            raise ValueError("targets must contain SLM1 and/or SLM2.")
        mode_parameters = tuple(parameters or self.DEFAULT_PARAMETERS)
        amplitude = abs(float(probe_amplitude_waves))
        if amplitude <= 0:
            raise ValueError("probe_amplitude_waves must be positive.")
        passes = max(1, int(passes))

        # Explicit initial virtual cast: configured state -> simulated hardware.
        self.controller.cast(target_names, persist=False)
        # Ensure both panels have a simulated cast even if only one is being
        # optimised.  Their current configured phases are part of the optical path.
        other = tuple(name for name in ("SLM1", "SLM2") if self.controller.virtual_engine.cast_hash(name) is None)
        if other:
            self.controller.cast(other, persist=False)

        initial = self._stack(
            z_plan_mm,
            frame_callback=frame_callback,
            progress_callback=progress_callback,
            cancelled=cancelled,
        )
        accepted_runs: list[dict[str, Any]] = []
        optimiser = SensorlessOptimiser(self.controller.store)
        active_run = None

        try:
            for pass_index in range(passes):
                pass_amp = amplitude / (2.0**pass_index)
                deltas = (-2.0 * pass_amp, -pass_amp, pass_amp, 2.0 * pass_amp)
                for slm_name in target_names:
                    for parameter in mode_parameters:
                        if cancelled is not None and cancelled():
                            raise InterruptedError("Blind correction cancelled.")
                        run = optimiser.plan(
                            parameter,
                            deltas,
                            slm_name=slm_name,
                            seed=int(seed) + pass_index,
                            higher_is_better=False,
                            minimum_fractional_improvement=0.002,
                            max_control_drift_fraction=0.05,
                        )
                        active_run = run
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "kind": "mode_start",
                                    "pass": pass_index + 1,
                                    "passes": passes,
                                    "slm": slm_name,
                                    "parameter": parameter,
                                    "probe_amplitude_waves": pass_amp,
                                }
                            )

                        for trial in list(run.trials):
                            if cancelled is not None and cancelled():
                                raise InterruptedError("Blind correction cancelled.")
                            optimiser.apply_trial(run, trial.trial_id)
                            self.controller.cast((slm_name,), persist=False)
                            stack = self._stack(
                                z_plan_mm,
                                frame_callback=frame_callback,
                                progress_callback=None,
                                cancelled=cancelled,
                            )
                            optimiser.record_score(
                                run,
                                trial.trial_id,
                                stack.objective.total,
                                capture_id=f"virtual:{stack.frames[-1].frame_id}",
                            )
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "kind": "candidate",
                                        "slm": slm_name,
                                        "parameter": parameter,
                                        "role": trial.role,
                                        "command": trial.command,
                                        "score": stack.objective.total,
                                    }
                                )

                        verification = optimiser.recommend(run)
                        self.controller.cast((slm_name,), persist=False)
                        verify_stack = self._stack(
                            z_plan_mm,
                            frame_callback=frame_callback,
                            progress_callback=None,
                            cancelled=cancelled,
                        )
                        accepted = optimiser.verify_and_decide(
                            run,
                            verify_stack.objective.total,
                            capture_id=f"virtual-verify:{verify_stack.frames[-1].frame_id}",
                        )
                        # verify_and_decide can rollback the ExperimentState.  Recast
                        # once so the virtual hardware always matches accepted state.
                        self.controller.cast((slm_name,), persist=False)
                        accepted_runs.append(
                            {
                                "run_id": run.run_id,
                                "pass": pass_index + 1,
                                "slm": slm_name,
                                "parameter": parameter,
                                "accepted": bool(accepted),
                                "accepted_command": run.accepted_command,
                                "verification_score": verify_stack.objective.total,
                            }
                        )
                        active_run = None
        except InterruptedError:
            if active_run is not None:
                # A candidate or recommended command may be on the virtual
                # panel when cancellation arrives. Never leave an unverified
                # trial cast as though it were the accepted correction.
                optimiser.rollback(active_run, reason="Virtual correction cancelled before verification")
                self.controller.cast((active_run.slm_name,), persist=False)
            final = self._stack(
                z_plan_mm,
                frame_callback=frame_callback,
                progress_callback=None,
                cancelled=None,
            )
            return BlindCorrectionResult(
                initial_objective=initial.objective.total,
                final_objective=final.objective.total,
                improvement_fraction=(
                    (initial.objective.total - final.objective.total)
                    / max(abs(initial.objective.total), EPS)
                ),
                accepted_runs=accepted_runs,
                final_stack=final,
                cancelled=True,
            )

        try:
            final = self._stack(
                z_plan_mm,
                frame_callback=frame_callback,
                progress_callback=progress_callback,
                cancelled=cancelled,
            )
        except InterruptedError:
            # Accepted modes remain accepted, but the cancelled final readout
            # must still report the state actually left on the virtual SLMs.
            final = self._stack(
                z_plan_mm,
                frame_callback=frame_callback,
                progress_callback=None,
                cancelled=None,
            )
            return BlindCorrectionResult(
                initial_objective=initial.objective.total,
                final_objective=final.objective.total,
                improvement_fraction=(initial.objective.total - final.objective.total)
                / max(abs(initial.objective.total), EPS),
                accepted_runs=accepted_runs,
                final_stack=final,
                cancelled=True,
            )
        improvement = (
            (initial.objective.total - final.objective.total)
            / max(abs(initial.objective.total), EPS)
        )
        return BlindCorrectionResult(
            initial_objective=initial.objective.total,
            final_objective=final.objective.total,
            improvement_fraction=float(improvement),
            accepted_runs=accepted_runs,
            final_stack=final,
            cancelled=False,
        )


@dataclass
class AlignmentAssistResult:
    before_objective: float
    after_objective: float
    improvement_fraction: float
    alignment_command: AlignmentCommand
    trials: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_objective": self.before_objective,
            "after_objective": self.after_objective,
            "improvement_fraction": self.improvement_fraction,
            "alignment_command": asdict(self.alignment_command),
            "trials": list(self.trials),
        }


class AlignmentAssistRunner:
    """Separate, explicitly enabled virtual axicon x/y alignment search."""

    def __init__(self, controller: Any, *, metric_engine: MetricEngine | None = None):
        self.controller = controller
        self.metric_engine = metric_engine or MetricEngine()

    def run(
        self,
        z_plan_mm: Sequence[float],
        *,
        candidate_offsets_um: Sequence[float] = (-100.0, -50.0, -25.0, 0.0, 25.0, 50.0, 100.0),
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        frame_callback: Callable[[CameraFrame], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> AlignmentAssistResult:
        if self.controller.operating_mode is not OperatingMode.VIRTUAL_LAB:
            raise RuntimeError("Alignment Assist is available only in VIRTUAL LAB.")
        engine = self.controller.virtual_engine
        baseline_command = engine.alignment_command
        before = capture_virtual_z_stack(
            self.controller,
            z_plan_mm,
            metric_engine=self.metric_engine,
            frame_callback=frame_callback,
            cancelled=cancelled,
        )
        trials: list[dict[str, Any]] = []
        current = baseline_command

        for axis in ("x", "y"):
            best_score = math.inf
            best_value = getattr(current, f"axicon_{axis}_um")
            for value in candidate_offsets_um:
                if cancelled is not None and cancelled():
                    raise InterruptedError("Alignment Assist cancelled.")
                kwargs = {
                    "axicon_x_um": current.axicon_x_um,
                    "axicon_y_um": current.axicon_y_um,
                }
                kwargs[f"axicon_{axis}_um"] = float(value)
                engine.set_alignment_command(**kwargs)
                stack = capture_virtual_z_stack(
                    self.controller,
                    z_plan_mm,
                    metric_engine=self.metric_engine,
                    frame_callback=frame_callback,
                    cancelled=cancelled,
                )
                row = {
                    "axis": axis,
                    "command_um": float(value),
                    "objective": stack.objective.total,
                }
                trials.append(row)
                if progress_callback is not None:
                    progress_callback({"kind": "alignment_candidate", **row})
                if stack.objective.total < best_score:
                    best_score = stack.objective.total
                    best_value = float(value)
            kwargs = {
                "axicon_x_um": current.axicon_x_um,
                "axicon_y_um": current.axicon_y_um,
            }
            kwargs[f"axicon_{axis}_um"] = best_value
            current = AlignmentCommand(**kwargs)
            engine.set_alignment_command(
                axicon_x_um=current.axicon_x_um,
                axicon_y_um=current.axicon_y_um,
            )

        after = capture_virtual_z_stack(
            self.controller,
            z_plan_mm,
            metric_engine=self.metric_engine,
            frame_callback=frame_callback,
            cancelled=cancelled,
        )
        return AlignmentAssistResult(
            before_objective=before.objective.total,
            after_objective=after.objective.total,
            improvement_fraction=(
                (before.objective.total - after.objective.total)
                / max(abs(before.objective.total), EPS)
            ),
            alignment_command=current,
            trials=trials,
        )
