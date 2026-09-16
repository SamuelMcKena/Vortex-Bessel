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
from vbb_study.digital_twin.vortex_wavefront_errors import unit_rms_zernike
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


TWOPI = 2.0 * np.pi
EPS = 1e-30


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
    simulation_window_mm: float = 8.0
    preview_grid_n: int = 256
    validation_grid_n: int = 512

    # Present GUI/HEDS phase geometry.
    slm_pixel_pitch_um: float = 8.0

    # The existing CSLM route contains 0.040 mm, but its own documentation marks
    # it as diagnostic/placeholder geometry.  Keep it editable and labelled.
    slm1_to_slm2_mm: float = 0.040
    slm1_to_slm2_source: str = "repo_placeholder_unmeasured"

    # User-reported bench values.  With f=300 mm lenses and 300 mm lens-centre
    # separation this is NOT silently treated as an ideal symmetric 4F.
    lens1_focal_length_mm: float = 300.0
    lens2_focal_length_mm: float = 300.0
    lens_to_lens_separation_mm: float = 300.0
    relay_geometry_source: str = "user_reported_f300_lenses_and_confirmed_300mm_lens_separation"
    pinhole_role: str = "+1 diffraction-order selection"
    pinhole_axial_position_mm: float | None = None

    # The optic is reported as a "20 degree Thorlabs axicon".  The repository's
    # current exact-refractive model requires a *base angle*.  Until the part
    # number / convention is bound, retain the old 2 deg model value only as an
    # explicit simulation placeholder.
    axicon_reported_label: str = "Thorlabs 20° axicon"
    axicon_model_base_angle_deg: float = 2.0
    axicon_model_angle_source: str = "repo_placeholder_not_bound_to_reported_20deg_label"
    axicon_refractive_index: float = 1.458
    axicon_external_index: float = 1.0

    # The physical camera reference is not yet established.  Virtual z is
    # therefore relative to the post-axicon model plane.
    camera_z_reference: str = "virtual post-axicon plane; physical camera z=0 UNCALIBRATED"

    # Until the pinhole axial location and full relay distances are measured, the
    # model uses an ideal selected-order handoff rather than inventing a 4F.
    relay_mode: str = "ideal_selected_order_surrogate"

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
            "Lens-centre to lens-centre separation confirmed as 300 mm.",
            "Pinhole is between the lenses and selects +1; exact axial position is unknown.",
            "Physical optic reported as a Thorlabs 20° axicon; model base-angle convention remains unbound.",
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
        self._cast_hashes: dict[str, str] = {}
        self._alignment = AlignmentCommand()
        self._frame_counter = 0

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

    def set_geometry(self, **updates: Any) -> None:
        with self._lock:
            self.geometry = replace(self.geometry, **updates)

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
            self._cast_hashes[key] = full_phase_hash or phase_sha256(full)

    def blank(self, name: str, shape: tuple[int, int]) -> None:
        zeros = np.zeros(shape, dtype=np.float64)
        self.set_cast_phase(name, zeros, zeros)

    def cast_hash(self, name: str) -> str | None:
        return self._cast_hashes.get(str(name).upper())

    def _grid(self) -> dict[str, Any]:
        n = self.grid_n
        width_m = float(self.geometry.simulation_window_mm) * 1e-3
        return make_xy_grid(n, width_m / n)

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
            hashes = dict(self._cast_hashes)
            geometry = self.geometry

        missing = [name for name in ("SLM1", "SLM2") if name not in optical]
        if missing:
            raise ProviderError(
                "Virtual bench has no cast phase for "
                + ", ".join(missing)
                + ". Cast both virtual SLMs before acquiring a frame."
            )

        grid = self._grid()
        wavelength_m = float(state.system.wavelength_nm) * 1e-9
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

        pupil_radius_m = max(
            0.5e-3,
            0.5 * float(state.slm1.phase.pupil_diameter_mm) * 1e-3,
        )
        hidden_phase = self._hidden_input_phase(grid, wavelength_m, pupil_radius_m)
        field = np.asarray(field, dtype=np.complex128) * np.exp(1j * hidden_phase)

        phi1 = _phase_sample_to_model(optical["SLM1"], state.slm1, grid)
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

        phi2 = _phase_sample_to_model(optical["SLM2"], state.slm2, grid)
        field = np.asarray(field, dtype=np.complex128) * np.exp(1j * phi2)

        # Deliberate claim boundary: the known 300 mm lens-centre separation and
        # unknown pinhole axial position are not forced into an ideal 4F formula.
        # VirtualSlmProvider has removed locked blaze terms from the optical phase,
        # so this field is the selected-order surrogate handoff.
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
                post_axicon,
                grid,
                wavelength_m,
                z_m,
                n_medium=1.0,
                bandlimit=True,
                include_evanescent=True,
            )
        else:
            observed = post_axicon

        radial_period_m = TWOPI / max(abs(float(axicon_meta["exact_kr_m_inv"])), EPS)
        sampling_per_radial_period = radial_period_m / max(float(grid["dx"]), EPS)
        warnings: list[str] = []
        if sampling_per_radial_period < 3.0:
            warnings.append(
                f"Axicon radial phase period is sampled by only {sampling_per_radial_period:.2f} "
                "pixels; use Validation quality before interpreting fine structure."
            )
        warnings.append(
            "4F order selection is an ideal selected-order surrogate because the physical pinhole "
            "axial position/full relay geometry are not yet bench-bound."
        )
        warnings.append(
            "The reported '20° Thorlabs axicon' is not yet mapped to the model base-angle "
            f"convention; using {geometry.axicon_model_base_angle_deg:g}° "
            f"({geometry.axicon_model_angle_source})."
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

    def acquire_intensity(
        self,
        state: ExperimentState,
        *,
        z_mm: float,
        exposure_us: float = 1000.0,
        gain: float = 0.0,
        full_scale: float = 4095.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        field, metadata = self._forward_complex_field(state, z_mm=float(z_mm))
        intensity = np.abs(field) ** 2
        peak = float(np.max(intensity))
        if peak <= EPS:
            camera = np.zeros_like(intensity, dtype=np.float64)
        else:
            # Exposure is a relative virtual detector gain.  It does not claim a
            # calibrated Beamage radiometric response.
            scale = 0.72 * float(full_scale) * max(float(exposure_us), 1.0) / 1000.0
            scale *= max(1.0 + float(gain) / 100.0, 0.0)
            camera = intensity / peak * scale

        if self.realistic_camera:
            # Deterministic per-frame noise: seeded from scenario + frame counter,
            # never from hidden values leaked through metadata.
            with self._lock:
                counter = self._frame_counter
            rng = np.random.default_rng(int(self._scenario.seed) * 1000003 + counter)
            background = 12.0
            read_sigma = 2.0
            shot = rng.normal(0.0, np.sqrt(np.maximum(camera, 0.0)) * 0.20)
            camera = camera + background + shot + rng.normal(0.0, read_sigma, camera.shape)
            metadata["virtual_camera_model"] = "background + shot-like noise + read noise"
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
        # models a selected-order handoff.  Remove only the locked blaze term via
        # the SAME PhaseService rather than manually subtracting a grating.
        state = self.state_supplier()
        phase_cfg = getattr(state, name.lower()).phase
        phase_cfg.switches.blaze = False
        return np.asarray(self.phase_service.generate(state).results[name].phase_rad, dtype=np.float64)

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
        except InterruptedError:
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

        final = self._stack(
            z_plan_mm,
            frame_callback=frame_callback,
            progress_callback=progress_callback,
            cancelled=cancelled,
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
