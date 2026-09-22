"""Resolution-aware virtual camera sampling for Virtual Lab experiments.

The optical propagation grid and the reported camera frame are kept explicit.
This makes it possible to compare a Beamage-4M-sized numerical observation with
an intentionally higher-resolution model observation without pretending that
simple image upsampling creates new optical information.

The real Beamage field of view / pixel pitch are not yet bench-bound here.
Therefore ``BEAMAGE_4M`` means *2048 x 2048 pixel-count equivalent*, not a claim
of calibrated physical sensor sampling.  ``MAXIMUM_TO_BEAMAGE`` is the most
useful resolution-comparison mode: propagate on the same high-resolution model
grid as ``MAXIMUM`` and then integrate 2x2 model samples into each 2048-square
virtual detector pixel.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from .virtual_radiometry import linear_camera_signal
from .virtual_lab_experiments import ExperimentalVirtualBenchEngine


class VirtualOutputResolution(str, Enum):
    NATIVE = "MODEL NATIVE"
    LIVE_1M = "LIVE MODEL — 1024×1024"
    BEAMAGE_4M = "BEAMAGE 4M — 2048×2048"
    MAXIMUM = "MAXIMUM MODEL — 4096×4096"
    MAXIMUM_TO_BEAMAGE = "MAX MODEL → BEAMAGE 4M — 4096→2048"


class ResolutionAwareVirtualBenchEngine(ExperimentalVirtualBenchEngine):
    """Experimental virtual bench with explicit propagation/sensor resolution."""

    def __init__(
        self,
        *args,
        output_resolution: VirtualOutputResolution | str = VirtualOutputResolution.NATIVE,
        beamage_n: int = 2048,
        maximum_grid_n: int = 4096,
        live_n: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._output_resolution = (
            output_resolution
            if isinstance(output_resolution, VirtualOutputResolution)
            else VirtualOutputResolution(str(output_resolution))
        )
        self.beamage_n = int(beamage_n)
        self.maximum_grid_n = int(maximum_grid_n)
        # Interactive grid.  512 visibly aliases the q=20 ring family while 2048
        # costs seconds per camera move, so half the Beamage pixel count is the
        # grid an operator can actually drive a live view on.
        self.live_n = int(live_n) if live_n is not None else max(16, self.beamage_n // 2)
        if self.beamage_n < 16:
            raise ValueError("beamage_n is unreasonably small.")
        if self.maximum_grid_n < self.beamage_n:
            raise ValueError("maximum_grid_n must be at least as large as beamage_n.")
        if self.maximum_grid_n % self.beamage_n != 0:
            raise ValueError("maximum_grid_n must be an integer multiple of beamage_n.")

    @property
    def output_resolution(self) -> VirtualOutputResolution:
        return self._output_resolution

    def set_output_resolution(self, mode: VirtualOutputResolution | str) -> None:
        value = mode if isinstance(mode, VirtualOutputResolution) else VirtualOutputResolution(str(mode))
        with self._lock:
            self._output_resolution = value
            self._invalidate_optics()

    def set_quality(self, quality: str) -> None:
        q = str(quality).strip().lower()
        if q == "maximum":
            with self._lock:
                self.quality = "maximum"
                self._invalidate_optics()
            return
        super().set_quality(q)

    @property
    def grid_n(self) -> int:
        mode = self._output_resolution
        if mode is VirtualOutputResolution.LIVE_1M:
            return self.live_n
        if mode is VirtualOutputResolution.BEAMAGE_4M:
            return self.beamage_n
        if mode in {VirtualOutputResolution.MAXIMUM, VirtualOutputResolution.MAXIMUM_TO_BEAMAGE}:
            return self.maximum_grid_n
        if str(self.quality).lower() == "maximum":
            return self.maximum_grid_n
        return super().grid_n

    @property
    def output_n(self) -> int:
        # Every mode except the explicit "ideal camera" MAXIMUM view reports
        # true Beamage pixels; the modes differ only in how finely the SLM relay
        # is computed.  The axicon grid is always chosen to resolve intensity.
        if self._output_resolution is VirtualOutputResolution.MAXIMUM:
            return self.maximum_grid_n
        return self.beamage_n

    @property
    def detector_n(self) -> int:
        return int(self.output_n)

    @property
    def shape_yx(self) -> tuple[int, int]:
        n = self.output_n
        return (n, n)

    @property
    def pixel_size_um(self) -> float:
        # Physical field of view is still the model simulation window.  This is
        # not asserted to be the actual Beamage pixel pitch until calibrated.
        return float(self.geometry.simulation_window_mm) * 1000.0 / float(self.output_n)

    def resolution_summary(self) -> dict[str, Any]:
        return {
            "output_resolution_mode": self._output_resolution.value,
            "propagation_grid_yx": [self.grid_n, self.grid_n],
            "output_frame_yx": [self.output_n, self.output_n],
            "beamage_pixel_count_reference_yx": [self.beamage_n, self.beamage_n],
            "live_model_grid_yx": [self.live_n, self.live_n],
            "maximum_model_grid_yx": [self.maximum_grid_n, self.maximum_grid_n],
            "physical_sampling_status": "PIXEL_COUNT_EQUIVALENT_ONLY_NOT_CAMERA_FOV_CALIBRATED",
        }

    def _sensor_integrate(self, intensity: np.ndarray) -> tuple[np.ndarray, str]:
        source = np.asarray(intensity, dtype=np.float64)
        target_n = self.output_n
        if source.shape == (target_n, target_n):
            return source, "native_samples"

        if source.shape[0] != source.shape[1]:
            raise ValueError("Virtual resolution layer expects a square model grid.")
        source_n = int(source.shape[0])
        if source_n < target_n:
            # Do not manufacture apparent resolution. This path is defensive;
            # the normal mode selection chooses a propagation grid >= output grid.
            raise ValueError(
                f"Cannot create a {target_n}x{target_n} quantitative frame from a "
                f"{source_n}x{source_n} model grid without inventing information."
            )
        if source_n % target_n != 0:
            raise ValueError("Model/output grids must have an integer integration ratio.")
        factor = source_n // target_n
        integrated = source.reshape(target_n, factor, target_n, factor).mean(axis=(1, 3))
        return np.asarray(integrated, dtype=np.float64), f"{factor}x{factor}_area_mean_pixel_integration"

    def acquire_intensity(
        self,
        state,
        *,
        z_mm: float,
        exposure_us: float = 1000.0,
        gain: float = 0.0,
        full_scale: float = 4095.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        model_intensity, metadata = self._model_intensity(state, float(z_mm))
        detector_intensity, sampling_method = self._sensor_integrate(model_intensity)

        camera, radiometry = linear_camera_signal(
            detector_intensity, exposure_us=exposure_us, gain=gain,
        )
        metadata.update(radiometry)

        if self.realistic_camera:
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
                "output_resolution_mode": self._output_resolution.value,
                "propagation_grid_yx": [self.grid_n, self.grid_n],
                "output_frame_yx": [self.output_n, self.output_n],
                "sensor_sampling_method": sampling_method,
                "beamage_resolution_reference_yx": [self.beamage_n, self.beamage_n],
                "physical_sensor_sampling_status": (
                    "PIXEL_COUNT_EQUIVALENT_ONLY; real Beamage FOV/pixel pitch remain uncalibrated"
                ),
            }
        )
        with self._lock:
            self._frame_counter += 1
        return camera, metadata
