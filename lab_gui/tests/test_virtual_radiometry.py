import numpy as np
import pytest

from labcontrol.virtual_lab import VirtualBenchEngine
from labcontrol.virtual_lab_resolution import ResolutionAwareVirtualBenchEngine
from labcontrol.virtual_radiometry import fixed_power_amplitude, camera_response_scale
from labcontrol.ui.image_view import render_preview


@pytest.mark.parametrize("engine_type", [VirtualBenchEngine, ResolutionAwareVirtualBenchEngine])
def test_camera_retains_axial_intensity_and_exposure_ratios(engine_type):
    engine = engine_type() if engine_type is VirtualBenchEngine else engine_type(beamage_n=16, maximum_grid_n=32)
    engine.realistic_camera = False
    engine._model_intensity = lambda state, z: (np.full((16, 16), 10.0 / z), {})
    near, meta = engine.acquire_intensity(None, z_mm=1)
    far, _ = engine.acquire_intensity(None, z_mm=10)
    longer, _ = engine.acquire_intensity(None, z_mm=10, exposure_us=2000)
    gain, _ = engine.acquire_intensity(None, z_mm=10, gain=100)
    dark, _ = engine.acquire_intensity(None, z_mm=1, exposure_us=0)
    assert np.allclose(far, near / 10)
    assert np.allclose(longer, far * 2)
    assert np.allclose(gain, far * 2)
    assert not dark.any()
    assert meta["per_frame_peak_normalisation"] is False
    # Changing sensor clipping range must not implicitly change sensitivity.
    higher_range, _ = engine.acquire_intensity(None, z_mm=1, full_scale=65535)
    assert np.array_equal(near, higher_range)


def test_saturation_and_noise_are_still_present():
    engine = ResolutionAwareVirtualBenchEngine(beamage_n=16, maximum_grid_n=32)
    engine.realistic_camera = False
    engine._model_intensity = lambda state, z: (np.full((16, 16), 10000.0), {})
    saturated, _ = engine.acquire_intensity(None, z_mm=1)
    assert np.all(saturated == 4095)
    engine._model_intensity = lambda state, z: (np.full((16, 16), 1.0), {})
    engine.realistic_camera = True
    first, _ = engine.acquire_intensity(None, z_mm=1)
    second, _ = engine.acquire_intensity(None, z_mm=1)
    assert not np.array_equal(first, second)


def test_gaussian_resizing_preserves_incident_power_before_clipping():
    powers = []
    x = np.linspace(-0.02, 0.02, 2001)
    for wx, wy in [(0.001, 0.001), (0.002, 0.002), (0.003, 0.002)]:
        amp = fixed_power_amplitude(wx, wy)
        power = amp**2 * np.trapezoid(np.exp(-2*x*x/wx**2), x) * np.trapezoid(np.exp(-2*x*x/wy**2), x)
        powers.append(power)
    assert np.allclose(powers, np.pi * 0.002**2 / 2, rtol=1e-8, atol=0)


@pytest.mark.parametrize("exposure,gain", [(float('nan'), 0), (-1, 0), (1, float('inf')), (1, -1)])
def test_invalid_camera_settings_are_rejected(exposure, gain):
    with pytest.raises(ValueError):
        camera_response_scale(exposure, gain)


@pytest.mark.parametrize("mode", ["sensor range", "sensor log"])
def test_fixed_display_does_not_equalise_dim_and_bright_frames(mode):
    bright = render_preview(np.full((16, 16), 1000), colour="grayscale", scale=mode, full_scale=4095)
    dim = render_preview(np.full((16, 16), 100), colour="grayscale", scale=mode, full_scale=4095)
    assert bright.mean() > dim.mean() > 0
