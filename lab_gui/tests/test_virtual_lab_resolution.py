from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QApplication

from labcontrol.ui.virtual_advanced_v3 import VirtualLabResolutionWindow
from labcontrol.virtual_lab_resolution import (
    ResolutionAwareVirtualBenchEngine,
    VirtualOutputResolution,
)


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_resolution_modes_use_expected_model_and_output_shapes_without_upsampling() -> None:
    # Small test dimensions exercise exactly the same logic without allocating
    # the production 2048/4096 arrays in CI.
    engine = ResolutionAwareVirtualBenchEngine(beamage_n=64, maximum_grid_n=128)

    engine.set_output_resolution(VirtualOutputResolution.BEAMAGE_4M)
    assert engine.grid_n == 64
    assert engine.shape_yx == (64, 64)

    engine.set_output_resolution(VirtualOutputResolution.MAXIMUM)
    assert engine.grid_n == 128
    assert engine.shape_yx == (128, 128)

    engine.set_output_resolution(VirtualOutputResolution.MAXIMUM_TO_BEAMAGE)
    assert engine.grid_n == 128
    assert engine.shape_yx == (64, 64)


def test_maximum_to_beamage_area_integrates_instead_of_interpolating() -> None:
    engine = ResolutionAwareVirtualBenchEngine(beamage_n=16, maximum_grid_n=32)
    engine.set_output_resolution(VirtualOutputResolution.MAXIMUM_TO_BEAMAGE)
    source = np.arange(32 * 32, dtype=float).reshape(32, 32)
    reduced, method = engine._sensor_integrate(source)
    expected = source.reshape(16, 2, 16, 2).mean(axis=(1, 3))
    assert reduced.shape == (16, 16)
    assert np.allclose(reduced, expected)
    assert method == "2x2_area_mean_pixel_integration"


def test_native_mode_retains_current_model_grid() -> None:
    engine = ResolutionAwareVirtualBenchEngine(beamage_n=64, maximum_grid_n=128)
    engine.set_output_resolution(VirtualOutputResolution.NATIVE)
    assert engine.grid_n == engine.geometry.preview_grid_n
    assert engine.shape_yx == (engine.geometry.preview_grid_n, engine.geometry.preview_grid_n)


def test_resolution_gui_exposes_beamage_maximum_and_controlled_comparison_modes() -> None:
    qt = app()
    window = VirtualLabResolutionWindow()
    try:
        values = {
            window.virtual_output_resolution.itemText(index)
            for index in range(window.virtual_output_resolution.count())
        }
        assert VirtualOutputResolution.BEAMAGE_4M.value in values
        assert VirtualOutputResolution.MAXIMUM.value in values
        assert VirtualOutputResolution.MAXIMUM_TO_BEAMAGE.value in values
        assert window.resolution_engine.beamage_n == 2048
        assert window.resolution_engine.maximum_grid_n == 4096
        assert "PIXEL_COUNT_EQUIVALENT" in window.resolution_engine.resolution_summary()[
            "physical_sampling_status"
        ]
    finally:
        window.close()
        qt.processEvents()
