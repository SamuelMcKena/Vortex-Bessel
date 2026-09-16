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
    engine = ResolutionAwareVirtualBenchEngine(beamage_n=2, maximum_grid_n=4)
    # Constructor deliberately rejects unrealistic dimensions in production, so
    # make a normal small engine then lower the test-only dimensions explicitly.
    # This avoids allocating high-resolution fields while testing integration.
