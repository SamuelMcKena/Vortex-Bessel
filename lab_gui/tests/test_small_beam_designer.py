import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from labcontrol.small_beam_design import design_small_beam, ideal_radial_reference
from labcontrol.ui.small_beam_designer import SmallBeamDesigner


def test_legacy_target_and_actual_ring_diameter_are_distinct():
    d = design_small_beam(charge=3, medium_index=2.44, diameter_definition="equivalent_q0")
    assert d['equivalent_l0_core_diameter_m'] == pytest.approx(3e-6)
    assert d['predicted_bessel_length_m'] == pytest.approx(150e-6)
    assert d['vortex_main_ring_diameter_m'] > 3e-6
    assert d['objective_map_demag'] < 1
    actual = design_small_beam(charge=3, medium_index=2.44, diameter_definition="actual")
    assert actual['vortex_main_ring_diameter_m'] == pytest.approx(3e-6)


def test_unphysical_small_q20_target_is_rejected():
    with pytest.raises(ValueError, match='no propagating'):
        design_small_beam(charge=20, diameter_um=3, medium_index=1)


def test_q0_has_central_spot_and_vortex_has_dark_centre():
    for q in [0, 3]:
        d = design_small_beam(charge=q, diameter_um=10)
        r, intensity = ideal_radial_reference(d)
        assert np.isfinite(intensity).all()
        assert intensity[0] == (1 if q == 0 else 0)
        assert r[0] == 0


def test_dialog_recalculates_and_invalidates_old_result():
    app = QApplication.instance() or QApplication([])
    dialog = SmallBeamDesigner()
    try:
        assert dialog.result['ell'] == 0
        dialog.load_legacy()
        assert dialog.result['ell'] == 3
        assert dialog.result['medium_index'] == 2.44
        dialog.diameter.setValue(5)
        assert dialog.result is None
        assert not dialog.export.isEnabled()
        dialog.calculate()
        assert dialog.result['requested_diameter_um'] == 5
    finally:
        dialog.close()
        app.processEvents()
