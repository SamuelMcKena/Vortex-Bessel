import numpy as np

from vbb_study.digital_twin.physical_error_diagnostics import (
    diagnostic_context,
    parameter_catalog,
    system_error_config_for_parameter,
)


def test_catalog_exposes_core_physical_error_planes():
    catalog = parameter_catalog()
    for name in (
        "axicon_decentre_x_um",
        "input_pointing_x_mrad",
        "slm1_registration_x_um",
        "fourf_iris_offset_x_um",
    ):
        assert name in catalog
        assert catalog[name].recommended_screen


def test_axicon_decentre_is_mapped_in_engineering_units():
    cfg = system_error_config_for_parameter("axicon_decentre_x_um", 300.0)
    assert np.isclose(cfg.axicon.decentre_m[0], 300e-6)
    assert cfg.axicon.decentre_m[1] == 0.0


def test_pointing_and_iris_offsets_are_applied_at_their_physical_planes():
    pointing = system_error_config_for_parameter("input_pointing_y_mrad", -0.75)
    iris = system_error_config_for_parameter("fourf_iris_offset_x_um", 150.0)
    assert np.isclose(pointing.beam.pointing_rad[1], -0.75e-3)
    assert np.isclose(iris.fourf.iris_offset_m[0], 150e-6)


def test_diagnostic_context_reports_nominal_hardware_scales():
    context = diagnostic_context()
    assert context["fourier_iris_radius_um"] > 0
    assert context["slm_pixel_pitch_um"] > 0
