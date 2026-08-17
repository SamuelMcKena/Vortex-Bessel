from vbb_study.publication import notebook_controls as nc


def test_notebook_controls_default_stage_validates():
    controls = nc.make_notebook_controls("scalar")
    controls.validate()
    assert controls.stage == "scalar"
    assert controls.run_mode in nc.ALLOWED_RUN_MODES
    assert controls.parameters["ell"] == 3


def test_notebook_controls_with_updates_separates_fields_and_parameters():
    controls = nc.make_notebook_controls("lab_realism")
    updated = controls.with_updates(run_mode="publication", objective_NA=0.55, custom_knob=123)
    assert updated.run_mode == "publication"
    assert updated.parameters["objective_NA"] == 0.55
    assert updated.parameters["custom_knob"] == 123


def test_describe_controls_returns_dataframe():
    df = nc.describe_controls(nc.make_notebook_controls("materials"))
    assert {"control", "value"}.issubset(df.columns)
    assert "stage" in set(df["control"])
