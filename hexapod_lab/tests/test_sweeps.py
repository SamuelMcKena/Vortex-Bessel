from hexapod_lab.recipe import StepKind, preflight_recipe
from hexapod_lab.sweeps import RasterSweepSpec, build_raster_sweep, numeric_range


def test_numeric_range_inclusive_both_directions():
    assert numeric_range(0.2, 0.4, 0.1) == [0.2, 0.3, 0.4]
    assert numeric_range(2.0, 1.8, -0.1) == [2.0, 1.9, 1.8]


def test_legacy_style_raster_has_closed_return_moves():
    recipe, summary = build_raster_sweep(
        RasterSweepSpec(
            write_dx_mm=-7.0,
            row_pitch_mm=0.02,
            velocity_start_mm_s=1.0,
            velocity_stop_mm_s=1.2,
            velocity_step_mm_s=0.1,
            include_attenuator_steps=False,
            return_velocity_mm_s=10.0,
        )
    )
    assert summary.total_write_lines == 3

    kinds = [step.kind for step in recipe.steps]
    assert StepKind.MOVE_LINE_VELOCITY in kinds

    # Every writing line is immediately preceded by OPEN and followed by CLOSED.
    for i, step in enumerate(recipe.steps):
        if step.kind != StepKind.MOVE_LINE_VELOCITY:
            continue
        dx = float(step.payload["delta_xyz_mm"][0])
        if dx < 0:
            assert recipe.steps[i - 1].kind == StepKind.POCKELS_CELL
            assert recipe.steps[i - 1].payload["open"] is True
            assert recipe.steps[i + 1].kind == StepKind.POCKELS_CELL
            assert recipe.steps[i + 1].payload["open"] is False

    errors = [
        issue
        for issue in preflight_recipe(recipe)
        if issue.severity == "error"
    ]
    assert not errors


def test_attenuator_sweep_generates_series():
    recipe, summary = build_raster_sweep(
        RasterSweepSpec(
            velocity_start_mm_s=1.0,
            velocity_stop_mm_s=1.0,
            velocity_step_mm_s=0.1,
            attenuation_start_percent=25.0,
            attenuation_stop_percent=35.0,
            attenuation_step_percent=5.0,
            include_attenuator_steps=True,
        )
    )
    assert summary.series_count == 3
    assert summary.total_write_lines == 3
    assert sum(
        step.kind == StepKind.ATTENUATOR_SET
        for step in recipe.steps
    ) == 3
