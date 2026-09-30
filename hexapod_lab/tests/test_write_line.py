"""The combined 'move while write' block (LabVIEW LINE Move_While Write)."""

import json

import pytest

from hexapod_lab.recipe import (
    MOTION_KINDS,
    POCKELS_KINDS,
    Recipe,
    RecipeStep,
    StepKind,
    preflight_recipe,
)
from hexapod_lab.sweeps import RasterSweepSpec, build_raster_sweep
from hexapod_lab.types import Pose6D


def test_write_line_counts_as_both_motion_and_beam():
    assert StepKind.WRITE_LINE in MOTION_KINDS
    assert StepKind.WRITE_LINE in POCKELS_KINDS


def test_write_line_needs_a_length_and_a_velocity():
    with pytest.raises(ValueError, match="non-zero"):
        RecipeStep.write_line(0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="velocity"):
        RecipeStep.write_line(-7.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        RecipeStep.write_line(-7.0, 0.0, 0.0, float("nan"))


def test_a_write_block_does_not_leave_the_beam_open():
    recipe = Recipe(
        steps=[
            RecipeStep.move_absolute(Pose6D(x=3.0)),
            RecipeStep.write_line(-6.0, 0.0, 0.0, 1.0),
            RecipeStep.move_line_velocity(6.0, 0.02, 0.0, 2.0),
        ]
    )
    assert not [
        issue for issue in preflight_recipe(recipe) if issue.severity == "error"
    ]


def test_write_block_warns_when_the_beam_is_already_open():
    recipe = Recipe(
        steps=[
            RecipeStep.pockels_cell(True),
            RecipeStep.write_line(-6.0, 0.0, 0.0, 1.0),
        ]
    )
    issues = preflight_recipe(recipe)
    assert any(
        issue.severity == "warning" and "already OPEN" in issue.message
        for issue in issues
    )
    # It closes the cell itself, so the recipe does not end OPEN.
    assert not [issue for issue in issues if issue.severity == "error"]


def test_loaded_write_block_with_a_bad_payload_is_reported():
    recipe = Recipe(
        steps=[
            RecipeStep.from_dict(
                {
                    "kind": "write_line",
                    "payload": {
                        "delta_xyz_mm": [1.0, 0.0],
                        "velocity_mm_s": 1.0,
                    },
                }
            )
        ]
    )
    assert any(
        issue.severity == "error" and "dX, dY and dZ" in issue.message
        for issue in preflight_recipe(recipe)
    )


def test_write_block_survives_a_save_load_round_trip(tmp_path):
    recipe = Recipe(
        name="Write test",
        steps=[RecipeStep.write_line(-7.0, 0.0, 0.0, 0.8)],
    )
    path = tmp_path / "recipe.json"
    recipe.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 3

    loaded = Recipe.load(path)
    assert loaded.steps[0].kind == StepKind.WRITE_LINE
    assert loaded.steps[0].payload["velocity_mm_s"] == pytest.approx(0.8)
    assert "beam OPEN during the move only" in loaded.steps[0].describe()


def test_version_2_recipes_still_load(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "name": "Legacy",
                "steps": [
                    {"kind": "laser_gate", "payload": {"enabled": True}},
                    {"kind": "pockels_cell", "payload": {"open": False}},
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = Recipe.load(path)
    assert [step.kind for step in loaded.steps] == [
        StepKind.POCKELS_CELL,
        StepKind.POCKELS_CELL,
    ]


def test_sweep_can_emit_one_write_block_per_line():
    recipe, summary = build_raster_sweep(
        RasterSweepSpec(
            write_dx_mm=-7.0,
            velocity_start_mm_s=1.0,
            velocity_stop_mm_s=1.2,
            velocity_step_mm_s=0.1,
            return_velocity_mm_s=2.0,
            use_write_blocks=True,
        )
    )
    kinds = [step.kind for step in recipe.steps]
    assert kinds.count(StepKind.WRITE_LINE) == 3
    assert StepKind.POCKELS_CELL not in kinds
    # Two modules per line instead of four.
    assert summary.total_steps == 6
    assert not [
        issue for issue in preflight_recipe(recipe) if issue.severity == "error"
    ]


def test_sweep_still_defaults_to_explicit_pockels_blocks():
    recipe, _ = build_raster_sweep(
        RasterSweepSpec(
            velocity_start_mm_s=1.0,
            velocity_stop_mm_s=1.0,
            velocity_step_mm_s=0.1,
        )
    )
    kinds = [step.kind for step in recipe.steps]
    assert StepKind.POCKELS_CELL in kinds
    assert StepKind.WRITE_LINE not in kinds
