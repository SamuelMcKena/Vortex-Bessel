import pytest

from hexapod_lab.recipe import Recipe, RecipeStep, preflight_recipe
from hexapod_lab.types import Pose6D


def test_preflight_requires_pockels_closed_at_end():
    recipe = Recipe(
        steps=[
            RecipeStep.pockels_cell(True),
            RecipeStep.move_absolute(Pose6D(x=1)),
        ]
    )
    issues = preflight_recipe(recipe)
    assert any(
        issue.severity == "error"
        and "Pockels cell OPEN" in issue.message
        for issue in issues
    )


def test_preflight_clean_recipe_with_attenuator():
    recipe = Recipe(
        steps=[
            RecipeStep.attenuator_set(25.0),
            RecipeStep.move_absolute(Pose6D(x=1)),
            RecipeStep.pockels_cell(True),
            RecipeStep.move_incremental(Pose6D(x=2)),
            RecipeStep.pockels_cell(False),
        ]
    )
    issues = preflight_recipe(recipe)
    assert not [issue for issue in issues if issue.severity == "error"]


def test_preflight_warns_if_attenuator_changes_with_beam_open():
    recipe = Recipe(
        steps=[
            RecipeStep.pockels_cell(True),
            RecipeStep.attenuator_set(50.0),
            RecipeStep.pockels_cell(False),
        ]
    )
    issues = preflight_recipe(recipe)
    assert any(
        issue.severity == "warning"
        and "attenuator is changed" in issue.message
        for issue in issues
    )


def test_legacy_laser_gate_recipe_loads_as_pockels():
    step = RecipeStep.from_dict(
        {
            "kind": "laser_gate",
            "label": "Laser gate ON",
            "payload": {"enabled": True},
        }
    )
    assert step.kind.value == "pockels_cell"
    assert step.payload["open"] is True


def test_attenuator_rejects_out_of_range():
    try:
        RecipeStep.attenuator_set(110.0)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range attenuator setpoint was accepted")


def test_line_velocity_recipe_step():
    step = RecipeStep.move_line_velocity(
        -7.0,
        0.02,
        0.0,
        1.5,
    )
    assert step.kind.value == "move_line_velocity"
    assert step.payload["delta_xyz_mm"] == [-7.0, 0.02, 0.0]
    assert step.payload["velocity_mm_s"] == 1.5
    issues = preflight_recipe(
        Recipe(
            steps=[
                step,
                RecipeStep.pockels_cell(False),
            ]
        )
    )
    assert not [issue for issue in issues if issue.severity == "error"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_recipe_factories_reject_non_finite_values(value):
    with pytest.raises(ValueError):
        RecipeStep.move_line_velocity(1.0, 0.0, 0.0, value)
    with pytest.raises(ValueError):
        RecipeStep.attenuator_set(value)
    with pytest.raises(ValueError):
        RecipeStep.wait(value)


def test_preflight_rejects_non_finite_loaded_line_velocity():
    recipe = Recipe(
        steps=[
            RecipeStep.from_dict(
                {
                    "kind": "move_line_velocity",
                    "payload": {
                        "delta_xyz_mm": [1.0, 0.0, 0.0],
                        "velocity_mm_s": float("nan"),
                    },
                }
            )
        ]
    )
    assert any(
        issue.severity == "error" and "finite" in issue.message
        for issue in preflight_recipe(recipe)
    )
