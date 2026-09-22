from hexapod_lab.recipe import Recipe, RecipeStep, preflight_recipe
from hexapod_lab.types import Pose6D


def test_preflight_requires_gate_off_at_end():
    r = Recipe(steps=[RecipeStep.laser_gate(True), RecipeStep.move_absolute(Pose6D(x=1))])
    issues = preflight_recipe(r)
    assert any(i.severity == "error" and "ends with the laser gate ON" in i.message for i in issues)


def test_preflight_clean_recipe():
    r = Recipe(steps=[
        RecipeStep.move_absolute(Pose6D(x=1)),
        RecipeStep.laser_gate(True),
        RecipeStep.move_incremental(Pose6D(x=2)),
        RecipeStep.laser_gate(False),
    ])
    issues = preflight_recipe(r)
    assert not [i for i in issues if i.severity == "error"]
