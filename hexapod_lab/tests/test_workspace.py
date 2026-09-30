from pathlib import Path

import pytest

from hexapod_lab.kinematics import RigKinematics, RigProfile
from hexapod_lab.providers import VirtualHexapodProvider
from hexapod_lab.recipe import Recipe, RecipeStep
from hexapod_lab.types import Pose6D
from hexapod_lab.workspace import WorkspaceLimits, validate_recipe_workspace


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def rig_and_limits():
    kinematics = RigKinematics(
        RigProfile.from_json(ROOT / "assets" / "cad_profile.json")
    )
    return kinematics, WorkspaceLimits.cad_conservative(kinematics)


def test_default_envelope_accepts_home_and_legacy_line(rig_and_limits):
    kinematics, limits = rig_and_limits
    assert limits.violations(kinematics, Pose6D()) == []
    assert limits.path_violations(
        kinematics,
        Pose6D(),
        Pose6D(x=-7.0),
    ) == []


def test_screenshot_pose_is_rejected(rig_and_limits):
    kinematics, limits = rig_and_limits
    issues = limits.violations(
        kinematics,
        Pose6D(x=7.0, z=14.0, u=20.0, v=6.0, w=9.0),
    )
    assert any(issue.code == "axis_u" for issue in issues)


def test_recipe_accumulation_and_velocity_are_checked(rig_and_limits):
    kinematics, limits = rig_and_limits
    recipe = Recipe(
        steps=[
            RecipeStep.move_incremental(Pose6D(x=-7.0))
            for _ in range(4)
        ]
        + [RecipeStep.move_line_velocity(1.0, 0.0, 0.0, 2.2)]
    )
    issues = validate_recipe_workspace(
        recipe,
        Pose6D(),
        limits,
        kinematics,
    )
    assert any(issue.step_index == 3 and "X=" in issue.message for issue in issues)
    assert any(issue.step_index == 4 and "velocity" in issue.message for issue in issues)


def test_workspace_configuration_round_trip(rig_and_limits):
    kinematics, limits = rig_and_limits
    restored = WorkspaceLimits.from_dict(limits.to_dict(), kinematics)
    assert restored == limits


def test_virtual_provider_rejects_motion_before_planning(rig_and_limits):
    kinematics, limits = rig_and_limits
    stage = VirtualHexapodProvider(
        motion_validator=lambda start, target: limits.require_path(
            kinematics,
            start,
            target,
        )
    )
    stage.connect()
    with pytest.raises(ValueError, match="workspace limits"):
        stage.move_absolute(Pose6D(x=30.0))
    assert not stage.is_busy()
