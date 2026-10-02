from pathlib import Path

import numpy as np

from hexapod_lab.kinematics import RigKinematics, RigProfile, hxp_bryant_rotation
from hexapod_lab.types import Pose6D


PROFILE = Path(__file__).resolve().parents[1] / "assets" / "cad_profile.json"


def test_home_geometry_has_six_equal_struts():
    rig = RigKinematics(RigProfile.from_json(PROFILE))
    lengths = rig.leg_lengths_mm(Pose6D())
    assert lengths.shape == (6,)
    assert np.allclose(lengths, lengths[0], atol=1e-6)
    assert np.isclose(lengths[0], 384.3155346, atol=1e-4)


def test_hxp_z_maps_to_cad_vertical_y():
    rig = RigKinematics(RigProfile.from_json(PROFILE))
    p0 = rig.top_joint_positions(Pose6D())
    p1 = rig.top_joint_positions(Pose6D(z=2.5))
    delta = p1 - p0
    assert np.allclose(delta[:, 0], 0.0, atol=1e-9)
    assert np.allclose(delta[:, 1], 2.5, atol=1e-9)
    assert np.allclose(delta[:, 2], 0.0, atol=1e-9)


def test_bryant_rotation_is_orthonormal():
    r = hxp_bryant_rotation(Pose6D(u=3.0, v=-4.0, w=8.0))
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-12)


def test_leg_transforms_preserve_bottom_body_anchor_and_move_top_rod_anchor():
    profile = RigProfile.from_json(PROFILE)
    rig = RigKinematics(profile)
    pose = Pose6D(x=2.0, y=-1.0, z=3.0, u=1.0, v=2.0, w=-1.0)
    transforms = rig.leg_axis_transforms(pose)
    top_positions = rig.top_joint_positions(pose)
    for leg, tf, top_new in zip(profile.legs, transforms, top_positions):
        b = np.array([*leg.bottom_joint_mm, 1.0])
        t = np.array([*leg.top_joint_mm, 1.0])
        assert np.allclose((tf["body"] @ b)[:3], leg.bottom_joint_mm, atol=1e-8)
        assert np.allclose((tf["rod"] @ t)[:3], top_new, atol=1e-8)
