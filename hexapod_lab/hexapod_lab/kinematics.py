from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math

import numpy as np

from .types import Pose6D


@dataclass(frozen=True, slots=True)
class LegDefinition:
    name: str
    bottom_joint_mm: tuple[float, float, float]
    top_joint_mm: tuple[float, float, float]
    solid_indices: tuple[int, int, int, int, int, int]
    # order: body, rod, top_ball, top_mount, bottom_ball, bottom_mount


@dataclass(frozen=True, slots=True)
class RigProfile:
    source_name: str
    units: str
    top_frame_origin_mm: tuple[float, float, float]
    base_frame_origin_mm: tuple[float, float, float]
    top_rigid_solid_indices: tuple[int, ...]
    base_rigid_solid_indices: tuple[int, ...]
    legs: tuple[LegDefinition, ...]

    @classmethod
    def from_json(cls, path: str | Path) -> "RigProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        legs = tuple(
            LegDefinition(
                name=item["name"],
                bottom_joint_mm=tuple(item["bottom_joint_mm"]),
                top_joint_mm=tuple(item["top_joint_mm"]),
                solid_indices=tuple(item["solid_indices"]),
            )
            for item in data["legs"]
        )
        return cls(
            source_name=data["source_name"],
            units=data.get("units", "mm"),
            top_frame_origin_mm=tuple(data["top_frame_origin_mm"]),
            base_frame_origin_mm=tuple(data["base_frame_origin_mm"]),
            top_rigid_solid_indices=tuple(data["top_rigid_solid_indices"]),
            base_rigid_solid_indices=tuple(data["base_rigid_solid_indices"]),
            legs=legs,
        )


# CAD frame from the supplied SolidWorks STEP:
#   CAD X = HXP X
#   CAD Y = HXP Z (vertical)
#   CAD Z = HXP Y
# This is intentionally explicit and can later be replaced by a measured registration.
_HXP_TO_CAD = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)


def _axis_rotation(axis: str, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(axis)


def hxp_bryant_rotation(pose: Pose6D) -> np.ndarray:
    """Return the HXP Tool-in-Work rotation matrix.

    Newport's HXP manual defines Bryant/Tait-Bryan ZYX rotations as clockwise
    W (about Z), then clockwise V (about the once-rotated Y), then clockwise U
    (about the twice-rotated X). With conventional right-handed active rotation
    matrices, clockwise is represented by negative angles.
    """

    u, v, w = map(math.radians, (pose.u, pose.v, pose.w))
    return _axis_rotation("z", -w) @ _axis_rotation("y", -v) @ _axis_rotation("x", -u)


def hxp_pose_to_cad_transform(pose: Pose6D, pivot_cad_mm: Iterable[float]) -> np.ndarray:
    """Return a 4x4 transform that applies an HXP relative pose to the CAD carriage.

    The rotation is about ``pivot_cad_mm`` and the HXP XYZ translation is mapped
    into the CAD frame using the explicit axis-registration matrix above.
    """

    pivot = np.asarray(tuple(pivot_cad_mm), dtype=float)
    if pivot.shape != (3,):
        raise ValueError("pivot_cad_mm must contain exactly 3 coordinates")

    r_hxp = hxp_bryant_rotation(pose)
    r_cad = _HXP_TO_CAD @ r_hxp @ _HXP_TO_CAD.T
    t_hxp = np.array([pose.x, pose.y, pose.z], dtype=float)
    t_cad = _HXP_TO_CAD @ t_hxp

    out = np.eye(4, dtype=float)
    out[:3, :3] = r_cad
    out[:3, 3] = pivot + t_cad - r_cad @ pivot
    return out


def transform_point(matrix4: np.ndarray, point3: Iterable[float]) -> np.ndarray:
    p = np.ones(4, dtype=float)
    p[:3] = tuple(point3)
    return (np.asarray(matrix4, dtype=float) @ p)[:3]


def rotation_from_to(source: Iterable[float], target: Iterable[float]) -> np.ndarray:
    """Shortest 3D rotation mapping source direction to target direction."""

    a = np.asarray(tuple(source), dtype=float)
    b = np.asarray(tuple(target), dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return np.eye(3)
    a /= na
    b /= nb
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    norm_cross = float(np.linalg.norm(cross))

    if norm_cross <= 1e-12:
        if dot > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.8:
            trial = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, trial)
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    axis = cross / norm_cross
    angle = math.atan2(norm_cross, dot)
    kx, ky, kz = axis
    k = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=float)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def anchored_transform(anchor_home: Iterable[float], anchor_new: Iterable[float], rotation3: np.ndarray) -> np.ndarray:
    """Rigid transform rotating around ``anchor_home`` and relocating it to ``anchor_new``."""

    a0 = np.asarray(tuple(anchor_home), dtype=float)
    a1 = np.asarray(tuple(anchor_new), dtype=float)
    r = np.asarray(rotation3, dtype=float)
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = a1 - r @ a0
    return out


@dataclass(slots=True)
class RigKinematics:
    profile: RigProfile

    def top_transform(self, pose: Pose6D) -> np.ndarray:
        return hxp_pose_to_cad_transform(pose, self.profile.top_frame_origin_mm)

    def top_joint_positions(self, pose: Pose6D) -> np.ndarray:
        m = self.top_transform(pose)
        return np.vstack([transform_point(m, leg.top_joint_mm) for leg in self.profile.legs])

    def bottom_joint_positions(self) -> np.ndarray:
        return np.asarray([leg.bottom_joint_mm for leg in self.profile.legs], dtype=float)

    def leg_lengths_mm(self, pose: Pose6D) -> np.ndarray:
        top = self.top_joint_positions(pose)
        bottom = self.bottom_joint_positions()
        return np.linalg.norm(top - bottom, axis=1)

    def leg_axis_transforms(self, pose: Pose6D) -> list[dict[str, np.ndarray]]:
        """Transforms for actual CAD arm sub-parts.

        For each leg, the body and bottom ball are rotated about the fixed bottom
        spherical-joint centre. The rod and top ball are rotated along the same
        new axis but are anchored to the moving top joint, which produces the
        telescoping visual effect. The top mount itself follows the carriage's
        full Tool transform; the bottom mount stays fixed.
        """

        top_tf = self.top_transform(pose)
        result: list[dict[str, np.ndarray]] = []
        for leg in self.profile.legs:
            b0 = np.asarray(leg.bottom_joint_mm, dtype=float)
            t0 = np.asarray(leg.top_joint_mm, dtype=float)
            t1 = transform_point(top_tf, t0)
            r = rotation_from_to(t0 - b0, t1 - b0)
            result.append(
                {
                    "body": anchored_transform(b0, b0, r),
                    "rod": anchored_transform(t0, t1, r),
                    "top_ball": anchored_transform(t0, t1, r),
                    "top_mount": top_tf,
                    "bottom_ball": anchored_transform(b0, b0, r),
                    "bottom_mount": np.eye(4),
                }
            )
        return result

    def sample_transform(self, pose: Pose6D) -> np.ndarray:
        return self.top_transform(pose)

    def beam_intersection_with_top_plane(self, pose: Pose6D, beam_origin_cad_mm: Iterable[float], beam_direction_cad: Iterable[float]) -> np.ndarray | None:
        """Intersect a fixed lab beam ray with the moving carriage plane.

        The carriage plane is approximated by the home top-plate plane transformed
        by the HXP pose. The supplied CAD has CAD-Y as its vertical direction.
        """

        m = self.top_transform(pose)
        p0 = transform_point(m, self.profile.top_frame_origin_mm)
        n0 = np.array([0.0, 1.0, 0.0])
        n = m[:3, :3] @ n0
        o = np.asarray(tuple(beam_origin_cad_mm), dtype=float)
        d = np.asarray(tuple(beam_direction_cad), dtype=float)
        d_norm = np.linalg.norm(d)
        if d_norm <= 1e-12:
            raise ValueError("beam direction must be non-zero")
        d /= d_norm
        denom = float(np.dot(n, d))
        if abs(denom) < 1e-10:
            return None
        ray_t = float(np.dot(n, p0 - o) / denom)
        if ray_t < 0:
            return None
        return o + ray_t * d
