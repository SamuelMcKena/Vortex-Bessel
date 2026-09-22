from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from .cad_loader import build_step_mesh_cache
from .kinematics import RigKinematics, RigProfile, transform_point
from .types import Pose6D

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError as exc:  # pragma: no cover - dependency checked at GUI launch
    raise RuntimeError("3D viewer requires pyvista and pyvistaqt; install requirements.txt") from exc


class Hexapod3DViewer(QtInteractor):
    """Live 3D digital-twin view with optional exact STEP meshes."""

    def __init__(self, parent, profile: RigProfile) -> None:
        super().__init__(parent)
        self.profile = profile
        self.kinematics = RigKinematics(profile)
        self._cad_actors: dict[int, object] = {}
        self._exact_cad = False
        self._pose = Pose6D()
        self._laser_on = False
        self._travel_points: list[np.ndarray] = []
        self._process_points: list[np.ndarray] = []
        self._last_travel_point: np.ndarray | None = None
        self._last_process_point: np.ndarray | None = None
        self._status_cb = None

        self.set_background("#10151c")
        self.add_axes(line_width=2)
        self.enable_anti_aliasing("fxaa")
        self._build_fallback_scene()
        self.reset_camera()
        self.camera_position = [(850, 650, 850), (0, 220, 0), (0, 1, 0)]

    def set_status_callback(self, callback) -> None:
        self._status_cb = callback

    def _status(self, text: str) -> None:
        if self._status_cb is not None:
            self._status_cb(text)

    @staticmethod
    def _set_actor_matrix(actor, matrix: np.ndarray) -> None:
        actor.user_matrix = np.asarray(matrix, dtype=float)

    def _make_line_poly(self, points: np.ndarray) -> pv.PolyData:
        mesh = pv.PolyData(points)
        if len(points) >= 2:
            mesh.lines = np.hstack(([len(points)], np.arange(len(points), dtype=np.int64)))
        return mesh

    def _build_fallback_scene(self) -> None:
        base_center = np.asarray(self.profile.base_frame_origin_mm, dtype=float)
        base = pv.Box(bounds=(-250, 250, base_center[1] - 12.5, base_center[1] + 12.5, -250, 250))
        self._base_actor = self.add_mesh(base, color="#4f5965", smooth_shading=True, name="base")

        top_center = np.asarray(self.profile.top_frame_origin_mm, dtype=float)
        top = pv.Box(bounds=(-250, 250, top_center[1] - 12.5, top_center[1] + 12.5, -250, 250))
        self._top_actor = self.add_mesh(top, color="#6f7b86", smooth_shading=True, name="top")

        bottom = self.kinematics.bottom_joint_positions()
        top_joints = self.kinematics.top_joint_positions(Pose6D())
        pts = np.empty((12, 3), dtype=float)
        lines = []
        for i in range(6):
            pts[2 * i] = bottom[i]
            pts[2 * i + 1] = top_joints[i]
            lines.extend([2, 2 * i, 2 * i + 1])
        self._leg_mesh = pv.PolyData(pts)
        self._leg_mesh.lines = np.asarray(lines, dtype=np.int64)
        self._leg_actor = self.add_mesh(self._leg_mesh,color="#aab5bf",line_width=12,render_lines_as_tubes=True,name="legs")

        self._sample_home_center = top_center + np.array([0.0, 28.0, 0.0])
        sample = pv.Box(bounds=(-25, 25, top_center[1] + 14.0, top_center[1] + 42.0, -25, 25))
        self._sample_actor = self.add_mesh(sample, color="#c7d1d9", opacity=0.45, name="sample")

        beam_origin = top_center + np.array([0.0, 360.0, 0.0])
        beam_end = np.asarray(self.profile.base_frame_origin_mm, dtype=float) - np.array([0.0, 80.0, 0.0])
        self._beam_origin = beam_origin
        self._beam_direction = np.array([0.0, -1.0, 0.0])
        self.add_mesh(pv.Line(beam_origin, beam_end), color="#d75f5f", line_width=3, name="laser-axis")
        self._beam_hit_mesh = pv.Sphere(radius=5.0, center=top_center)
        self._beam_hit_actor = self.add_mesh(self._beam_hit_mesh, color="#f0d6d6", name="beam-hit")

        self._travel_mesh = self._make_line_poly(np.empty((0, 3)))
        self._travel_actor = self.add_mesh(self._travel_mesh, color="#7f8a96", line_width=3, name="travel-trace")
        self._process_mesh = self._make_line_poly(np.empty((0, 3)))
        self._process_actor = self.add_mesh(self._process_mesh, color="#e5b15c", line_width=5, name="process-trace")

        self.add_text("CAD-derived Stewart rig • fixed lab laser axis", position="upper_left", font_size=10, name="mode-label")

    def clear_traces(self) -> None:
        self._travel_points.clear(); self._process_points.clear()
        self._last_travel_point = None; self._last_process_point = None
        self._sync_trace_mesh(self._travel_mesh, self._travel_points, self._travel_actor)
        self._sync_trace_mesh(self._process_mesh, self._process_points, self._process_actor)
        self.render()

    @staticmethod
    def _sync_trace_mesh(mesh: pv.PolyData, points: list[np.ndarray]) -> None:
        if not points:
            mesh.points = np.empty((0, 3)); mesh.lines = np.empty(0, dtype=np.int64); return
        arr = np.asarray(points, dtype=float); mesh.points = arr
        if len(arr) >= 2:
            mesh.lines = np.hstack(([len(arr)], np.arange(len(arr), dtype=np.int64)))
        else:
            mesh.lines = np.empty(0, dtype=np.int64)

    def _append_trace(self, points: list[np.ndarray], p: np.ndarray, *, threshold_mm: float, process: bool) -> None:
        last = self._last_process_point if process else self._last_travel_point
        if last is None or float(np.linalg.norm(p - last)) >= threshold_mm:
            points.append(p.copy())
            if process: self._last_process_point = p.copy()
            else: self._last_travel_point = p.copy()

    def update_state(self, pose: Pose6D, laser_on: bool) -> None:
        self._pose = pose; self._laser_on = bool(laser_on)
        top_tf = self.kinematics.top_transform(pose)
        top_joints = self.kinematics.top_joint_positions(pose)
        bottom = self.kinematics.bottom_joint_positions()
        self._set_actor_matrix(self._top_actor, top_tf)
        self._set_actor_matrix(self._sample_actor, top_tf)
        points = self._leg_mesh.points.copy()
        for i in range(6):
            points[2 * i] = bottom[i]; points[2 * i + 1] = top_joints[i]
        self._leg_mesh.points = points

        hit = self.kinematics.beam_intersection_with_top_plane(pose, self._beam_origin, self._beam_direction)
        if hit is not None:
            self._beam_hit_mesh.points = pv.Sphere(radius=5.0, center=hit).points
            self._append_trace(self._travel_points, hit, threshold_mm=0.20, process=False)
            if laser_on: self._append_trace(self._process_points, hit, threshold_mm=0.08, process=True)
        self._sync_trace_mesh(self._travel_mesh, self._travel_points, self._travel_actor)
        self._sync_trace_mesh(self._process_mesh, self._process_points, self._process_actor)
        if self._exact_cad: self._update_exact_cad(pose)
        self.render()

    def _update_exact_cad(self, pose: Pose6D) -> None:
        top_tf = self.kinematics.top_transform(pose)
        for idx in self.profile.top_rigid_solid_indices:
            actor = self._cad_actors.get(idx)
            if actor is not None: self._set_actor_matrix(actor, top_tf)
        leg_tfs = self.kinematics.leg_axis_transforms(pose)
        keys = ("body", "rod", "top_ball", "top_mount", "bottom_ball", "bottom_mount")
        for leg, transforms in zip(self.profile.legs, leg_tfs):
            for solid_idx, key in zip(leg.solid_indices, keys):
                actor = self._cad_actors.get(solid_idx)
                if actor is not None: self._set_actor_matrix(actor, transforms[key])

    def load_step(self, step_path: str | Path) -> None:
        self._status("Tessellating STEP model… first load can take a little while")
        cache = build_step_mesh_cache(step_path)
        max_index = max([*self.profile.base_rigid_solid_indices,*self.profile.top_rigid_solid_indices]+[idx for leg in self.profile.legs for idx in leg.solid_indices])
        if len(cache.mesh_paths) <= max_index:
            raise RuntimeError(f"This rig profile expects at least {max_index + 1} STEP solids, but {len(cache.mesh_paths)} were found")
        for actor in self._cad_actors.values():
            try: self.remove_actor(actor, reset_camera=False)
            except Exception: pass
        self._cad_actors.clear()
        top_set=set(self.profile.top_rigid_solid_indices); base_set=set(self.profile.base_rigid_solid_indices)
        leg_by_index={idx for leg in self.profile.legs for idx in leg.solid_indices}
        wanted=sorted(top_set|base_set|leg_by_index)
        for idx in wanted:
            mesh=pv.read(str(cache.mesh_paths[idx]))
            if idx in top_set: color="#6d7882"
            elif idx in base_set: color="#4d5660"
            else: color="#9da8b3"
            self._cad_actors[idx]=self.add_mesh(mesh,color=color,smooth_shading=True,name=f"cad-{idx:02d}")
        self._exact_cad=True
        self._base_actor.GetProperty().SetOpacity(0.05)
        self._top_actor.GetProperty().SetOpacity(0.05)
        self._leg_actor.GetProperty().SetOpacity(0.08)
        self._update_exact_cad(self._pose)
        self.reset_camera()
        self._status(f"Loaded exact STEP assembly ({len(cache.mesh_paths)} solids); CAD articulation enabled")
