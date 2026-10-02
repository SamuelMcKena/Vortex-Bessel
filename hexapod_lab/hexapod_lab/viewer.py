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
        # Traces are stored in SAMPLE-LOCAL CAD coordinates so they stay
        # attached to the sample when the carriage subsequently moves. They are
        # kept as alternating beam-ON/beam-OFF runs so a repositioning move is
        # never rendered as though it were written material.
        self._segments: list[tuple[bool, list[np.ndarray]]] = []
        self._beam_hit_world: np.ndarray | None = None
        self._beam_hit_sample_xy: tuple[float, float] | None = None
        self._show_travel = True
        self._show_written = True
        self._status_cb = None
        # A static scene does not need re-rendering 20 times a second; the
        # twin only redraws when the pose, the beam state or a trace changes.
        self._last_render_key: tuple | None = None
        # Completed runs never change, so their point arrays are cached and
        # only the run currently being written is re-serialised each frame.
        self._run_arrays: dict[int, tuple[int, np.ndarray]] = {}

        self.set_background("#071018", top="#111a24")
        self.add_axes(line_width=2, color="#71808e")
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

        self._sample_half_size_mm = np.array([25.0, 14.0, 25.0])
        self._sample_home_center = top_center + np.array(
            [0.0, self._sample_half_size_mm[1] + 2.0, 0.0]
        )
        sample = pv.Box(
            bounds=(
                -self._sample_half_size_mm[0],
                self._sample_half_size_mm[0],
                top_center[1] + 2.0,
                top_center[1] + 2.0 + 2.0 * self._sample_half_size_mm[1],
                -self._sample_half_size_mm[2],
                self._sample_half_size_mm[2],
            )
        )
        self._sample_actor = self.add_mesh(
            sample,
            color="#a9d7ee",
            opacity=0.30,
            smooth_shading=True,
            specular=0.55,
            specular_power=35,
            show_edges=True,
            edge_color="#8ec5df",
            name="sample",
        )
        self._sample_surface_point_home = np.array(
            [
                0.0,
                top_center[1] + 2.0 + 2.0 * self._sample_half_size_mm[1],
                0.0,
            ],
            dtype=float,
        )

        # Fixed lab ray rendered ONLY from above to the moving sample surface.
        # It never passes visually through the hexapod mechanism.
        self._beam_origin = top_center + np.array([0.0, 340.0, 0.0])
        self._beam_direction = np.array([0.0, -1.0, 0.0])
        self._beam_mesh = pv.Line(
            self._beam_origin,
            self._sample_surface_point_home,
        )
        self._beam_actor = self.add_mesh(
            self._beam_mesh,
            color="#ff4f49",
            line_width=5,
            opacity=0.22,
            render_lines_as_tubes=True,
            name="laser-segment",
        )
        self._beam_hit_mesh = pv.Sphere(
            radius=4.8,
            center=self._sample_surface_point_home,
            theta_resolution=28,
            phi_resolution=20,
        )
        self._beam_hit_actor = self.add_mesh(
            self._beam_hit_mesh,
            color="#ffd06a",
            opacity=0.45,
            name="beam-hit",
        )

        anchor = np.asarray([self._sample_surface_point_home], dtype=float)
        self._travel_mesh = self._make_line_poly(anchor)
        self._travel_actor = self.add_mesh(
            self._travel_mesh,
            color="#7d8c99",
            line_width=2,
            opacity=0.35,
            name="travel-trace",
        )
        self._travel_actor.SetVisibility(False)
        self._process_mesh = self._make_line_poly(anchor.copy())
        self._process_actor = self.add_mesh(
            self._process_mesh,
            color="#ffb236",
            line_width=7,
            render_lines_as_tubes=True,
            name="process-trace",
        )
        self._process_actor.SetVisibility(False)

        # Calibrated sample outline, drawn on the moving sample surface.
        self._sample_outline_mesh = self._make_line_poly(anchor.copy())
        self._sample_outline_actor = self.add_mesh(
            self._sample_outline_mesh,
            color="#4fa8d8",
            line_width=4,
            render_lines_as_tubes=True,
            name="sample-outline",
        )
        self._sample_outline_actor.SetVisibility(False)

        self.add_text(
            "LIVE DIGITAL TWIN  •  fixed beam / moving sample",
            position="upper_left",
            font_size=10,
            color="#a9b8c5",
            name="mode-label",
        )

    @property
    def beam_hit_sample_xy(self) -> tuple[float, float] | None:
        """Current laser footprint in moving sample coordinates (X,Y), mm."""
        return self._beam_hit_sample_xy

    @property
    def beam_hit_world(self) -> np.ndarray | None:
        return None if self._beam_hit_world is None else self._beam_hit_world.copy()

    def _beam_sample_surface_intersection(
        self,
        pose: Pose6D,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        top_tf = self.kinematics.top_transform(pose)
        surface_point = transform_point(
            top_tf,
            self._sample_surface_point_home,
        )
        surface_normal = top_tf[:3, :3] @ np.array([0.0, 1.0, 0.0])
        denom = float(np.dot(surface_normal, self._beam_direction))
        if abs(denom) < 1e-10:
            return None, top_tf
        ray_t = float(
            np.dot(
                surface_normal,
                surface_point - self._beam_origin,
            )
            / denom
        )
        if ray_t < 0.0:
            return None, top_tf
        return self._beam_origin + ray_t * self._beam_direction, top_tf

    def _invalidate(self) -> None:
        self._last_render_key = None

    def clear_traces(self) -> None:
        self._invalidate()
        self._run_arrays.clear()
        self._segments.clear()
        self._sync_traces()
        self.render()

    def set_trace_visibility(self, *, travel: bool | None = None, written: bool | None = None) -> None:
        self._invalidate()
        if travel is not None:
            self._show_travel = bool(travel)
        if written is not None:
            self._show_written = bool(written)
        self._sync_traces()
        self.render()

    def set_sample_outline(self, polygon_xy: Iterable[tuple[float, float]] | None) -> None:
        """Draw the calibrated sample boundary on the sample surface.

        ``polygon_xy`` is expressed in the same sample-local (X, Z) frame that
        :attr:`beam_hit_sample_xy` reports, so the outline lines up exactly with
        the captured edge points.
        """

        self._invalidate()
        polygon = [] if polygon_xy is None else [tuple(p) for p in polygon_xy]
        if len(polygon) < 3:
            self._sample_outline_actor.SetVisibility(False)
            self.render()
            return
        height = float(self._sample_surface_point_home[1]) + 0.15
        loop = [*polygon, polygon[0]]
        points = np.asarray(
            [[float(x), height, float(z)] for x, z in loop],
            dtype=float,
        )
        self._sample_outline_mesh.points = points
        self._sample_outline_mesh.lines = np.hstack(
            ([len(points)], np.arange(len(points), dtype=np.int64))
        )
        self._sample_outline_actor.SetVisibility(True)
        self._set_actor_matrix(
            self._sample_outline_actor,
            self.kinematics.top_transform(self._pose),
        )
        self.render()

    def set_view(self, name: str) -> None:
        """Snap the camera to a named operator view."""
        presets = {
            "iso": [(850, 650, 850), (0, 220, 0), (0, 1, 0)],
            "top": [(0, 1250, 0.1), (0, 220, 0), (0, 0, -1)],
            "front": [(0, 330, 1250), (0, 220, 0), (0, 1, 0)],
            "side": [(1250, 330, 0), (0, 220, 0), (0, 1, 0)],
        }
        self.camera_position = presets.get(str(name).lower(), presets["iso"])
        self.render()

    @staticmethod
    def _multi_cell_lines(runs: list[list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        cells: list[int] = []
        for run in runs:
            if len(run) < 2:
                continue
            start = len(points)
            points.extend(run)
            cells.append(len(run))
            cells.extend(range(start, start + len(run)))
        if not points:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=np.int64)
        return (
            np.asarray(points, dtype=float),
            np.asarray(cells, dtype=np.int64),
        )

    def _run_array(self, index: int, run: list[np.ndarray]) -> np.ndarray:
        cached = self._run_arrays.get(index)
        if cached is not None and cached[0] == len(run):
            return cached[1]
        array = np.asarray(run, dtype=float)
        self._run_arrays[index] = (len(run), array)
        return array

    def _sync_traces(self) -> None:
        for laser_on, mesh, actor, visible in (
            (False, self._travel_mesh, self._travel_actor, self._show_travel),
            (True, self._process_mesh, self._process_actor, self._show_written),
        ):
            arrays = [
                self._run_array(index, run)
                for index, (state, run) in enumerate(self._segments)
                if state is laser_on and len(run) >= 2
            ]
            if not arrays or not visible:
                actor.SetVisibility(False)
                continue
            points = np.concatenate(arrays)
            counts = [len(array) for array in arrays]
            cells = np.empty(sum(counts) + len(counts), dtype=np.int64)
            position = 0
            start = 0
            for count in counts:
                cells[position] = count
                cells[position + 1 : position + 1 + count] = np.arange(
                    start, start + count, dtype=np.int64
                )
                position += count + 1
                start += count
            mesh.points = points
            mesh.lines = cells
            actor.SetVisibility(True)

    #: Cap on stored trace points, so an all-day session cannot grow the
    #: polydata without bound. Oldest runs are dropped first.
    MAX_TRACE_POINTS = 60_000

    def _trim_segments(self) -> None:
        total = sum(len(run) for _, run in self._segments)
        dropped = False
        while total > self.MAX_TRACE_POINTS and len(self._segments) > 1:
            total -= len(self._segments[0][1])
            self._segments.pop(0)
            dropped = True
        if dropped:
            # Cache keys are positional, so they no longer line up.
            self._run_arrays.clear()

    def _append_trace(self, point: np.ndarray, *, laser_on: bool) -> None:
        threshold = 0.05 if laser_on else 0.20
        if not self._segments:
            self._segments.append((laser_on, [point.copy()]))
            return
        state, run = self._segments[-1]
        if state is not laser_on:
            # Continue from the exact pose where the beam state changed.
            self._segments.append((laser_on, [run[-1].copy(), point.copy()]))
            return
        if float(np.linalg.norm(point - run[-1])) >= threshold:
            run.append(point.copy())

    def update_state(self, pose: Pose6D, laser_on: bool) -> None:
        laser_on = bool(laser_on)
        key = (pose.as_tuple(), laser_on, self._show_travel, self._show_written)
        if key == self._last_render_key:
            # Nothing the twin draws has moved since the last frame.
            return
        self._last_render_key = key
        self._pose = pose; self._laser_on = laser_on
        top_tf = self.kinematics.top_transform(pose)
        top_joints = self.kinematics.top_joint_positions(pose)
        bottom = self.kinematics.bottom_joint_positions()
        self._set_actor_matrix(self._top_actor, top_tf)
        self._set_actor_matrix(self._sample_actor, top_tf)
        points = self._leg_mesh.points.copy()
        for i in range(6):
            points[2 * i] = bottom[i]; points[2 * i + 1] = top_joints[i]
        self._leg_mesh.points = points

        hit, top_tf = self._beam_sample_surface_intersection(pose)
        self._beam_hit_world = None if hit is None else hit.copy()
        self._beam_hit_sample_xy = None

        if hit is not None:
            # Laser graphic terminates exactly at the top surface of the sample.
            self._beam_mesh.points = np.vstack([self._beam_origin, hit])
            self._beam_actor.SetVisibility(True)
            beam_property = self._beam_actor.GetProperty()
            # Colour, not only opacity, separates an armed beam from an idle one.
            beam_property.SetColor(
                (1.0, 0.31, 0.29) if laser_on else (0.36, 0.45, 0.52)
            )
            beam_property.SetOpacity(0.98 if laser_on else 0.14)
            beam_property.SetLineWidth(7.0 if laser_on else 2.0)

            sphere = pv.Sphere(
                radius=5.5 if laser_on else 3.0,
                center=hit,
                theta_resolution=28,
                phi_resolution=20,
            )
            self._beam_hit_mesh.points = sphere.points
            self._beam_hit_actor.SetVisibility(True)
            hit_property = self._beam_hit_actor.GetProperty()
            hit_property.SetColor(
                (1.0, 0.70, 0.21) if laser_on else (0.55, 0.62, 0.68)
            )
            hit_property.SetOpacity(0.95 if laser_on else 0.18)

            inv_top = np.linalg.inv(top_tf)
            local_hit = transform_point(inv_top, hit)
            self._beam_hit_sample_xy = (
                float(local_hit[0]),
                float(local_hit[2]),
            )
            self._append_trace(local_hit, laser_on=laser_on)
            self._trim_segments()
        else:
            self._beam_actor.SetVisibility(False)
            self._beam_hit_actor.SetVisibility(False)

        self._sync_traces()
        # The path is sample-local, so it moves with the sample instead of
        # hanging in laboratory space after the carriage moves.
        self._set_actor_matrix(self._travel_actor, top_tf)
        self._set_actor_matrix(self._process_actor, top_tf)
        self._set_actor_matrix(self._sample_outline_actor, top_tf)

        if self._exact_cad:
            self._update_exact_cad(pose)
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
            self._cad_actors[idx] = self.add_mesh(
                mesh,
                color=color,
                smooth_shading=True,
                specular=0.48,
                specular_power=34,
                ambient=0.18,
                diffuse=0.78,
                name=f"cad-{idx:02d}",
            )
        self._exact_cad=True
        self._invalidate()
        self._base_actor.GetProperty().SetOpacity(0.05)
        self._top_actor.GetProperty().SetOpacity(0.05)
        self._leg_actor.GetProperty().SetOpacity(0.08)
        self._update_exact_cad(self._pose)
        self.reset_camera()
        self.camera_position = [
            (760, 590, 820),
            (0, 245, 0),
            (0, 1, 0),
        ]
        self._status(
            f"Loaded exact STEP assembly ({len(cache.mesh_paths)} solids); "
            "CAD articulation enabled"
        )
