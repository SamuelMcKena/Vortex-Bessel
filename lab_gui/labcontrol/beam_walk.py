"""Multi-plane beam-camera relative propagation analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PlaneObservation:
    frame_id: str
    z_mm: float
    centre_y_px: float
    centre_x_px: float
    ring_radius_px: float | None = None
    power_proxy: float | None = None


@dataclass(frozen=True)
class BeamWalkResult:
    per_frame: tuple[dict[str, Any], ...]
    per_plane: tuple[dict[str, Any], ...]
    fit: dict[str, float | str | bool]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_frame": list(self.per_frame),
            "per_plane": list(self.per_plane),
            "fit": dict(self.fit),
            "interpretation": self.interpretation,
        }


def _r_squared(values: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((values - np.mean(values)) ** 2))
    return 1.0 if denominator == 0 else float(
        1.0 - np.sum((values - predicted) ** 2) / denominator
    )


def fit_beam_walk(
    observations: Iterable[PlaneObservation],
    *,
    pixel_size_um: float,
    camera_axis_calibrated: bool = False,
) -> BeamWalkResult:
    rows = list(observations)
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Camera pixel size must be finite and positive.")
    if len({float(row.z_mm) for row in rows}) < 2:
        raise ValueError("Beam-walk fitting requires at least two distinct z planes.")
    if not all(
        np.isfinite([row.z_mm, row.centre_y_px, row.centre_x_px]).all() for row in rows
    ):
        raise ValueError("Beam-walk observations must contain finite z and centres.")

    per_plane: list[dict[str, Any]] = []
    for z in sorted({float(row.z_mm) for row in rows}):
        group = [row for row in rows if float(row.z_mm) == z]
        centres = np.asarray([[row.centre_y_px, row.centre_x_px] for row in group])
        radii = [row.ring_radius_px for row in group if row.ring_radius_px is not None]
        powers = [row.power_proxy for row in group if row.power_proxy is not None]
        per_plane.append(
            {
                "z_mm": z,
                "repeats": len(group),
                "mean_y_px": float(np.mean(centres[:, 0])),
                "mean_x_px": float(np.mean(centres[:, 1])),
                "std_y_px": float(np.std(centres[:, 0], ddof=1)) if len(group) > 1 else 0.0,
                "std_x_px": float(np.std(centres[:, 1], ddof=1)) if len(group) > 1 else 0.0,
                "mean_ring_radius_px": float(np.mean(radii)) if radii else None,
                "mean_power_proxy": float(np.mean(powers)) if powers else None,
            }
        )

    z = np.asarray([row["z_mm"] for row in per_plane], dtype=np.float64)
    x = np.asarray([row["mean_x_px"] for row in per_plane], dtype=np.float64)
    y = np.asarray([row["mean_y_px"] for row in per_plane], dtype=np.float64)
    fit_x = np.polyfit(z, x, 1)
    fit_y = np.polyfit(z, y, 1)
    predicted_x = np.polyval(fit_x, z)
    predicted_y = np.polyval(fit_y, z)
    sx_mrad = float(fit_x[0] * pixel_size_um)
    sy_mrad = float(fit_y[0] * pixel_size_um)
    total_mrad = float(np.hypot(sx_mrad, sy_mrad))
    repeat_scatter = float(
        np.sqrt(
            np.mean(
                [row["std_x_px"] ** 2 + row["std_y_px"] ** 2 for row in per_plane]
            )
        )
    )
    label = (
        "calibrated optical-axis angle"
        if camera_axis_calibrated
        else "measured beam-camera relative propagation mismatch"
    )
    interpretation = (
        "Camera travel is independently calibrated; the small-angle slope is reported against the "
        "calibrated optical axis."
        if camera_axis_calibrated
        else "Camera travel is not independently calibrated parallel to the desired optical axis. "
        "This is measured beam-camera relative propagation mismatch, not an absolute laser angle."
    )
    fit = {
        "label": label,
        "camera_axis_calibrated": camera_axis_calibrated,
        "x_intercept_px": float(fit_x[1]),
        "y_intercept_px": float(fit_y[1]),
        "x_px_per_mm": float(fit_x[0]),
        "y_px_per_mm": float(fit_y[0]),
        "x_mrad": sx_mrad,
        "y_mrad": sy_mrad,
        "total_mrad": total_mrad,
        "direction_deg": float(math.degrees(math.atan2(sy_mrad, sx_mrad))),
        "r2_x": _r_squared(x, predicted_x),
        "r2_y": _r_squared(y, predicted_y),
        "repeat_scatter_px_rms": repeat_scatter,
    }
    return BeamWalkResult(
        per_frame=tuple(
            {
                "frame_id": row.frame_id,
                "z_mm": float(row.z_mm),
                "centre_y_px": float(row.centre_y_px),
                "centre_x_px": float(row.centre_x_px),
                "ring_radius_px": row.ring_radius_px,
                "power_proxy": row.power_proxy,
            }
            for row in rows
        ),
        per_plane=tuple(per_plane),
        fit=fit,
        interpretation=interpretation,
    )


def compare_beam_walk(
    reference: BeamWalkResult,
    vortex: BeamWalkResult,
    *,
    material_difference_mrad: float = 0.25,
) -> dict[str, float | str | bool]:
    dx = float(vortex.fit["x_mrad"]) - float(reference.fit["x_mrad"])
    dy = float(vortex.fit["y_mrad"]) - float(reference.fit["y_mrad"])
    difference = float(np.hypot(dx, dy))
    differs = difference > float(material_difference_mrad)
    return {
        "delta_x_mrad": dx,
        "delta_y_mrad": dy,
        "slope_difference_mrad": difference,
        "material_difference_threshold_mrad": float(material_difference_mrad),
        "materially_different": differs,
        "classification": (
            "possible vortex/SLM/asymmetric contribution"
            if differs
            else "primarily common post-axicon or measurement geometry"
        ),
    }
