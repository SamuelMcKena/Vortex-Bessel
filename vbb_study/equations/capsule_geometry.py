"""Reusable capsule-geometry equations for application-planning proxies.

A capsule target is a geometric outline: a rectangle with semicircular end caps
along the z axis. Comparing a thresholded XZ proxy with this target measures
geometric overlap only. It does not predict actual welding, bonding, ablation,
void formation, or refractive-index change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1.0e-30


def capsule_mask_2d(
    x_um: Any,
    z_um: Any,
    *,
    width_um: float,
    length_um: float,
    center_x_um: float = 0.0,
    center_z_um: float | None = None,
) -> np.ndarray:
    """Return a Boolean XZ capsule mask.

    ``x_um`` is the transverse axis and ``z_um`` is the axial/depth axis. The
    capsule has full width ``width_um`` and full length ``length_um``. If length
    is not larger than width, the target degenerates to a rounded compact target
    rather than a weld/material-response model.
    """

    x = np.asarray(x_um, dtype=float)
    z = np.asarray(z_um, dtype=float)
    if x.ndim != 1 or z.ndim != 1:
        raise ValueError("x_um and z_um must be one-dimensional axes")
    width = max(float(width_um), 0.0)
    length = max(float(length_um), 0.0)
    if width <= 0.0 or length <= 0.0:
        return np.zeros((len(x), len(z)), dtype=bool)
    cx = float(center_x_um)
    cz = float(0.5 * (z[0] + z[-1]) if center_z_um is None and len(z) else center_z_um or 0.0)
    half_w = 0.5 * width
    half_l = 0.5 * length
    straight_half = max(0.0, half_l - half_w)
    X, Z = np.meshgrid(x, z, indexing="ij")
    dx = np.abs(X - cx)
    dz = np.abs(Z - cz)
    if straight_half <= 0.0:
        return (dx * dx + dz * dz) <= half_w * half_w
    body = (dz <= straight_half) & (dx <= half_w)
    cap_distance = np.sqrt(dx * dx + (dz - straight_half) ** 2)
    caps = (dz > straight_half) & (cap_distance <= half_w)
    return body | caps


def capsule_area_um2(*, width_um: float, length_um: float) -> float:
    """Return the analytic capsule area in square microns."""

    width = max(float(width_um), 0.0)
    length = max(float(length_um), 0.0)
    if width <= 0.0 or length <= 0.0:
        return 0.0
    if length <= width:
        return float(np.pi * (0.5 * width) ** 2)
    return float(width * (length - width) + np.pi * (0.5 * width) ** 2)


def capsule_equivalent_diameter_um(area_um2: float) -> float:
    """Return the diameter of a circle with the supplied area in microns."""

    area = max(float(area_um2), 0.0)
    return float(2.0 * np.sqrt(area / np.pi)) if area > 0.0 else 0.0


def capsule_aspect_ratio(*, width_um: float, length_um: float) -> float:
    """Return ``length / width`` for a capsule target."""

    width = max(float(width_um), EPS)
    return float(max(float(length_um), 0.0) / width)


def overlap_score(mask_a: Any, mask_b: Any) -> float:
    """Return Jaccard overlap between two Boolean masks."""

    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("masks must have the same shape")
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def width_length_from_mask(mask_xz: Any, x_um: Any, z_um: Any) -> dict[str, float]:
    """Return equivalent width, length, and area from a thresholded XZ mask."""

    mask = np.asarray(mask_xz, dtype=bool)
    x = np.asarray(x_um, dtype=float)
    z = np.asarray(z_um, dtype=float)
    if mask.shape != (len(x), len(z)):
        raise ValueError("mask shape must be (len(x_um), len(z_um))")
    if not np.any(mask):
        return {"width_um": 0.0, "length_um": 0.0, "area_um2": 0.0}
    ix, iz = np.where(mask)
    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 0.0
    dz = float(np.mean(np.diff(z))) if len(z) > 1 else 0.0
    width = float((np.max(x[ix]) - np.min(x[ix])) + abs(dx))
    length = float((np.max(z[iz]) - np.min(z[iz])) + abs(dz))
    area = float(np.count_nonzero(mask) * abs(dx or 1.0) * abs(dz or 1.0))
    return {"width_um": max(0.0, width), "length_um": max(0.0, length), "area_um2": max(0.0, area)}


def edge_uniformity_score(mask_xz: Any, x_um: Any, z_um: Any) -> float:
    """Return a bounded score for z-to-z width uniformity of a mask."""

    mask = np.asarray(mask_xz, dtype=bool)
    x = np.asarray(x_um, dtype=float)
    z = np.asarray(z_um, dtype=float)
    if mask.shape != (len(x), len(z)) or not np.any(mask):
        return 0.0
    widths = []
    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col])[0]
        if rows.size:
            widths.append(float(np.max(x[rows]) - np.min(x[rows])))
    if len(widths) < 2:
        return 0.0
    arr = np.asarray(widths, dtype=float)
    cv = float(np.std(arr) / (np.mean(arr) + EPS))
    return float(1.0 / (1.0 + cv))


def side_lobe_contamination_score(proxy_mask: Any, target_mask: Any) -> float:
    """Return the fraction of proxy pixels outside the desired target."""

    proxy = np.asarray(proxy_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if proxy.shape != target.shape:
        raise ValueError("masks must have the same shape")
    proxy_count = np.count_nonzero(proxy)
    if proxy_count == 0:
        return 0.0
    return float(np.count_nonzero(proxy & ~target) / proxy_count)


def accepted_depth_from_xz_proxy(mask_xz: Any, z_um: Any) -> float:
    """Return the non-negative accepted axial/depth span from an XZ proxy mask."""

    mask = np.asarray(mask_xz, dtype=bool)
    z = np.asarray(z_um, dtype=float)
    if mask.ndim != 2 or mask.shape[1] != len(z) or not np.any(mask):
        return 0.0
    cols = np.where(np.any(mask, axis=0))[0]
    if cols.size < 2:
        return 0.0
    return float(max(0.0, z[cols[-1]] - z[cols[0]]))


__all__ = [
    "accepted_depth_from_xz_proxy",
    "capsule_area_um2",
    "capsule_aspect_ratio",
    "capsule_equivalent_diameter_um",
    "capsule_mask_2d",
    "edge_uniformity_score",
    "overlap_score",
    "side_lobe_contamination_score",
    "width_length_from_mask",
]
