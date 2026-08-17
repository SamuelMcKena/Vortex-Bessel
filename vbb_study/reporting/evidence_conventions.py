"""Repository-wide conventions for comparable optical-field evidence.

This module contains *numerical presentation contracts*, not Matplotlib styling
for its own sake.  The aim is to prevent comparative beam figures from changing
meaning when different renderers choose different normalisation, coordinates or
reference planes.

Primary intensity evidence uses a linear common scale.  Diagnostic log views,
beam-following crops and individually normalised shape plots may exist only as
explicitly supplementary outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


INTENSITY_CMAP = "turbo"
SIGNED_CMAP = "coolwarm"
PRIMARY_HEATMAP_RANGE = (0.0, 1.0)
DEFAULT_FIGURE_DPI = 260
HIGH_RES_FIGURE_DPI = 300


@dataclass(frozen=True)
class CanonicalEvidenceSpec:
    """Meaning of one canonical sweep evidence package."""

    z_ref_mm: float
    z_ref_policy: str = "a_priori_fixed_physical_plane_shared_by_all_comparison_cases"
    longitudinal_frame: str = "fixed_laboratory_coordinates_no_z_dependent_recentering"
    longitudinal_normalisation: str = "one_common_peak_for_all_comparable_longitudinal_heatmaps"
    transverse_heatmap_normalisation: str = "one_common_peak_for_all_comparable_zref_2d_maps"
    profile_normalisation: str = "nominal_case_2d_peak_at_same_zref"
    profile_sampling: str = "direct_complex_field_Fourier_series_not_intensity_image_interpolation"
    primary_intensity_scale: str = "linear_0_to_1"
    primary_intensity_colormap: str = INTENSITY_CMAP
    signed_quantity_colormap: str = SIGNED_CMAP
    per_case_peak_normalisation_allowed_primary: bool = False
    per_z_normalisation_allowed_primary: bool = False
    beam_following_crop_allowed_primary_longitudinal: bool = False
    log_intensity_allowed_primary: bool = False

    def validate(self) -> None:
        if not np.isfinite(self.z_ref_mm) or self.z_ref_mm < 0.0:
            raise ValueError("z_ref_mm must be finite and non-negative")
        if self.per_case_peak_normalisation_allowed_primary:
            raise ValueError("primary comparative evidence cannot peak-normalise each case")
        if self.per_z_normalisation_allowed_primary:
            raise ValueError("primary longitudinal evidence cannot normalise each z independently")
        if self.beam_following_crop_allowed_primary_longitudinal:
            raise ValueError("primary longitudinal evidence must remain in fixed laboratory coordinates")
        if self.log_intensity_allowed_primary:
            raise ValueError("log intensity is supplementary, not the primary heatmap convention")


def common_positive_peak(arrays: Iterable[np.ndarray]) -> float:
    """Return one finite positive maximum across a comparison family."""

    peak = 0.0
    count = 0
    for values in arrays:
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            peak = max(peak, float(np.max(finite)))
            count += 1
    if count == 0 or not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("comparison family has no finite positive intensity peak")
    return peak


def common_heatmap_scale(arrays: Sequence[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """Return arrays divided by one family-wide peak and that physical peak.

    This is deliberately different from normalising each panel independently.
    The returned arrays are suitable for the canonical linear [0,1] intensity
    heatmap convention.
    """

    peak = common_positive_peak(arrays)
    return [np.asarray(values, dtype=float) / peak for values in arrays], peak


def nominal_profile_scale(nominal_2d_intensity: np.ndarray) -> float:
    """Return the nominal 2-D peak used for comparative 1-D profiles."""

    return common_positive_peak([np.asarray(nominal_2d_intensity, dtype=float)])


def canonical_evidence_filenames(family: str, z_ref_mm: float) -> dict[str, str]:
    """Return stable canonical output names for a sweep family."""

    clean = str(family).strip().replace(" ", "_").replace("/", "_")
    if not clean:
        raise ValueError("family must be non-empty")
    tag = f"{float(z_ref_mm):g}".replace("-", "m").replace(".", "p")
    return {
        "longitudinal": f"{clean}__longitudinal_fixed_lab.png",
        "zref_profiles": f"{clean}__profiles_zref_{tag}mm.png",
        "metrics_csv": f"{clean}__metrics.csv",
        "raw_npz": f"{clean}__evidence.npz",
        "manifest": f"{clean}__manifest.json",
    }


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


__all__ = [
    "CanonicalEvidenceSpec",
    "DEFAULT_FIGURE_DPI",
    "HIGH_RES_FIGURE_DPI",
    "INTENSITY_CMAP",
    "PRIMARY_HEATMAP_RANGE",
    "SIGNED_CMAP",
    "canonical_evidence_filenames",
    "common_heatmap_scale",
    "common_positive_peak",
    "ensure_parent",
    "nominal_profile_scale",
]
