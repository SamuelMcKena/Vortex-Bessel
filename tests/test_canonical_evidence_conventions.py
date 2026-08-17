from __future__ import annotations

import numpy as np
import pytest

from vbb_study.reporting.evidence_conventions import (
    CanonicalEvidenceSpec,
    INTENSITY_CMAP,
    SIGNED_CMAP,
    canonical_evidence_filenames,
    common_heatmap_scale,
    nominal_profile_scale,
)


def test_primary_evidence_contract_is_fixed_lab_linear_common_scale() -> None:
    spec = CanonicalEvidenceSpec(z_ref_mm=30.0)
    spec.validate()
    assert spec.longitudinal_frame == "fixed_laboratory_coordinates_no_z_dependent_recentering"
    assert spec.primary_intensity_scale == "linear_0_to_1"
    assert spec.primary_intensity_colormap == "turbo"
    assert spec.per_case_peak_normalisation_allowed_primary is False
    assert spec.per_z_normalisation_allowed_primary is False
    assert spec.beam_following_crop_allowed_primary_longitudinal is False
    assert spec.log_intensity_allowed_primary is False
    assert INTENSITY_CMAP == "turbo"
    assert SIGNED_CMAP == "coolwarm"


def test_common_heatmap_scale_never_peak_normalises_each_panel() -> None:
    a = np.asarray([[0.0, 1.0], [2.0, 0.0]])
    b = np.asarray([[0.0, 4.0], [1.0, 0.0]])
    scaled, peak = common_heatmap_scale([a, b])
    assert peak == 4.0
    assert np.max(scaled[0]) == 0.5
    assert np.max(scaled[1]) == 1.0


def test_profiles_use_nominal_2d_peak_and_may_exceed_one() -> None:
    nominal = np.asarray([[0.0, 2.0], [1.0, 0.0]])
    scale = nominal_profile_scale(nominal)
    perturbed_line = np.asarray([0.0, 3.0, 0.0]) / scale
    assert scale == 2.0
    assert np.max(perturbed_line) == 1.5


def test_canonical_filenames_encode_pair_and_zref() -> None:
    names = canonical_evidence_filenames("phase2h_six_sector_rot_x", 30.0)
    assert names["longitudinal"].endswith("__longitudinal_fixed_lab.png")
    assert names["zref_profiles"].endswith("__profiles_zref_30mm.png")
    assert names["raw_npz"].endswith("__evidence.npz")
    assert names["manifest"].endswith("__manifest.json")


def test_invalid_reference_plane_is_rejected() -> None:
    with pytest.raises(ValueError):
        CanonicalEvidenceSpec(z_ref_mm=-1.0).validate()
