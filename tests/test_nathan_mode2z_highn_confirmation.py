from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbb_study.digital_twin.nathan_mode2z_highn_confirmation import (
    MODE2Z_HN_ALLOWED_OUTCOMES,
    MODE2Z_HN_DEFAULT_ETA_VALUES,
    MODE2Z_HN_MIN_GRID_N,
    MODE2Z_HN_Z_M,
    Mode2ZHighNConfig,
    mode2z_highn_audit,
    mode2z_highn_outcome,
    write_mode2z_highn_outputs,
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("mode2z_highn")
    config = Mode2ZHighNConfig(
        grid_n=192,
        eta_values=MODE2Z_HN_DEFAULT_ETA_VALUES,
        baseline_summary_path=None,
        publication_quality=False,
    )
    return write_mode2z_highn_outputs(
        root / "outputs",
        document_path=root / "docs/87_nathan_mode2z_targeted_highn_confirmation.md",
        parent_document_path=None,
        config=config,
    )


def _audit_rows(*, onset: float = 0.7):
    rows = []
    for eta, corr, edge, width, fwhm in (
        (0.0, 0.975, 7.0, 0.16, 0.10),
        (0.4, 0.980, 7.8, 0.15, 0.09),
        (0.6, 0.985, 8.6, 0.14, 0.08),
        (0.7, 0.989, 9.2, 0.13, 0.075),
        (0.8, 0.992, 9.8, 0.125, 0.072),
        (1.0, 0.995, 10.5, 0.12, 0.070),
    ):
        rows.append({
            "eta": eta,
            "z60_correlation_to_v0": corr,
            "edge_gradient_sharpness_mm_inv": edge,
            "threshold_transition_width_mm": width,
            "bright_ridge_fwhm_mm": fwhm,
            "peak_intensity": 72.0 - 1.5 * eta,
            "useful_region_power": 1000.0 - 4.0 * eta,
            "strict_hexagon_pass": eta >= onset,
        })
    return rows


def test_publication_contract_is_n1536_selected_eta_and_exact_z60():
    config = Mode2ZHighNConfig()
    assert config.grid_n == MODE2Z_HN_MIN_GRID_N == 1536
    assert config.eta_values == (0.0, 0.4, 0.6, 0.7, 0.8, 1.0)
    assert config.z_m == pytest.approx(MODE2Z_HN_Z_M)
    with pytest.raises(ValueError):
        Mode2ZHighNConfig(grid_n=384, publication_quality=True).validate()
    with pytest.raises(ValueError):
        Mode2ZHighNConfig(z_m=0.061).validate()


def test_only_selected_realistic_focus_planes_are_generated(generated):
    result = generated["result"]
    assert set(result.planes_by_eta) == set(result.config.eta_values)
    assert all(plane.shape == (192, 192) for plane in result.planes_by_eta.values())
    assert all(row["optical_route"] == "realistic" for row in result.summary_rows)
    assert all(float(row["z_mm"]) == pytest.approx(60.0) for row in result.summary_rows)


def test_metrics_are_native_and_not_display_interpolated(generated):
    result = generated["result"]
    assert result.audit["native_metrics_only"] is True
    assert result.audit["sampling_audit"]["carrier_band_nyquist_pass"] is True
    assert result.audit["sampling_audit"]["samples_per_carrier_period"] > 2.0
    assert result.audit["sampling_audit"]["filtered_band_nyquist_margin"] > 1.0
    for row in result.summary_rows:
        assert row["metrics_native_grid_n"] == 192
        assert row["display_interpolation_used_for_metrics"] is False


def test_input_interpolation_retains_exact_continuous_endpoint(generated):
    rows = {float(row["eta"]): row for row in generated["result"].input_rows}
    assert rows[1.0]["local_angle_rms_rad"] < 1e-10
    assert rows[0.0]["local_angle_rms_rad"] > rows[1.0]["local_angle_rms_rad"]


def test_audit_confirms_selected_onset_and_stable_monotonic_endpoints():
    high = _audit_rows(onset=0.7)
    baseline = [
        {**row, "grid_n": 1024}
        for row in _audit_rows(onset=0.7)
    ]
    audit = mode2z_highn_audit(high, baseline)
    assert audit["strict_onset_confirmed"] is True
    assert audit["highn_selected_strict_onset_eta"] == pytest.approx(0.7)
    assert audit["correlation_nondecreasing"] is True
    assert audit["edge_gradient_nondecreasing"] is True
    assert audit["transition_width_nonincreasing"] is True
    assert audit["ridge_fwhm_nonincreasing"] is True
    assert audit["endpoint_improvements_stable"] is True
    outcome, _ = mode2z_highn_outcome(audit)
    assert outcome == "M2Z-HN-A"


def test_shifted_selected_onset_is_reported_not_forced():
    high = _audit_rows(onset=0.8)
    baseline = [{**row, "grid_n": 1024} for row in _audit_rows(onset=0.7)]
    audit = mode2z_highn_audit(high, baseline)
    assert audit["strict_onset_confirmed"] is False
    assert audit["highn_selected_strict_onset_eta"] == pytest.approx(0.8)
    outcome, _ = mode2z_highn_outcome(audit)
    assert outcome == "M2Z-HN-B"


def test_outcome_is_predeclared_and_width_plateaus_are_counted(generated):
    result = generated["result"]
    assert result.outcome in MODE2Z_HN_ALLOWED_OUTCOMES
    assert result.audit["highn_unique_transition_width_levels"] >= 1
    assert result.audit["highn_unique_fwhm_levels"] >= 1


def test_publication_figures_tables_and_document_are_written(generated):
    for key in ("focus_png", "focus_pdf", "convergence_png", "threshold_png", "width_png"):
        assert Path(generated["figure_paths"][key]).is_file()
    for key in (
        "summary_csv",
        "summary_json",
        "convergence_csv",
        "convergence_json",
        "input_truth_csv",
        "audit_json",
        "manifest",
        "outcome_report",
        "document_path",
    ):
        assert Path(generated[key]).is_file()


def test_scope_is_sequential_realistic_and_threshold_is_not_universal(generated):
    manifest = json.loads(Path(generated["manifest"]).read_text(encoding="utf-8"))
    outcome = json.loads(Path(generated["outcome_report"]).read_text(encoding="utf-8"))
    document = Path(generated["document_path"]).read_text(encoding="utf-8")
    assert manifest["optical_route"] == "realistic sequential common-4F"
    assert manifest["targeted_confirmation_only"] is True
    assert manifest["split_arm_pbs_architecture_used"] is False
    assert manifest["microfabrication_sample_plane_success_claim"] is False
    assert outcome["no_universal_tolerance_claim"] is True
    assert "not a universal experimental tolerance" in document
    assert np.isclose(float(manifest["z_m"]), 0.06)
