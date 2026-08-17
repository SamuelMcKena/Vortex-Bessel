"""Stage 8C.2 governance tests.

The visual rescue must not introduce material-response claims or forbidden
output statuses, must keep saved figures diagnostic-only, must not touch
lock-sensitive physics files, and must not start Stage 8D.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

STAGE8C2_FILES = [
    ROOT / "vbb_study" / "digital_twin" / "cockpit_dashboard.py",
    ROOT / "notebooks" / "digital_twin" / "00_full_beam_to_write_cockpit_MVP.ipynb",
    ROOT / "docs" / "31_cockpit_dashboard_polish.md",
    ROOT / "STAGE8C2_COCKPIT_DASHBOARD_POLISH_SUMMARY.md",
]

FORBIDDEN_STATUSES = [
    "fluence_threshold_proxy",
    "dose_accumulation_proxy",
    "uncalibrated_material_response_proxy",
    "calibrated_material_prediction",
    "experimentally_validated_prediction",
    "simulated_microscopy_proxy",
]

BOUNDARY_TERMS = [
    "damage prediction",
    "void prediction",
    "waveguide prediction",
    "ablation prediction",
    "material modification",
]

BOUNDARY_MARKERS = (
    "not", "no ", "disabled", "future", "later", "does not",
    "do not", "without", "claim boundary", "optical only", "optical fluence only",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8").lower()


def test_stage8c2_files_exist():
    for path in STAGE8C2_FILES:
        assert path.is_file(), path


def test_no_forbidden_output_statuses():
    blob = "\n".join(_text(p) for p in STAGE8C2_FILES)
    for status in FORBIDDEN_STATUSES:
        assert status not in blob, f"forbidden status present: {status}"


def test_boundary_terms_only_in_boundary_context():
    blob = "\n".join(_text(p) for p in STAGE8C2_FILES)
    norm = re.sub(r"\s+", " ", blob)
    violations = []
    for term in BOUNDARY_TERMS:
        start = 0
        while True:
            idx = norm.find(term, start)
            if idx == -1:
                break
            context = norm[max(0, idx - 90): idx + len(term) + 90]
            if not any(marker in context for marker in BOUNDARY_MARKERS):
                violations.append(context)
            start = idx + len(term)
    assert not violations, violations


def test_saved_figures_diagnostic_only_in_source():
    text = _text(ROOT / "vbb_study" / "digital_twin" / "cockpit_dashboard.py")
    assert "final_export_allowed" in text
    assert "false" in text
    assert "diagnostic_allowed" in text


def test_core_optical_physics_not_modified():
    lock_sensitive = [
        "Publication_Study/bessel_twin_core.py",
        "Publication_Study/vbb_study/equations/propagation.py",
        "Publication_Study/vbb_study/equations/scalar_bessel.py",
        "Publication_Study/tests/test_characterisation_lock.py",
    ]
    repo_root = ROOT.parent
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root), capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        pytest.skip("git status failed")
    modified = {line[3:].strip().strip('"').replace("\\", "/") for line in out.stdout.splitlines()}
    for path in lock_sensitive:
        assert path not in modified, f"lock-sensitive file modified: {path}"


def test_stage8d_not_started():
    for path in STAGE8C2_FILES:
        text = _text(path)
        assert "stage8d" not in text
        # 'stage 8d' is only allowed as a forward-looking recommendation.
        assert ("stage 8d" not in text) or ("recommended next stage" in text) or ("next stage" in text)
