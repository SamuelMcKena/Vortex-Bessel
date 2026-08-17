"""
Stage 8C governance tests.

Enforces the Stage 8C claim boundary: docs exist and forbid absorbed-energy /
material-modification / damage claims; the source modules use forbidden claim
phrases only in negation/caveat context; the optional diagnostic figure (if it
exists) is stamped final_export_allowed=False; and Stage 8C leaves known
lock-sensitive files untouched.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
DOC_28 = DOCS / "28_surfacefield_energy_scaled_cockpit.md"
SUMMARY = ROOT / "STAGE8C_SURFACEFIELD_ENERGY_SCALED_COCKPIT_SUMMARY.md"

STAGE8C_MODULES = [
    ROOT / "vbb_study" / "digital_twin" / "field_coupling.py",
    ROOT / "vbb_study" / "digital_twin" / "field_fluence.py",
    ROOT / "vbb_study" / "digital_twin" / "field_figures.py",
]

EXAMPLE_FIGURE = ROOT / "outputs" / "figures" / "digital_twin" / "stage8c_surfacefield_energy_scaled_preview.png"
EXAMPLE_CSV = ROOT / "outputs" / "csv" / "digital_twin" / "stage8c_field_fluence_summary_example.csv"

# Forbidden positive-claim phrases. May appear ONLY in a negation/caveat context.
FORBIDDEN_CLAIM_PHRASES = [
    "calibrated material prediction",
    "damage prediction",
    "void prediction",
    "waveguide prediction",
    "ablation prediction",
    "deposited energy volume",
]

# Statuses Stage 8C must never assign to its own outputs.
FORBIDDEN_STATUSES = [
    "fluence_threshold_proxy",
    "dose_accumulation_proxy",
    "uncalibrated_material_response_proxy",
    "calibrated_material_prediction",
    "experimentally_validated_prediction",
]

NEGATION_MARKERS = (
    "not ", "no ", "never", "without", "cannot", "forbid", "boundary",
    "does not", "must not", "n't", "neither", "nor ",
)


def _has_negation(line: str) -> bool:
    low = line.lower()
    return any(m in low for m in NEGATION_MARKERS)


def _forbidden_phrase_violations(text: str, window: int = 70) -> list[str]:
    """Return forbidden-phrase occurrences not preceded by a negation marker.

    Operates on whitespace-normalised text so prose line-wrapping does not split
    a negation from the phrase it qualifies.
    """
    norm = re.sub(r"\s+", " ", text.lower())
    violations: list[str] = []
    for phrase in FORBIDDEN_CLAIM_PHRASES:
        start = 0
        while True:
            idx = norm.find(phrase, start)
            if idx == -1:
                break
            before = norm[max(0, idx - window):idx]
            if not any(m in before for m in NEGATION_MARKERS):
                ctx = norm[max(0, idx - window):idx + len(phrase)]
                violations.append(f"{phrase!r} -> ...{ctx}")
            start = idx + len(phrase)
    return violations


# ---------------------------------------------------------------------------
# Docs exist
# ---------------------------------------------------------------------------


def test_doc_28_exists():
    assert DOC_28.is_file(), f"Missing doc: {DOC_28}"


def test_summary_exists():
    assert SUMMARY.is_file(), f"Missing summary: {SUMMARY}"


# ---------------------------------------------------------------------------
# Claim boundary in docs
# ---------------------------------------------------------------------------


def test_doc_claim_boundary_present():
    text = DOC_28.read_text(encoding="utf-8").lower()
    # Must explicitly state the optical-only boundary.
    assert "not absorbed" in text
    assert "not a dose" in text or "not dose" in text
    assert "not a damage prediction" in text or "not damage" in text
    assert "material modification" in text


def test_doc_states_volume_not_pulse_energy():
    text = DOC_28.read_text(encoding="utf-8").lower()
    assert "volume integral" in text
    assert "not" in text  # the section explains why it is NOT total pulse energy


def test_doc_explains_propagation_drift():
    text = DOC_28.read_text(encoding="utf-8").lower()
    assert "propagation" in text and "drift" in text


# ---------------------------------------------------------------------------
# Forbidden claim phrases only appear in negation context (module source)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_path", STAGE8C_MODULES, ids=lambda p: p.name)
def test_module_forbidden_phrases_only_in_negation(module_path):
    assert module_path.is_file(), f"Missing module: {module_path}"
    violations = _forbidden_phrase_violations(module_path.read_text(encoding="utf-8"))
    assert not violations, (
        f"{module_path.name} uses forbidden claim phrase(s) without negation: {violations}"
    )


@pytest.mark.parametrize("module_path", STAGE8C_MODULES, ids=lambda p: p.name)
def test_module_does_not_assign_forbidden_status(module_path):
    text = module_path.read_text(encoding="utf-8").lower()
    for status in FORBIDDEN_STATUSES:
        assert status not in text, (
            f"{module_path.name} references forbidden status '{status}'."
        )


def test_modules_declare_fluence_or_optical_status():
    """At least one Stage 8C module must declare the allowed model status."""
    blob = "\n".join(m.read_text(encoding="utf-8") for m in STAGE8C_MODULES)
    assert "fluence_prediction" in blob


# ---------------------------------------------------------------------------
# Optional figure metadata (only if the example PNG exists)
# ---------------------------------------------------------------------------


def test_example_figure_metadata_if_present():
    if not EXAMPLE_FIGURE.is_file():
        pytest.skip("Stage 8C example figure not generated (PNG is gitignored).")
    from PIL import Image
    with Image.open(EXAMPLE_FIGURE) as img:
        meta = dict(getattr(img, "text", {}) or {})
    assert meta.get("final_export_allowed") == "False", (
        f"Example figure must be stamped final_export_allowed=False; got {meta!r}"
    )
    assert meta.get("model_status") == "fluence_prediction"


def test_example_csv_status_if_present():
    if not EXAMPLE_CSV.is_file():
        pytest.skip("Stage 8C example CSV not generated.")
    text = EXAMPLE_CSV.read_text(encoding="utf-8").lower()
    assert "fluence_prediction" in text
    assert "final_export_allowed" in text


# ---------------------------------------------------------------------------
# Figure builder stamps governance metadata in source
# ---------------------------------------------------------------------------


def test_figure_builder_stamps_final_export_false():
    text = (ROOT / "vbb_study" / "digital_twin" / "field_figures.py").read_text(encoding="utf-8")
    assert 'final_export_allowed' in text
    assert '"False"' in text or "'False'" in text
    assert "diagnostic_allowed" in text


# ---------------------------------------------------------------------------
# Lock-sensitive files untouched
# ---------------------------------------------------------------------------


def test_lock_sensitive_files_not_modified():
    """Stage 8C must not modify core physics or the characterisation lock."""
    lock_sensitive = [
        "Publication_Study/bessel_twin_core.py",
        "Publication_Study/tests/test_characterisation_lock.py",
        "Publication_Study/vbb_study/equations/propagation.py",
        "Publication_Study/vbb_study/equations/scalar_bessel.py",
    ]
    repo_root = ROOT.parent  # C:/PhD/Code
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root), capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        pytest.skip("git not available for lock-sensitive file check.")
    if out.returncode != 0:
        pytest.skip("git status failed; cannot check lock-sensitive files.")

    modified = set()
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        # porcelain format: XY <path>  (path starts at column 3)
        path = line[3:].strip().strip('"')
        modified.add(path.replace("\\", "/"))

    for f in lock_sensitive:
        assert f not in modified, f"Lock-sensitive file modified by working tree: {f}"
