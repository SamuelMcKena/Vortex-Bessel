"""Stage 8C.2 cockpit notebook polish tests.

Confirms the roadmap, explicit rendering (display/plt.show), the final
visual-check cell, and the preview-path wiring were added without breaking the
single-control-cell contract.
"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "digital_twin" / "00_full_beam_to_write_cockpit_MVP.ipynb"


def _load() -> dict:
    assert NB_PATH.is_file(), f"Notebook missing: {NB_PATH}"
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    nb = _load()
    parts = []
    for cell in nb["cells"]:
        src = cell.get("source", [])
        parts.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(parts)


def test_notebook_valid_and_compiles():
    nb = _load()
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])))


def test_first_code_cell_still_the_control_block():
    nb = _load()
    first_code = next(c for c in nb["cells"] if c.get("cell_type") == "code")
    src = "".join(first_code.get("source", []))
    assert "FULL BEAM-TO-WRITE COCKPIT MVP - USER CONTROLS" in src


def test_roadmap_section_present():
    src = _source().lower()
    assert "notebook roadmap" in src
    assert "set controls" in src
    assert "inspect the dashboard" in src


def test_explicit_rendering_calls_present():
    src = _source()
    assert "display(fig)" in src
    assert "plt.show()" in src


def test_final_visual_check_cell_present():
    src = _source()
    assert "FINAL VISUAL CHECK" in src
    # The visual check asserts the figure exists and is diagnostic-only.
    assert "fig.stage8c2_metadata" in src
    assert 'assert "fig" in globals()' in src


def test_preview_png_path_wired():
    src = _source()
    assert "stage8c2_integrated_cockpit_dashboard_preview.png" in src


def test_overall_status_surfaced_in_notebook():
    src = _source()
    assert "overall_status" in src


def test_demo_field_not_saved_as_preview():
    # The visual-check save must exclude unit_test_or_demo_only fields.
    src = _source()
    assert 'stack.source_status != "unit_test_or_demo_only"' in src


def test_runtime_check_cell_present():
    """The notebook must guard against stale imports / wrong path / blank figures."""
    src = _source()
    assert "RUNTIME CHECK" in src
    assert "autoreload" in src
    assert 'run_line_magic("matplotlib", "inline")' in src
    # Prints the resolved module path so the user can confirm the right code is loaded.
    assert "cockpit   :" in src
    assert "cd.__file__" in src or "Path(cd.__file__)" in src
    # Fails loudly rather than silently rendering the old dashboard.
    assert "Stale cockpit_dashboard import" in src


def test_dashboard_cell_dedupes_inline_render():
    """display(fig) + plt.close(fig) keeps exactly one inline image (no Agg blank / dup)."""
    src = _source()
    assert "plt.close(fig)" in src


def test_single_dashboard_creating_cell():
    """Exactly one cell should build the dashboard figure (no stale duplicate)."""
    nb = _load()
    n = sum(
        1 for c in nb["cells"]
        if c.get("cell_type") == "code"
        and "plot_integrated_cockpit_dashboard(" in "".join(c.get("source", []))
        and "fig =" in "".join(c.get("source", []))
    )
    assert n == 1, f"expected exactly one dashboard-creating cell, found {n}"
