from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"


def test_non_quicklook_notebooks_expose_editable_controls():
    missing = []
    for path in NOTEBOOK_ROOT.rglob("*.ipynb"):
        if "quicklook" in path.parts:
            continue
        nb = nbformat.read(path, as_version=4)
        text = "\n".join(cell.source for cell in nb.cells)
        if "STAGE88: editable controls" not in text:
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, "missing editable controls in: " + ", ".join(missing)


def test_notebooks_do_not_reintroduce_person_specific_quicklook_labels():
    forbidden = ["richard_phase", "supervisor_facing", "supervisor_study"]
    offenders = []
    for path in NOTEBOOK_ROOT.rglob("*.ipynb"):
        nb = nbformat.read(path, as_version=4)
        text = "\n".join(cell.source for cell in nb.cells).lower()
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}:{token}")
    assert not offenders, ", ".join(offenders)
