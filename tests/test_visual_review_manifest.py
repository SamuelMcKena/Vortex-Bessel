from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_visual_review_manifest_exists_and_has_quicklook_figures():
    path = ROOT / "outputs" / "csv" / "review" / "visual_review_manifest.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert {"path", "figure_family", "status", "width_px", "height_px"}.issubset(df.columns)
    assert (df["figure_family"] == "quicklook_current").any()


def test_quicklook_contact_sheet_exists():
    path = ROOT / "outputs" / "figures" / "review" / "quicklook_visual_contact_sheet.png"
    assert path.exists()
