"""Stage 4 notebook triage.

Runs all 22 curated notebooks via papermill from a foreign CWD (C:\PhD\Code),
NOT from Publication_Study/. Records PASS / FAIL / TIMEOUT, wall-clock time,
failing cell, exception type, first line of traceback, and failure category.

Results written to:
  Publication_Study/outputs/notebook_triage/triage_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import papermill as pm

STUDY_ROOT = Path("C:/PhD/Code/Publication_Study")
KERNEL_CWD = "C:/PhD/Code"  # foreign -- NOT Publication_Study/
OUTPUT_DIR = STUDY_ROOT / "outputs" / "notebook_triage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PER_NB_TIMEOUT = 300  # seconds per notebook

# Canonical list — matches the 22 curated notebooks (no archive/, no outputs/).
NOTEBOOKS: list[str] = [
    "notebooks/00_study_overview_and_conventions.ipynb",
    "notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb",
    "notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb",
    "notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb",
    "notebooks/scalar/04_scalar_parameter_sweeps.ipynb",
    "notebooks/scalar/05_scalar_validation_suite.ipynb",
    "notebooks/lab_realism/01_holographic_axicon_route.ipynb",
    "notebooks/lab_realism/02_physical_axicon_route.ipynb",
    "notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb",
    "notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb",
    "notebooks/lab_realism/05_through_sample_interface.ipynb",
    "notebooks/lab_realism/06_full_source_to_sample_journey.ipynb",
    "notebooks/vector/01_vector_beam_theory_atlas.ipynb",
    "notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb",
    "notebooks/vector/03_vector_hardware_routes.ipynb",
    "notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb",
    "notebooks/materials/02_material_calibration_template.ipynb",
    "notebooks/materials/03_application_design_tables.ipynb",
    "notebooks/publication_exports/03_report_export.ipynb",
    "notebooks/advanced/01_capsule_weld_feature_design.ipynb",
    "notebooks/advanced/02_hexagonal_polygonal_beams.ipynb",
    "notebooks/advanced/03_discrete_nfold_beams.ipynb",
]


def _categorise(ename: str | None, evalue: str) -> str:
    """Map (ename, evalue) → one of the four triage categories."""
    if not ename:
        return "runtime-bug"
    ename = str(ename)
    evalue = str(evalue or "")
    # path: import of vbb_study or bessel_twin_core fails — should be zero post-install
    if ename in ("ModuleNotFoundError", "ImportError") and any(
        p in evalue for p in ("vbb_study", "bessel_twin_core")
    ):
        return "path"
    # missing-symbol: import of any other name that doesn't exist
    if ename in ("ModuleNotFoundError", "ImportError", "AttributeError", "NameError"):
        return "missing-symbol"
    # data-dependency: a file on disk is missing
    if ename in ("FileNotFoundError", "OSError", "PermissionError") or any(
        kw in evalue for kw in ("No such file", "cannot find", "does not exist")
    ):
        return "data-dependency"
    return "runtime-bug"


def _run_one(nb_rel: str) -> dict:
    nb_path = STUDY_ROOT / nb_rel
    out_name = Path(nb_rel).name.replace(".ipynb", "_executed.ipynb")
    out_path = OUTPUT_DIR / out_name

    t0 = time.time()
    status = "FAIL"
    fail_cell: int | None = None
    exc_type: str | None = None
    exc_line: str | None = None
    category: str | None = None

    try:
        pm.execute_notebook(
            str(nb_path),
            str(out_path),
            kernel_name="python3",
            cwd=KERNEL_CWD,
            execution_timeout=PER_NB_TIMEOUT,
        )
        status = "PASS"

    except pm.exceptions.PapermillExecutionError as e:
        fail_cell = getattr(e, "exec_count", None)
        exc_type = str(getattr(e, "ename", "PapermillExecutionError"))
        exc_line = str(getattr(e, "evalue", str(e)))[:300]
        category = _categorise(exc_type, exc_line)

    except Exception as e:
        exc_type = type(e).__name__
        exc_line = str(e)[:300]
        if "Timeout" in exc_type or "timeout" in exc_line.lower():
            status = "TIMEOUT"
            category = "runtime-bug"
        else:
            category = _categorise(exc_type, exc_line)

    elapsed = round(time.time() - t0, 1)
    return {
        "notebook": nb_rel,
        "status": status,
        "wall_s": elapsed,
        "fail_cell": fail_cell,
        "exc_type": exc_type,
        "exc_line": exc_line,
        "category": category,
    }


def main() -> None:
    results: list[dict] = []
    total_start = time.time()

    for nb_rel in NOTEBOOKS:
        print(f"\n{'='*70}", flush=True)
        print(f"Running: {nb_rel}", flush=True)
        r = _run_one(nb_rel)
        results.append(r)

        tag = r["status"]
        print(f"  [{tag}]  {r['wall_s']:.1f}s", flush=True)
        if r["exc_type"]:
            print(f"  cell {r['fail_cell']} | {r['exc_type']}: {(r['exc_line'] or '')[:120]}", flush=True)
        if r["category"]:
            print(f"  category: {r['category']}", flush=True)

    total_elapsed = round(time.time() - total_start, 1)
    print(f"\n{'='*70}", flush=True)
    print(f"Triage complete — {len(results)} notebooks in {total_elapsed:.1f}s", flush=True)

    # Category summary
    from collections import Counter
    cats = Counter(r["category"] for r in results if r["category"])
    statuses = Counter(r["status"] for r in results)
    print(f"Statuses: {dict(statuses)}", flush=True)
    print(f"Categories: {dict(cats)}", flush=True)

    path_fails = [r for r in results if r["category"] == "path"]
    if path_fails:
        print(f"\n!!! PATH FAILURES DETECTED ({len(path_fails)}) — editable install may be broken !!!", flush=True)
        for r in path_fails:
            print(f"  {r['notebook']}: {r['exc_line']}", flush=True)
    else:
        print("\nOK: zero path-category failures (editable install is working).", flush=True)

    json_path = OUTPUT_DIR / "triage_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
