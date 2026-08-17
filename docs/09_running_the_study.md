# Running the Study

This document explains how to install, smoke-test, run individual stages,
regenerate outputs, and debug common problems.

---

## 1. Environment Setup

### Preferred Python environment

The study was developed with the lab virtual environment at
`C:\PhD\.venv2`.  To recreate from scratch:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Or with the existing lab environment:

```powershell
C:\PhD\.venv2\Scripts\python.exe -m pip install -r requirements.txt
```

### Required packages

See `requirements.txt` at the repo root.  Key dependencies:
`numpy`, `scipy`, `matplotlib`, `pandas`, `Pillow`, `jupyter`,
`nbconvert`, `nbformat`.

---

## 2. Smoke Test (quick sanity check)

Before running the full study, verify that imports, paths, canonical docs,
and the notebook list all pass:

```powershell
python tools\smoke_test_publication_study.py
```

or (inside the `Publication_Study` folder):

```powershell
python tools\smoke_test_study.py
```

The smoke test checks:
- Core imports (`bessel_twin_core`, `vbb_study` and its submodules).
- `setup_study.validate_workspace()` passes.
- All REQUIRED_DOCS exist in `docs/`.
- All REQUIRED_NOTEBOOKS exist in their new subdirectory locations.
- REQUIRED_SOURCE_FILES exist.

It does **not** execute the notebooks.

---

## 3. List the Canonical Notebook Order

```powershell
python Publication_Study\run_study.py --list
```

To list by stage:

```powershell
python Publication_Study\run_study.py --list --stage scalar
python Publication_Study\run_study.py --list --stage lab_realism
python Publication_Study\run_study.py --list --stage vector
python Publication_Study\run_study.py --list --stage materials
python Publication_Study\run_study.py --list --stage advanced
python Publication_Study\run_study.py --list --stage publication_exports
```

---

## 4. Dry Run (preview without executing)

```powershell
python Publication_Study\run_study.py --dry-run
python Publication_Study\run_study.py --dry-run --stage scalar
```

---

## 5. Run the Full Study

```powershell
python Publication_Study\run_study.py --timeout-s 1800
```

---

## 6. Run a Single Stage

```powershell
python Publication_Study\run_study.py --stage scalar
python Publication_Study\run_study.py --stage lab_realism
python Publication_Study\run_study.py --stage vector
python Publication_Study\run_study.py --stage materials
python Publication_Study\run_study.py --stage advanced
python Publication_Study\run_study.py --stage publication_exports
```

---

## 7. Run a Single Notebook

```powershell
python Publication_Study\run_study.py --only notebooks\scalar\04_scalar_parameter_sweeps.ipynb
```

---

## 8. Run a Slice

```powershell
python Publication_Study\run_study.py `
    --start-at notebooks\scalar\02_scalar_ideal_vs_lab_diagnostics.ipynb `
    --stop-after notebooks\scalar\05_scalar_validation_suite.ipynb
```

---

## 9. Clean Output Before a Run

Preview what would be cleaned:

```powershell
python Publication_Study\run_study.py --clean-output figures csv manifests --dry-run
```

Clean and run:

```powershell
python Publication_Study\run_study.py --clean-output figures csv manifests --stage scalar
```

With no folder names, cleans all default folders (figures, csv, holograms,
manifests, jupyter_runtime):

```powershell
python Publication_Study\run_study.py --clean-output
```

Scientific outputs are **never deleted** unless `--clean-output` is used.

---

## 10. Debug a Failed Notebook

Continue after failure instead of stopping:

```powershell
python Publication_Study\run_study.py --continue-on-error --stage scalar
```

Skip finalization (faster turnaround):

```powershell
python Publication_Study\run_study.py --no-finalize --only notebooks\scalar\04_scalar_parameter_sweeps.ipynb
```

---

## 11. Output Locations

All generated files live under `Publication_Study/outputs/`:

| Subfolder | Contents |
|---|---|
| `outputs/figures/` | PNG/SVG/PDF plots |
| `outputs/csv/` | Summary tables |
| `outputs/holograms/` | SLM phase pattern exports |
| `outputs/manifests/` | Run start/finish JSON manifests |
| `outputs/jupyter_runtime/` | Jupyter kernel runtime files (auto-generated) |

---

## 12. Regenerate Outputs Only (no notebooks)

To collect all current artifacts into manifests without running notebooks:

```powershell
python Publication_Study\finalize_outputs.py
```

---

## 13. Backward-Compatible Runner

The original runner name still works:

```powershell
python Publication_Study\run_publication_study.py --list
python Publication_Study\run_publication_study.py --dry-run
python Publication_Study\run_publication_study.py --stage scalar
```

`run_publication_study.py` is now a thin wrapper that delegates to
`run_study.py`.

---

## 14. Common Failure Modes

| Symptom | Fix |
|---|---|
| `FileNotFoundError: workspace validation failed` | Run smoke test; check REQUIRED_NOTEBOOKS paths match actual file locations |
| `Unknown notebook: X` in `--only` or `--start-at` | Run `--list` to see canonical notebook paths; note they now include subdirectory prefix |
| Jupyter runtime permission errors | Runner sets `JUPYTER_RUNTIME_DIR` automatically; ensure `outputs/jupyter_runtime/` is writable |
| Missing packages | Reinstall: `pip install -r requirements.txt` |
| Stale outputs | Use `--clean-output` before re-running the affected stage |
| `Cannot find repo root` | Run the runner from inside the repo; `find_repo_root` searches upward for `Publication_Study/bessel_twin_core.py` |
