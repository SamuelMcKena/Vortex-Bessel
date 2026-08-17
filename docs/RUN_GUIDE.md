# Running the Vortex-Bessel codebase

The purpose of this repository is to remove the old question of **which script or notebook is the newest one**. Start from the repository root and use `vortex-bessel` / `run_study.py` as the normal entry point.

## 1. Create the environment

Python 3.13 is the audited interpreter family for the current numerical code.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-report.txt
python -m pip install -e . --no-deps
```

For the measured axicon-aberration package, also install its local requirements:

```powershell
python -m pip install -r notebooks\experimental\axicon_aberration_correction\requirements.txt
```

## 2. See everything that is runnable

```powershell
vortex-bessel --list
```

`python run_study.py --list` is equivalent. The runner discovers the notebooks that actually exist in the repository; it does not keep a stale historical notebook list.

## 3. Re-run the old Publication_Study cleanly

The maintained replacement for the original numbered `00`–`10` Publication_Study sequence is:

```powershell
vortex-bessel --stage publication
```

The historical entry-point name is retained too:

```powershell
python run_publication_study.py
```

Both execute the current replacement notebooks rather than resurrecting the old files. Some historical notebooks have been split into two clearer current notebooks, so the maintained publication suite contains more than eleven notebook files while covering the same workflow roles.

Use a dry run to inspect the exact current sequence:

```powershell
vortex-bessel --stage publication --dry-run
```

## 4. Main numerical workflows

```powershell
vortex-bessel --stage scalar
vortex-bessel --stage lab_realism
vortex-bessel --stage vector
vortex-bessel --stage materials
vortex-bessel --stage advanced
vortex-bessel --stage digital_twin
```

Run all model/numerical topic stages in canonical order with:

```powershell
vortex-bessel --stage all
```

`all` deliberately excludes the measured experimental stage because the raw camera/BeamGage acquisition data are local rather than committed to Git.

Executed notebooks are written beneath `outputs/executed_notebooks/`; source notebooks are not overwritten.

## 5. Run one notebook

```powershell
vortex-bessel --only notebooks/scalar/04_scalar_parameter_sweeps.ipynb
```

## 6. Axicon aberration correction from measured z-scans

The current measured correction workflow is:

`notebooks/experimental/axicon_aberration_correction/Bessel_zscan_alignment_correction.ipynb`

Set the local acquisition path first:

```powershell
$env:BESSEL_ZSCAN_DATA_DIR = 'D:\path\to\z-scan'
vortex-bessel --stage experimental
```

The measured z-stack is experimental evidence. The retrieved modal phase/correction is a model inference. Any SLM mask marked `UNCALIBRATED_DO_NOT_APPLY` or `NOMINAL_PREVIEW_NOT_FOR_DISPLAY` is not a hardware command until the SLM LUT/phase stroke, illuminated footprint, parity/rotation and camera-to-SLM transform are calibrated and a new post-correction z-scan verifies the result.

## 7. Validation and tests

Run the code/physics regression suite with:

```powershell
python -m compileall -q vbb_study tools tests
python tools\audit_publication_study_coverage.py
python tools\audit_clean_layout.py
python tools\audit_notebook_runtime_paths.py
python -m pytest tests -q
```

For a faster smoke check:

```powershell
vortex-bessel --list
vortex-bessel --stage publication --dry-run
vortex-bessel --stage scalar --dry-run
vortex-bessel --stage lab_realism --dry-run
```

## 8. Where old Publication_Study work went

- `docs/PUBLICATION_STUDY_MAP.md` maps the original numbered publication notebooks and old `NB_*` notebooks onto current topic-organised notebooks and modules.
- `docs/PUBLICATION_STUDY_COVERAGE.md` records what was retained and what was intentionally left as historical duplicate/provenance material.
- `docs/CURRENT_CODE_INDEX.md` is the fastest “I want to do X — which code do I run?” reference.

The old repository remains useful for provenance/history, but routine work should start here.