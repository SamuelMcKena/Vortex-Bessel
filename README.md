# Vortex-Bessel

Clean, consolidated research code for numerical and experimental structured-beam work.

This repository is intended to be the **day-to-day source of truth**: one current package, one organised notebook tree and one normal runner, rather than searching through dated Publication_Study backups and old phase branches for whichever implementation was newest.

It is built from the Phase 2K mathematical/physics-audited codebase and includes the current measured q=20 axicon-aberration correction work.

## Start here

Install the audited numerical environment and editable command-line entry point:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-report.txt
python -m pip install -e . --no-deps
```

See everything that is currently runnable:

```powershell
vortex-bessel --list
```

Run a topic directly:

```powershell
vortex-bessel --stage scalar
vortex-bessel --stage lab_realism
vortex-bessel --stage vector
vortex-bessel --stage materials
vortex-bessel --stage advanced
vortex-bessel --stage digital_twin
```

Run every numerical/model stage:

```powershell
vortex-bessel --stage all
```

The measured experimental stage is intentionally separate because the raw camera/BeamGage data are local:

```powershell
$env:BESSEL_ZSCAN_DATA_DIR = 'D:\path\to\z-scan'
vortex-bessel --stage experimental
```

`python run_study.py ...` remains equivalent if you do not want to install the editable command. Executed notebooks are written under `outputs/executed_notebooks/`; source notebooks are left untouched.

## Navigation

- `docs/CURRENT_CODE_INDEX.md` — practical “I want to do X — which code do I run?” index, including ideal beams, system errors, lateral axicon decentre, rounded tips, vector beams, materials, digital twin, q=20 correction and presentation figures.
- `docs/RUN_GUIDE.md` — environment and execution instructions.
- `docs/PUBLICATION_STUDY_MAP.md` — maps the old numbered Publication_Study and `NB_*` workflows to their current replacements.
- `docs/TOOLS_GUIDE.md` — specialist validation/error-study/figure utilities.
- `CLEANROOM_PROVENANCE.md` — source branches and migration/claim-boundary decisions.

## Repository layout

- `vbb_study/` — authoritative beam models, propagation, SLM/4F/axicon routes, calibration and digital-twin source.
- `notebooks/scalar/` — ideal and lab-realistic scalar Bessel/vortex-Bessel diagnostics, robustness, sweeps and validation.
- `notebooks/lab_realism/` — holographic and physical axicon routes, first-order filtering, sample interface and full optical journey.
- `notebooks/vector/` — vector/Jones-field and hardware-route studies.
- `notebooks/materials/` — material/application proxy calculations and calibration templates.
- `notebooks/advanced/` — capsule/weld, hexagonal/polygonal and discrete N-fold studies.
- `notebooks/digital_twin/` — later bench/digital-twin and structured-vector route work.
- `notebooks/experimental/axicon_aberration_correction/` — measured z-scan analysis, q=20 modal retrieval, inverse-error validation and correction proposal work.
- `tests/` — numerical, physics-contract and regression tests.
- `reference_kernels/` — independent numerical/reference implementations.
- `tools/` — specialist validation, reproducibility, error-study and figure utilities.
- `calibration/` and `configs/` — bench-calibration contracts and study configuration.
- `outputs/validation/` — governed machine-readable validation evidence retained from the audited codebase.

Compatibility modules such as `bessel_twin_core.py`, `publication_diagnostics.py`, `interface_correction_diagnosis.py` and `run_publication_study.py` remain so older notebooks/scripts can still resolve established imports, but new work should start from `vbb_study/` and `run_study.py`.

## Publication_Study cleanup

The useful Publication_Study functionality has been carried forward, but duplicate `vbb_study - Copy`, dated `backups/`, Python caches, `run_* - Copy.py` files and the bulk historical generated-output tree are not treated as competing current implementations.

If an old note names a previous notebook/script, use `docs/PUBLICATION_STUDY_MAP.md`. If a useful old calculation is genuinely missing here, migrate and test it here rather than resurrecting a backup as a second current implementation.

The historical `phd-structured-beam-study` repository remains the provenance/history store; routine development and analysis should happen here.

## Scientific claim boundary

The repository separates analytic/numerical reference calculations, nominal optical-system prediction, measured experimental evidence and hardware-validated correction.

For the q=20 aberration work, the measured z-stack is experimental evidence, while the recovered modal phase and proposed SLM correction remain model inference. Files marked `UNCALIBRATED_DO_NOT_APPLY` or `NOMINAL_PREVIEW_NOT_FOR_DISPLAY` are not hardware-ready until the SLM LUT/phase stroke, illuminated footprint, coordinate parity/rotation and camera-to-SLM transform are calibrated and a fresh post-correction z-scan verifies the result.

## Checks

```powershell
python -m compileall -q vbb_study tools tests
python tools\audit_clean_layout.py
python tools\audit_notebook_runtime_paths.py
python -m pytest tests -q
vortex-bessel --list
vortex-bessel --stage scalar --dry-run
```
