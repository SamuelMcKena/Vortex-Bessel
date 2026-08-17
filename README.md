# Vortex-Bessel

Clean, consolidated research code for numerical and experimental structured-beam work.

This repository is intended to be the **day-to-day source of truth**: one current package, one organised notebook tree and one normal runner, rather than searching through dated Publication_Study backups and old phase branches for whichever implementation was newest.

It is built from the Phase 2K mathematical/physics-audited codebase and includes the current measured q=20 axicon-aberration correction work.

## Start here

Install the audited numerical environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-report.txt
```

See the current runnable study map:

```powershell
python run_study.py --list
```

Run a topic directly:

```powershell
python run_study.py --stage scalar
python run_study.py --stage lab_realism
python run_study.py --stage vector
python run_study.py --stage materials
python run_study.py --stage advanced
python run_study.py --stage digital_twin
```

Run every numerical/model stage:

```powershell
python run_study.py --stage all
```

The measured experimental stage is intentionally separate because the raw camera/BeamGage data are local:

```powershell
$env:BESSEL_ZSCAN_DATA_DIR = 'D:\path\to\z-scan'
python run_study.py --stage experimental
```

Executed notebooks are written under `outputs/executed_notebooks/`; the source notebooks are left untouched.

See **`docs/RUN_GUIDE.md`** for the full run instructions and **`docs/PUBLICATION_STUDY_MAP.md`** for a direct map from the old numbered Publication_Study notebooks and `NB_*` workflows to their current locations.

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
- `tools/` — validation, reproducibility and evidence utilities.
- `calibration/` and `configs/` — bench-calibration contracts and study configuration.
- `outputs/validation/` — governed machine-readable validation evidence retained from the audited codebase.

Compatibility modules such as `bessel_twin_core.py`, `publication_diagnostics.py`, `interface_correction_diagnosis.py` and `run_publication_study.py` remain so older notebooks/scripts can still resolve their established imports, but new work should start from `vbb_study/` and `run_study.py`.

## Publication_Study cleanup

The useful Publication_Study functionality has been carried forward, but the duplicate `vbb_study - Copy`, dated `backups/`, Python caches, `run_* - Copy.py` files and bulk historical generated-output tree are not treated as competing current implementations.

The original numbered workflows are mapped here:

`docs/PUBLICATION_STUDY_MAP.md`

The historical `phd-structured-beam-study` repository should be kept as the provenance/history store; routine development and analysis should happen here.

## Scientific claim boundary

The repository separates analytic/numerical reference calculations, nominal optical-system prediction, measured experimental evidence and hardware-validated correction.

For the q=20 aberration work, the measured z-stack is experimental evidence, while the recovered modal phase and proposed SLM correction remain model inference.  Files marked `UNCALIBRATED_DO_NOT_APPLY` or `NOMINAL_PREVIEW_NOT_FOR_DISPLAY` are not hardware-ready until the SLM LUT/phase stroke, illuminated footprint, coordinate parity/rotation and camera-to-SLM transform are calibrated and a fresh post-correction z-scan verifies the result.

## Checks

```powershell
python -m compileall -q vbb_study tools tests
python -m pytest tests -q
python run_study.py --list
python run_study.py --stage scalar --dry-run
```

See `CLEANROOM_PROVENANCE.md` for the source branches and migration/claim-boundary decisions.