# Running the Vortex-Bessel codebase

The purpose of this repository is to remove the old question of **which script or notebook is the newest one**.  Start from the repository root and use `run_study.py` as the normal entry point.

## 1. Create the environment

Python 3.13 is the audited interpreter family for the current numerical code.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-report.txt
```

For the measured axicon-aberration package, also install its local requirements:

```powershell
python -m pip install -r notebooks\experimental\axicon_aberration_correction\requirements.txt
```

## 2. See everything that is runnable

```powershell
python run_study.py --list
```

The runner discovers the notebooks that actually exist in the repository.  It does not keep a stale historical list.

## 3. Main numerical workflows

```powershell
python run_study.py --stage scalar
python run_study.py --stage lab_realism
python run_study.py --stage vector
python run_study.py --stage materials
python run_study.py --stage advanced
python run_study.py --stage digital_twin
```

Run all model/numerical stages in canonical order with:

```powershell
python run_study.py --stage all
```

`all` deliberately excludes the measured experimental stage because the raw camera/BeamGage acquisition data are local rather than committed to Git.

Use a dry run first when desired:

```powershell
python run_study.py --stage lab_realism --dry-run
```

Executed notebooks are written beneath `outputs/executed_notebooks/`; source notebooks are not overwritten.

## 4. Run one notebook

```powershell
python run_study.py --only notebooks/scalar/04_scalar_parameter_sweeps.ipynb
```

## 5. Axicon aberration correction from measured z-scans

The current measured correction workflow is:

`notebooks/experimental/axicon_aberration_correction/Bessel_zscan_alignment_correction.ipynb`

Set the local acquisition path first:

```powershell
$env:BESSEL_ZSCAN_DATA_DIR = 'D:\path\to\z-scan'
python run_study.py --stage experimental
```

The measured z-stack is experimental evidence.  The retrieved modal phase/correction is a model inference.  Any SLM mask marked `UNCALIBRATED_DO_NOT_APPLY` or `NOMINAL_PREVIEW_NOT_FOR_DISPLAY` is not a hardware command until the SLM LUT/phase stroke, illuminated footprint, parity/rotation and camera-to-SLM transform are calibrated and a new post-correction z-scan verifies the result.

## 6. Validation and tests

Run the code/physics regression suite with:

```powershell
python -m compileall -q vbb_study tools tests
python -m pytest tests -q
```

For a faster smoke check:

```powershell
python run_study.py --list
python run_study.py --stage scalar --dry-run
python run_study.py --stage lab_realism --dry-run
```

## 7. Where old Publication_Study work went

See `docs/PUBLICATION_STUDY_MAP.md`.  It maps the original numbered publication notebooks and the old `NB_*` notebooks onto the current topic-organised notebooks and code modules.

The old repository remains useful for provenance/history, but routine work should start here.