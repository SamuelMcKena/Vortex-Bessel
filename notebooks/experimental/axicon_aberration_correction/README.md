# Axicon vortex–Bessel z-scan correction

This directory contains the experimental BeamGage z-scan analysis, constrained
q=20 modal retrieval, inverse-error validation, SLM2 correction-mask preview,
high-resolution propagation/profile figures, and measured beam-axis diagnostics.

## Main entry point

Open `Bessel_zscan_alignment_correction.ipynb` in JupyterLab and run the cells in
order. The notebook defaults to a local directory named `z-scan 2 1010` beside
the notebook. To keep the raw acquisition elsewhere, set an environment variable
before starting Jupyter:

```powershell
$env:BESSEL_ZSCAN_DATA_DIR = 'D:\path\to\z-scan 2 1010'
jupyter lab
```

Install the Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Analysis modules

- `modal_vortex_bessel.py`: BeamGage loading and constrained vortex–Bessel modal fitting.
- `q20_modal_analysis.py`: all-plane q=20 fit and phase-only correction proposal.
- `phase_error_recreation.py`: per-plane inverse-error recreation test.
- `single_mask_inverse_forward_test.py`: stricter single-input-mask propagation test.
- `comprehensive_error_validation.py`: high-resolution slices, XZ/YZ maps, 1D/radial/angular profiles, metrics, and 3D meshes.
- `measured_beam_path_trajectory.py`: absolute full-sensor beam-centre trajectory.
- `slm2_complete_mask_preview.py`: native 1920×1080 correction-only or composed SLM2 phase previews.
- `iterative_correction_controller.py`: held-out-plane correction proposal controller.

## Data and outputs

Raw `.bmg` acquisitions and generated `outputs/` are intentionally excluded from
Git because they are large and reproducible. Lightweight derived metrics and the
current uncalibrated correction proposal are retained with the source.

### Curated figure set

The clean repository intentionally does **not** retain every historical rendering.
The current figure set is under:

`figures/current_q20/`

It contains the newest realigned q=20 outputs together with the comprehensive
all-z validation, phase-error-recreation, single-mask inverse-forward tests,
measured beam-axis diagnostics, closed-loop gain-selection results and current
SLM2 previews. Earlier `pre_realign` duplicates and the first root-level
post-correction figures were removed from the clean figure tree because newer
versions supersede them.

The correction mesh and SLM mask are **model predictions**, not post-correction
camera measurements. Files prefixed `UNCALIBRATED_DO_NOT_APPLY` must not be sent
to hardware until the SLM phase LUT, beam footprint, parity/rotation, and
camera-to-SLM transform have been calibrated and a new experimental z-scan has
passed validation.
