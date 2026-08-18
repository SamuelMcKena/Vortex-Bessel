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
- `q20_modal_analysis.py`: all-plane q=20 fit and legacy/diagnostic phase products.
- `q20_phase_physics.py`: authoritative z-to-annulus physics and residual-phase assembly. It treats the z-stack as radial diversity for **one transverse phase**, removes the target q=20 vortex from the residual, fixes only the unobservable annular piston gauge, and blocks hardware-ready claims.
- `rebuild_q20_presentation_from_bmg.py`: rebuilds the current q=20 presentation evidence directly from the complete 18-plane × 4-repeat BMG acquisition and generates measured-only XZ/YZ, the physics-safe residual-phase diagnostic, and a single-transverse-phase propagation comparison.
- `phase_error_recreation.py`: per-plane inverse-error recreation test.
- `single_mask_inverse_forward_test.py`: stricter single-input-mask propagation test.
- `comprehensive_error_validation.py`: high-resolution slices, XZ/YZ maps, 1D/radial/angular profiles, metrics, and 3D meshes.
- `measured_beam_path_trajectory.py`: absolute full-sensor beam-centre trajectory.
- `slm2_complete_mask_preview.py`: native 1920×1080 correction-only or composed SLM2 phase previews.
- `iterative_correction_controller.py`: held-out-plane correction proposal controller.

## Physics-safe presentation rebuild

For presentation figures derived from the experimental q=20 scan, use:

```powershell
$env:BESSEL_ZSCAN_DATA_DIR = 'D:\path\to\z-scan 2 1010'
python rebuild_q20_presentation_from_bmg.py
```

The rebuild deliberately fails unless it finds the complete acquisition expected
for this dataset: **18 z planes and 4 BMG repeats per plane (72 raw BMG files)**.
It writes:

- `01_measured_q20_BMG_stack_all_planes.png`
- `02_measured_q20_XZ_all_planes.png`
- `03_measured_q20_YZ_all_planes.png`
- `04_measured_q20_XZ_YZ_combined_all_planes.png`
- `05_retrieved_residual_phase_physics.png`
- `06_single_transverse_phase_forward_model.png`
- `q20_presentation_rebuild_provenance.json`

The first four are **measured-data figures**. The final two are explicitly
**model-inferred diagnostics**.

### Phase interpretation lock

The q=20 z-scan does not imply separate transverse and longitudinal correction
masks. For an axicon/conical wave, each measured z plane samples a different
annulus of one transverse input field. The code uses the stationary-phase
mapping `rho_z = z tan(alpha)` (with the exact k-vector form for the angle) to
turn z diversity into radial annulus diversity.

The modal retrieval already factors out the programmed q=20 vortex. Therefore
`q*theta` must **not** be inserted into the residual aberration correction. Also,
independent intensity fits do not measure annulus-to-annulus global phase
(piston), so the new reconstruction makes that gauge ambiguity explicit instead
of converting it into apparent radial structure.

See `Q20_PHASE_RECONSTRUCTION_PHYSICS.md` for the derivation, scientific limits,
and presentation wording.

The older normalized z-order-to-radius phase-map rendering remains only as
legacy diagnostic evidence for backwards compatibility with existing validation
scripts. It is **not** the authoritative physical reconstruction and should not
be promoted as an SLM correction map.

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
