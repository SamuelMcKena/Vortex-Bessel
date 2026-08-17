# Digital Twin — Implementation Stage Plan (8A–8J)

**Stage:** 8A  
**Last updated:** 2026-06-18  
**Status:** Blueprint (documentation only)

This document defines the implementation stages for the full beam-to-write digital twin.
Each stage is independent and gated: a later stage is not started until the deliverables and
acceptance criteria for the earlier stage are met.

---

## Constraints Applying to All Stages

- The `bessel_twin_core` physics engine is NOT modified.
- The characterisation locks (`test_characterisation_lock.py`, `test_characterisation_lock_prod.py`)
  must remain green after every stage.
- No new numerical nonlinear physics before Stage 8F.
- No fake calibrated-material outputs — all calibration requires real experiment data.
- No engine or tolerance changes.
- Additive only: new files; existing notebooks and Python source untouched unless explicitly listed.

---

## Stage 8A — Blueprint Documentation (THIS STAGE)

**Goal:** Create master architecture documentation for the digital twin.

**Deliverables:**
- `docs/20_full_beam_to_write_digital_twin_blueprint.md`
- `docs/21_digital_twin_model_status_and_claim_boundaries.md`
- `docs/22_digital_twin_figure_and_output_spec.md`
- `docs/23_digital_twin_math_and_physics_stack.md`
- `docs/24_digital_twin_stage_plan.md` (this file)
- `STAGE8A_DIGITAL_TWIN_BLUEPRINT_SUMMARY.md`
- Optional: CSVs in `outputs/csv/digital_twin/`
- `tests/test_stage8a_blueprint_docs.py`

**Acceptance criteria:**
- All 5 docs exist in `docs/` with correct filenames.
- `test_stage8a_blueprint_docs.py` passes (all 11 model statuses present in doc 21;
  claim boundaries defined; stage plan includes all stages 8A–8J).
- Fast lock remains green.

**No engine changes, no new Python modules.**

---

## Stage 8B — Engine 1 Adapter Module

**Goal:** Create a thin `vbb_study/digital_twin/` package that wraps `bessel_twin_core.run_case`
as a typed, documented interface for the digital twin pipeline.

**Deliverables:**
- `vbb_study/digital_twin/__init__.py` — package init, exports
- `vbb_study/digital_twin/engine1.py` — typed wrapper around `bt.run_case`; returns
  `Engine1Result` dataclass with `surface_field`, `axicon_result`, `design`, `validity_report`,
  `model_status = "optical_prediction"`
- `tests/test_stage8b_engine1_adapter.py` — verifies Engine1Result structure, model_status field,
  fast and balanced presets

**Acceptance criteria:**
- `Engine1Result.model_status == "optical_prediction"`
- Output matches `bt.run_case` for the same config (pixel-level equality on field)
- Fast lock green

---

## Stage 8C — Engine 2: Exposure / Fluence / Writing

**Goal:** Implement Engine 2 (exposure, fluence, incubation, writing trajectory) as a typed module.

**Deliverables:**
- `vbb_study/digital_twin/exposure.py`
  - `compute_fluence(engine1_result, pulse_energy_J) → FluenceResult`
  - `compute_threshold_crossing(fluence_result, F_th_1shot, incubation_exponent, N) → ThresholdResult`
  - `compute_writing_trajectory(fluence_result, scan_speed_m_s, rep_rate_Hz, N_shots) → TrajectoryResult`
  - `compute_dose_accumulation(trajectory_result) → DoseResult`
- All result dataclasses carry `model_status` field set to appropriate status key.
- `tests/test_stage8c_exposure.py` — unit tests for each function; checks model_status, energy
  conservation (total fluence integral ≈ E_eff), threshold crossing area at known fluence

**Acceptance criteria:**
- Energy conservation check passes (drift < 1%)
- All model_status fields correct
- Fast lock green

---

## Stage 8D — Engine 3: Material Response Proxy

**Goal:** Implement Engine 3 proxy material model as a typed module (stubs for nonlinear/thermal
hooks; no numerical implementation yet).

**Deliverables:**
- `vbb_study/digital_twin/material.py`
  - `compute_modification_proxy(dose_result, F_th_N) → ModificationProxyResult`
    - `model_status = "uncalibrated_material_response_proxy"`
  - `compute_simulated_image(modification_proxy, psf_fwhm_m) → SimulatedImageResult`
    - `model_status = "simulated_microscopy_proxy"`
  - `_nonlinear_proxy_hook(fluence_map, M_order) → array` — stub, raises NotImplementedError
  - `_thermal_proxy_hook(fluence_sequence, tau_thermal) → array` — stub, raises NotImplementedError
  - `load_calibration(csv_path) → CalibrationData` — reads calibration CSV, validates provenance
  - `compute_calibrated_prediction(dose_result, calibration) → CalibratedResult`
    - `model_status = "calibrated_material_prediction"` — only if cal data present

**Acceptance criteria:**
- Default path produces `uncalibrated_material_response_proxy` status, not calibrated
- Calibrated path requires valid calibration CSV (no fake calibration)
- Stub hooks raise NotImplementedError when called
- Fast lock green

---

## Stage 8E — Figure Generation Pipeline

**Goal:** Implement the full 16-figure + GIF pipeline from the spec in `docs/22_...`.

**Deliverables:**
- `vbb_study/digital_twin/figures.py`
  - One function per figure: `render_fig01_slm_pattern`, ..., `render_fig16_calibration_residuals`
  - `render_all(outputs, out_dir, caption_cfg) → list[Path]` — calls all applicable figures
  - `render_animated_gif(trajectory_result, out_path) → Path`
- Figures saved to `outputs/figures/digital_twin/`
- Each figure has model status in caption and proxy stamp if required by status
- `tests/test_stage8e_figures.py` — smoke test: runs all figures at fast preset, checks files exist,
  checks proxy stamp present for proxy-status figures

**Acceptance criteria:**
- All 16 PNG files generated for a fast-preset run
- GIF generated, ≤ 5 MB
- Proxy figures carry orange stamp
- Fast lock green

---

## Stage 8F — Nonlinear and Thermal Proxy Implementation

**Goal:** Replace the stub nonlinear and thermal hooks with heuristic implementations.
These are explicitly NOT calibrated and must carry the appropriate model status.

**Deliverables:**
- `exposure.py` — implement `_nonlinear_proxy_hook` using I^(M-1) scaling
- `material.py` — implement `_thermal_proxy_hook` using exponential decay accumulation
- Configuration flag `enable_nonlinear_proxy: bool`, `enable_thermal_proxy: bool` — both default False
- Updated figures: when hooks enabled, proxy stamp updated to include hook name
- `tests/test_stage8f_proxy_hooks.py` — tests that hooks produce valid arrays, that model_status
  is not upgraded when hooks are enabled, that both hooks are disabled by default

**Acceptance criteria:**
- Hooks disabled by default, must be explicitly enabled
- Model status not upgraded by enabling hooks
- Fast lock green

---

## Stage 8G — Calibration Data Ingestion

**Goal:** Implement the calibration data loader and enable `calibrated_material_prediction` outputs
when real calibration data is present.

**Deliverables:**
- `vbb_study/digital_twin/calibration.py`
  - `CalibrationRecord` dataclass: `material`, `wavelength_nm`, `pulse_duration_fs`,
    `NA`, `F_th_1shot_J_cm2`, `incubation_exponent_S`, `calibration_source`, `measured_at_utc`
  - `load_calibration_csv(path) → CalibrationRecord` — validates all required fields
  - Schema validation: rejects `calibration_source = "synthetic"` or missing fields
- `calibration/` directory added with `calibration_template.csv` (not a calibrated prediction)
- Updated Engine 3 to use calibration record when present
- `tests/test_stage8g_calibration.py` — tests schema validation, rejects synthetic data,
  checks that calibrated path upgrades model_status correctly

**Acceptance criteria:**
- Synthetic / missing calibration data rejected
- `calibrated_material_prediction` status only when valid record present
- Fast lock green

---

## Stage 8H — Simulated Microscopy with PSF Calibration

**Goal:** Extend the simulated microscopy proxy to use a calibrated PSF from a beads measurement
when available, and validate the simulated image against a known target.

**Deliverables:**
- `material.py` — `load_psf_calibration(path) → PSFData`; replaces Gaussian PSF when present
- Updated `compute_simulated_image` to accept optional PSF data
- `tests/test_stage8h_simulated_image.py` — checks that Gaussian PSF is used by default,
  that calibrated PSF is used when provided, that image size matches field grid

**Acceptance criteria:**
- Default Gaussian PSF gives `simulated_microscopy_proxy` status
- PSF calibration upgrade does not change model status (image is still a proxy until full validation)
- Fast lock green

---

## Stage 8I — Quicklook Notebook and Interactive Controls

**Goal:** Create the interactive digital twin quicklook notebook with all interactive controls
from `docs/22_...`.

**Deliverables:**
- `notebooks/digital_twin/00_digital_twin_quicklook.ipynb`
  - Interactive controls for all parameters listed in `docs/22_...`
  - Shows FIG-04 (intensity), FIG-05 (fluence), FIG-07 (threshold crossing) at minimum
  - All outputs labelled with model status
  - Fast preset only; nothing saved by default
- `notebooks/digital_twin/01_digital_twin_full_run.ipynb`
  - Full 16-figure run at balanced or publication preset
  - Saves all figures to `outputs/figures/digital_twin/`
  - Runs Engine 1 → 2 → 3 in sequence

**Acceptance criteria:**
- Quicklook notebook runs clean at fast preset
- All model status labels visible in output cells
- Fast lock green

---

## Stage 8J — Integration Tests, Validation Record, Publication Exports

**Goal:** Final integration: end-to-end test of the full pipeline; validation record (if experiment
data available); publication-export gating.

**Deliverables:**
- `tests/test_stage8j_full_pipeline.py` — end-to-end test: fast preset, all three engines,
  all 16 figures generated, model status correct throughout
- `STAGE8J_VALIDATION_RECORD.md` — documents what has been experimentally validated (expected
  to be minimal at first; updated as experiments are run)
- Publication export gate: `digital_twin/figures.py::export_for_publication()` checks that
  all figures in the export set have `model_status` ≥ `fluence_prediction` and raises an error
  if any proxy-status figure is included without an explicit override flag

**Acceptance criteria:**
- End-to-end test passes at fast preset
- Validation record exists (may be empty — records what has NOT been validated)
- Publication export gate raises on proxy figures unless `allow_proxy_in_publication=True`
- Fast and prod characterisation locks both green

---

## Stage Plan Summary

| Stage | Name | New modules | Key test |
|---|---|---|---|
| 8A | Blueprint | docs only | test_stage8a_blueprint_docs |
| 8B | Engine 1 adapter | digital_twin/engine1.py | test_stage8b_engine1_adapter |
| 8C | Engine 2: exposure | digital_twin/exposure.py | test_stage8c_exposure |
| 8D | Engine 3: material proxy | digital_twin/material.py | test_stage8d_material_proxy |
| 8E | Figure pipeline | digital_twin/figures.py | test_stage8e_figures |
| 8F | Nonlinear/thermal hooks | (in exposure.py, material.py) | test_stage8f_proxy_hooks |
| 8G | Calibration ingestion | digital_twin/calibration.py | test_stage8g_calibration |
| 8H | PSF calibration | (in material.py) | test_stage8h_simulated_image |
| 8I | Quicklook notebook | notebooks/digital_twin/ | (manual run) |
| 8J | Integration + export gate | (in figures.py) | test_stage8j_full_pipeline |
