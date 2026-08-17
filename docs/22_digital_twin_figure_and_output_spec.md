# Digital Twin — Figure and Output Specification

**Stage:** 8A  
**Last updated:** 2026-06-18  
**Status:** Blueprint (documentation only)

This document specifies the 16 required figures and animated GIF that the full digital twin
pipeline must produce, along with output format requirements and interactive control parameters.

---

## Output Directories

| Type | Directory |
|---|---|
| Figures | `outputs/figures/digital_twin/` |
| CSVs | `outputs/csv/digital_twin/` |
| Holograms | `outputs/holograms/` (existing, unchanged) |

All figure files are PNG, 300 DPI, unless otherwise noted.  The animated GIF is an exception.

---

## Caption Requirements

Every figure caption must state:
- Preset / N / device_downsample (e.g., "balanced preset, N=1024, ds=2")
- Pulse energy (e.g., "E_p = 1.0 µJ")
- Model status (e.g., "[fluence_prediction]")
- Any proxy warnings required by the status (see `docs/21_...`)

---

## Figure Registry

### FIG-01: `dt_fig01_slm_pattern.png`

**Title:** SLM hologram pattern (Engine 1 input)  
**Engine:** 1  
**Model status:** `optical_prediction`  
**Content:** Greyscale SLM phase pattern, showing blaze grating + vortex phase.  Blaze period Λ
(px) and topological charge ℓ in title.  
**Size:** Single panel, square, SLM grid pixels.  
**Caption must include:** preset, N, Λ, ℓ, `[optical_prediction]`.

---

### FIG-02: `dt_fig02_4f_first_order_filter.png`

**Title:** 4f relay — first-order filter diagram (holographic route only)  
**Engine:** 1  
**Model status:** `optical_prediction`  
**Content:** Two panels: (a) Fourier plane intensity showing DC, +1, −1 orders and filter aperture;
(b) filtered first-order field at relay output.  Order separation and filter pass fraction in title.  
**Caption must include:** blaze period Λ, filter radius (px), first-order selected fraction η,
`[optical_prediction]`.

---

### FIG-03: `dt_fig03_propagation_axial.png`

**Title:** Beam propagation — on-axis intensity vs z  
**Engine:** 1  
**Model status:** `optical_prediction`  
**Content:** On-axis intensity I(0, z) vs z from SLM to beyond the focal plane.  Bessel zone and
strict region boundaries marked.  
**Caption must include:** preset, N, Bessel zone length, `[optical_prediction]`.

---

### FIG-04: `dt_fig04_surface_field_intensity.png`

**Title:** Transverse intensity at sample surface  
**Engine:** 1  
**Model status:** `optical_prediction`  
**Content:** 4-panel linked field view at surface z: |U|², domain-coloured phase, x-z longitudinal
slice, radial profile + J_ℓ overlay.  (Reuses `linked_field_views` from `viz_fields.py`.)  
**Caption must include:** preset, N, ds, design ℓ, measured winding (charge label), `[optical_prediction]`.

---

### FIG-05: `dt_fig05_fluence_map_transverse.png`

**Title:** Spatial fluence distribution at sample surface — F(x,y)  
**Engine:** 2  
**Model status:** `fluence_prediction`  
**Content:** 2D colour map of F(x,y) [J/cm²] at the surface plane.  Colour scale: linear or
gamma=0.45 (state in caption).  Contours at F_th_lit and 0.5×F_th_lit if threshold supplied.  
**Caption must include:** E_p [µJ], peak fluence [J/cm²], colour scale mode, `[fluence_prediction]`.

---

### FIG-06: `dt_fig06_fluence_map_xz.png`

**Title:** Longitudinal fluence map — F(x, z)  
**Engine:** 2  
**Model status:** `fluence_prediction`  
**Content:** x-z slice of F(x, z) from BL-ASM propagation cube.  Bessel zone extent and core
diameter marked.  
**Caption must include:** E_p [µJ], colour scale, Bessel zone, `[fluence_prediction]`.

---

### FIG-07: `dt_fig07_threshold_crossing_map.png`

**Title:** Threshold-crossing map — F(x,y) vs F_th (literature)  
**Engine:** 2  
**Model status:** `fluence_threshold_proxy`  
**Content:** Binary or continuous map: regions where F(x,y) ≥ F_th_lit.  Threshold value and
source (literature reference) stated.  Proxy warning stamp in orange.  
**Caption must include:** E_p [µJ], F_th_lit [J/cm²] and source, `[fluence_threshold_proxy]`,
"PLANNING PROXY".

---

### FIG-08: `dt_fig08_incubation_curve.png`

**Title:** Incubation threshold curve — F_th(N) vs N  
**Engine:** 2  
**Model status:** `fluence_threshold_proxy`  
**Content:** F_th(N) = F_th(1) × N^(S−1) plotted for N = 1 to N_max.  Operating point (E_p,
shots_per_site) marked.  Literature S and F_th(1) values stated with source.  
**Caption must include:** F_th(1) [J/cm²], S (incubation exponent), source, `[fluence_threshold_proxy]`.

---

### FIG-09: `dt_fig09_writing_trajectory.png`

**Title:** Writing trajectory diagram  
**Engine:** 2  
**Model status:** `dose_accumulation_proxy`  
**Content:** Plan view (x-y) of scan path with beam spot overlay showing shot overlap.  Shot count
N per site, scan speed v, overlap fraction η shown.  
**Caption must include:** v [mm/s], f_rep [kHz], N, η, `[dose_accumulation_proxy]`, "GEOMETRIC PROXY".

---

### FIG-10: `dt_fig10_dose_accumulation_map.png`

**Title:** Accumulated dose map — multi-shot writing  
**Engine:** 2  
**Model status:** `dose_accumulation_proxy`  
**Content:** 2D map of cumulative fluence ΣF(x,y) over all shots in trajectory.  Normalised to
F_th_lit for spatial reference.  Proxy warning stamp.  
**Caption must include:** N_shots, E_p [µJ], ΣF_peak [J/cm²], `[dose_accumulation_proxy]`, "GEOMETRIC PROXY".

---

### FIG-11: `dt_fig11_modification_volume_proxy.png`

**Title:** Modification volume proxy — threshold-crossing extent  
**Engine:** 3  
**Model status:** `uncalibrated_material_response_proxy`  
**Content:** 3-panel: (a) x-y cross-section of modification proxy volume; (b) x-z cross-section;
(c) estimated feature width, depth, and length.  Proxy stamp in orange.  
**Caption must include:** F_th source, `[uncalibrated_material_response_proxy]`,
"PROXY — NOT A CALIBRATED PREDICTION".

---

### FIG-12: `dt_fig12_simulated_top_view.png`

**Title:** Simulated microscopy — top-view proxy  
**Engine:** 3  
**Model status:** `simulated_microscopy_proxy`  
**Content:** Simulated top-view image of modification proxy convolved with Gaussian PSF at the
microscope wavelength and NA.  PSF width stated.  Proxy stamp.  
**Caption must include:** PSF FWHM [µm], `[simulated_microscopy_proxy]`, "SIMULATED PROXY".

---

### FIG-13: `dt_fig13_ideal_vs_lab_fluence_comparison.png`

**Title:** Ideal vs lab-realistic fluence comparison  
**Engine:** 1 + 2  
**Model status:** `fluence_prediction`  
**Content:** Side-by-side fluence maps for ideal (no SLM encoding) and lab-realistic (holographic
SLM, first-order filter) routes.  Peak fluence ratio and core diameter for both.  
**Caption must include:** both routes, E_p [µJ], peak fluence [J/cm²] each, `[fluence_prediction]`.

---

### FIG-14: `dt_fig14_parameter_sensitivity.png`

**Title:** Parameter sensitivity — fluence vs E_p and blaze period  
**Engine:** 1 + 2  
**Model status:** `fluence_prediction`  
**Content:** Heat map or line plot of peak fluence vs pulse energy for multiple blaze periods Λ.
Operating point marked.  
**Caption must include:** sweep range, fixed parameters, `[fluence_prediction]`.

---

### FIG-15: `dt_fig15_hardware_route_comparison.png`

**Title:** Hardware route comparison — holographic vs physical axicon  
**Engine:** 1 + 2  
**Model status:** `fluence_prediction`  
**Content:** Side-by-side: (a) holographic route fluence map with charge label; (b) physical axicon
route fluence map with charge label (winding ≈ 0 for full-mode).  Efficiency and core diameter both.  
**Caption must include:** both routes, charge labels (measured winding), E_p, `[fluence_prediction]`,
F-A3p note for physical route.

---

### FIG-16: `dt_fig16_calibration_residuals.png`

**Title:** Calibration residuals (gated — requires calibration data)  
**Engine:** 3  
**Model status:** `calibrated_material_prediction`  
**Content:** Comparison of calibrated prediction to measured threshold crossings.  Residuals
(predicted minus measured) as function of fluence and shot count.  
**Notes:** This figure is NOT generated if no calibration data is present.  A placeholder panel
with "CALIBRATION DATA REQUIRED" is saved instead.  
**Caption must include:** calibration source, material, regime, `[calibrated_material_prediction]`.

---

### GIF-01: `dt_anim_writing_trajectory.gif`

**Title:** Animated writing trajectory — dose accumulation over time  
**Engine:** 2  
**Model status:** `dose_accumulation_proxy`  
**Content:** Animation showing dose map evolving as scan progresses.  One frame per N=10 shots.
Frame rate: 10 fps.  Total frames ≤ 50.  
**Format:** Animated GIF, ≤ 5 MB.  
**Caption must include:** N_total, v, f_rep, `[dose_accumulation_proxy]`, "GEOMETRIC PROXY".

---

## Output Format Requirements

| Property | Requirement |
|---|---|
| Figure DPI | 300 (bitmap), scalable for line art |
| Figure format | PNG (bitmap figures), GIF (animation) |
| Colour maps | `viridis` (intensity), `colorcet.CET_C2` (phase), `inferno` (fluence) |
| Proxy stamp | Orange text box: "PROXY — [status]", bottom-left corner |
| Caption format | `vbb_style.save_figure(fig, path, caption_text)` |
| CSV format | UTF-8, comma-separated, with `model_status` column |

---

## Interactive Controls

The following parameters must be exposed as interactive controls in the digital twin quicklook
notebook (Stage 8I):

| Control | Parameter | Range |
|---|---|---|
| Pulse energy | E_p [µJ] | 0.1 – 10.0 |
| Topological charge | ℓ | 0, 1, 2, 3, 4 |
| Blaze period | Λ [px] | 8, 12, 16, 20, 32 |
| Objective NA | NA | 0.3, 0.45, 0.6 |
| Target core diameter | 2r₀ [µm] | 1 – 10 |
| Target Bessel length | L_B [µm] | 50 – 500 |
| Scan speed | v [mm/s] | 0.1 – 100 |
| Rep rate | f_rep [kHz] | 1, 50, 100, 200 |
| Shots per site | N | 1 – 1000 |
| Hardware route | — | holographic, physical |
| Material | — | Cr:ZnSe (proxy), [calibrated] |
| Preset | — | fast, balanced |

Interactive controls run at `preset='fast'` only.  Publication-grade runs use the locked stage runner.

---

## Model Status Column in CSVs

All CSVs in `outputs/csv/digital_twin/` must include:

```
model_status, allowed_claims_summary, calibration_source, generated_at_utc
```

Where `calibration_source` is `"literature"` (proxy), `"experiment:[ref]"` (calibrated), or
`"none"` (optical-only output).
