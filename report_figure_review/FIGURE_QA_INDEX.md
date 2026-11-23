# Figure-by-figure visual and scientific QA — 23 November 2026

**ALL PREVIOUS OUTPUTS UNAPPROVED.** Status describes the *previous* 34-figure package, not an accepted final product. Do not insert BLOCK figures in report. Many REVISE figures remain model-only and may be omitted entirely. All 34 were visually inspected as previews. Numeric/physics contracts were checked against available manifests and repository docs, not independently fully recalculated for every figure.

| Figure | Decision | Defect / required action |
|---|---|---|
| `fig2_1_gaussian_vs_bessel` | **BLOCK** | Gaussian invisible on 0–50 mm axis (zR≈0.047 mm); zoom focus and remove legend collision |
| `fig2_2_bessel_vortex_families` | **REVISE** | Small panels; confirm independent normalisations and wrapped phase meaning |
| `fig3_1_asm_validation_screening` | **BLOCK** | Radial numerical/theory disagreement; N=512 exceeds chosen 80% Nyquist guard |
| `fig3_2_ideal_beam_family` | **REVISE** | Analytic demonstration, not component-resolved or bench-validated model |
| `fig3_3_parameter_analytic_sweeps` | **REVISE** | Markers are evaluations of the same analytic formulas, not independent simulation |
| `fig3_4_pupil_geometry` | **REVISE** | Geometric pupil-only; missing requested wave-model axial ripple benchmark |
| `fig3_5_self_healing` | **REVISE** | Axial image thin/compressed; obstacle, shadow distance and similarity ROI need clarity |
| `fig3_6_vector_jones_fields` | **BLOCK** | 10-panel atlas illegible at 160 mm; split and enlarge |
| `fig4_0_axicon_angle_diagnostic` | **REVISE** | Nominal 10° prediction 45.54 µm lies outside quoted measured 46–50 µm interval; unresolved SKU |
| `fig4_1_component_phase_and_camera` | **BLOCK** | Text overlaps and Fourier orders are a toy illustrative spectrum |
| `fig4_2_field_through_4f_relay` | **REVISE** | Small maps; show phases; 31.4% stop transmission is nominal not measured |
| `fig4_3_ideal_vs_camera_integrated` | **BLOCK** | Overlapping row captions; not ideal-vs-bench comparison |
| `fig4_4_pointing_Snell_sweep` | **REVISE** | Only ray model, not delivered field; include near 0.75 mrad, iris clipping and wave propagation |
| `fig4_5_gaussian_decentre` | **REVISE** | Input Gaussian only, no propagated Bessel-error response |
| `fig4_6_slm2_vortex_registration` | **REVISE** | Phase masks only, does not demonstrate vortex singularity splitting |
| `fig4_7_fourier_iris_offset` | **REVISE** | Gaussian surrogate order, no pixelated SLM +1-order model or l dependence |
| `fig4_8_lens_decentre_phase` | **REVISE** | Thin-lens phase only, not relay-to-camera response |
| `fig4_9_ring_fit_vs_centroid` | **REVISE** | Synthetic test and busy labels; retain as illustration only if legible |
| `fig4_10_axicon_oblique_rays` | **REVISE** | Panel at 0.75 degree, requested reference case is 0.75 milliradian |
| `fig4_11_rounded_tip_sag` | **REVISE** | Sag curves only; no Brzobohaty axial-intensity benchmark |
| `fig4_12_angular_aberration_dictionary` | **BLOCK** | Tiny panel labels; 0.45/1.15 rad rather than requested 0.1/0.25 waves; no harmonics |
| `fig4_13_error_signature_matrix` | **BLOCK** | Ad-hoc synthetic metric responses not validated propagated error signature matrix |
| `fig4_14_synthetic_inverse_identifiability` | **BLOCK** | No multi-z intensity propagation or two-parameter phase retrieval |
| `fig5_2_measured_ring_and_uniformity` | **REVISE** | Only six of eighteen measured planes retained as main-ring candidates; flag selection and saturation |
| `fig5_3_ring_centre_beam_walk` | **BLOCK** | Disjoint quality-screened subsets and uncalibrated sensor axes; do not infer physical beam slope |
| `fig5_4_camera_pixel_integration` | **REVISE** | Pure camera simulation, NOT measured fourfold camera crop |
| `fig5_4b_sampling_nyquist` | **REVISE** | Differentiate asymptotic ring spacing from nonuniform first-ring period |
| `fig5_5_modal_phase_coefficients` | **REVISE** | Uncalibrated retrieval, not measured physical aberration or validated correction |
| `fig6_1_audited_gate_event_counts` | **BLOCK** | Includes invented schematic traces; genuine 11-Nov raw CH1/CH2 CSV exists |
| `fig6_2_audited_trigger_delay_summary` | **REVISE** | Replace summary with real 60 jitter captures, histogram and ECDF |
| `fig6_3_audited_photodiode_levels` | **REVISE** | Uncalibrated detector voltage, digitizer step and linearity caveats |
| `schematic_4f_fourier_plane` | **PASS_WITH_CAPTION** | Nominal f=300mm; L1-pinhole=300mm, pinhole-L2=300mm, 600mm lens separation |
| `schematic_laser_timing_signal_chain` | **PASS_WITH_CAPTION** | Signal-chain schematic only; CH2 detector transient not calibrated optical energy |
| `schematic_slm_roles` | **PASS_WITH_CAPTION** | SLM1 correction before SLM2; SLM2 chosen vortex/carrier; 300mm SLM gap provisional |

## Caption contract
Each final caption must say: *what panel shows; simulated vs measured vs retrieved; physical plane and coordinate frame; units and normalisation; algorithm/definition and acquisition selection; key fitted quantities with uncertainties; known uncalibrated variables; what the figure does NOT prove.* Original numbers cannot be re-used without source manifest and independent check.

## First-figure definitive bug
F2.1 Gaussian paraxial waist ~3.92 µm gives Rayleigh range ~0.0469 mm at λ=1029 nm. On 0–50 mm horizontal map it occupies ~0.1% of width; F2.1(a) is visually blank. Show a labelled focus zoom (e.g. −0.2 to +0.2 mm) and separately long-range Bessel zone or explicit inset. Do not merely increase brightness or hide different axial scales.

## Key source data already in repository
- `notebooks/experimental/axicon_aberration_correction/bessel_zscan_metrics.csv` — 18-plane measured q=20 derived ring data (main-ring candidate indices 6,7,8,14,15,16 require transparent selection).
- `notebooks/experimental/axicon_aberration_correction/UNCALIBRATED_DO_NOT_APPLY_aberration_coefficients.csv` — inferred phase coefficients, **NOT** a physical correction measurement.
- `vbb_study/digital_twin/nominal_f300_4f.py` and `configs/hardware/cslm_f300_nominal_4f_profile.json` — nominal relay boundary and 300 mm segments.
- Raw Agilent DSO3062A 11 Nov scope captures: stored in earlier conversation archive `Laser_Lab_Results_2026-11-11.zip`; *not yet copied to this GitHub branch*.

## Correct architecture
SLM1 condition/correct upstream field; propagate approximately 300 mm to SLM2 (estimate); SLM2 supplies desired vortex charge and carrier, unless an explicitly historical charge-split experiment. SLM2→L1 nominal 300 mm, L1→Fourier stop 300 mm, stop→L2 300 mm, L2→relay image nominal 300 mm. Axicon angle and part number unresolved. Never relabel illustrative ~10° model as calibrated bench geometry.
