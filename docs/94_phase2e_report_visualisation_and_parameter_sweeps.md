# Phase 2E - Report Visualisation and Parameter Sweeps

> **Propagation supersession notice (Phase 2E-PROP):** the original dense scalar x-z/y-z
> figures and their shared-transverse-peak interpretation are retained only as superseded
> diagnostics. The forensic replacement and current publication block are documented in
> `docs/95_phase2e_propagation_forensic_repair.md`. This notice does not supersede the 3D,
> sweep, realism, energy, or hero products.

**Status:** outcome **PHASE2E-A**. This is a visual/presentation layer and a separately scoped
diagnostic-screening layer. It does not rewrite Phase 1/1R/2A/2B/2C physics, overwrite accepted
arrays, or promote uncalibrated dimensions, fluence, or material response.

## Purpose

The pack gives G0, B0, V1, V3, H1 continuous and H1 sector-averaged one report-grade visual
language. It contains core intensity/phase/propagation/profile atlases; pure transverse 3D intensity
surfaces; dedicated full-computational-window x-z/y-z intensity maps; H1 Stokes/orientation comparisons;
ideal/realistic/degraded routes; energy accounting;
pedagogical schematics; seven broad beam-parameter families; and ten source-scale physical-error
families.

Output root: `outputs/figures/phase2e_report_visualisation`.

## Consistency Contract

- Fonts, panel labels, line widths and colour maps come from one serialised style object.
- Source-scale transverse axes are in millimetres. The accepted Phase 2C inset retains micrometres.
- Like-for-like scalar core/3D crops are +/-0.25 mm; H1
  crops are +/-0.30 mm. Route-realism crops remain
  +/-0.80 mm so the displaced degraded endpoint is visible.
- Transverse ROI occupancy is presentation-governed: the H1 x-y crop is tight enough to expose the sixfold core;
  effective-NA and displaced-realism plates retain their matched full fields but add labelled common
  +/-0.18 mm detail insets so compact beam structure remains legible.
- Propagation comparisons use z=0--200 mm, the complete 10 mm source-grid window, matched transverse limits and 1025x601
  fixed-coordinate spectral maps. They are physical Fourier-series evaluations of the BL-ASM,
  not resized inherited stack cuts. No x-z/y-z ROI is applied. Every atlas pairs an uncapped global
  linear `I/Imax` map with the same global linear ratio shown on a fixed 0--0.01 colour range; values
  above 0.01 are visibly saturated only in the second view. A separate globally linear shared-peak
  curve preserves axial intensity evolution. No per-z renormalisation, logarithm, gamma,
  percentile-based limit or spatial interpolation is applied, and no metric or physical coordinate
  is supplied by the renderer.
- Scalar headline core and propagation plates use the accepted Phase 2A finite-aperture
  `realistic_fixed_bench_route`, including its 1.8 mm hard pupil and fixed-bench terms. The ideal
  untruncated Gaussian-axicon field appears only as a labelled control in the pupil-boundary audit.
  H1 retains its validated common-4F/vector-ASM route, which does not apply the scalar objective-pupil stop.
- Every 3D surface uses x/y in mm, height in normalised intensity, z limits 0--1 and one view angle.
- The top-down parity panel uses the same cropped array and colour limits as its oblique surface.
- Linear/log state and global/per-panel normalisation are explicit in each record.

## Native Versus Display Data

Accepted Phase 2B fixed-grid arrays retain authority for accepted endpoint metrics. Dense x-z/y-z
maps are recomputed by direct centred inverse-DFT synthesis over the complete source-grid window using
the same Matsushima band limit; native inverse-FFT parity and adjacent N=768-to-N=1024 convergence
are recorded in the case summary. The convergence gate requires correlation >=0.98 and relative
L2 <=0.20. Scalar same-source scalable-ASM line parity has a declared correlation floor of
0.95; inherited N=512 SAS comparisons are retained separately as
cross-grid diagnostics. Projected-vector H1 is gated by exact native vector-ASM parity and accepted
endpoint reproduction; its N=1024-to-N=1536 line comparison is diagnostic because scalar SAS is
not an interchangeable validator for a divergence-projected vector spectrum. Scalable angular spectrum arrays are physical resamplings used for
focus presentation. Only transverse x-y fields are cropped. For report rendering, scalar SAS crops are
cubic-resampled x4 and H1 SAS crops x3;
the same display array feeds each 3D/top-down pair. Dense propagation plates use the native
1025x601 spectral array without spatial interpolation. The route-realism plate uses bicubic `imshow`
for its wide view and cubic display-only beam-centred detail insets. All metrics are computed before display
interpolation. No metric is read from a raster image. The Phase 2C hero is a presentation-only
vertical composite of accepted Phase 2C figures so their objective ROIs use the report width; its
numerical evidence remains the Phase 2C CSV files.

## Propagation Boundary Audit

The apparent B0 axial beading and upper-field flare were tested rather than cosmetically removed.
The canonical route clips a 2.0 mm 1/e Gaussian radius
with a 1.8 mm hard pupil, retaining
0.802 of the ideal source power. The corresponding
geometric hard-pupil Bessel-zone end is 112.5 mm;
the 1/e beam-radius estimate is 125.0 mm.
Removing the hard pupil reduces 20--100 mm axial ripple RMS from
0.0783 to
0.0043. BL-ASM and unbandlimited ASM
agree at correlation 1.00000000, with
maximum normalised difference
2.89e-05. The modulation
and post-zone flare are therefore finite-aperture effects in the accepted hard-pupil model, not
Nyquist failure or a Matsushima-mask artifact.

## Diagnostic Sweep Boundary

The 17 sweep families contain 85 native SAS points. Seven broad
beam-parameter families use an analytic finite-energy vortex-Bessel screening field. Ten physical
error families retain canonical dual-SLM quantisation, common-4F filtering, objective pupil and
axicon propagation while varying one pre-propagation control at its declared plane: input decentre,
input tilt, SLM phase error, Fourier-iris offset, pupil decentre, axicon decentre, defocus,
astigmatism, coma or spherical aberration. All remain diagnostic, not calibrated tolerances or
replacement fixed-bench claims. Every point records Nyquist and SAS validity plus native metrics.

## Report Hero Figures

- `hero_h1_continuous_vs_averaged` -> `outputs/figures/phase2e_report_visualisation/03_h1/h1_continuous_vs_averaged_matched.png`
- `hero_vortex_ideal_realistic_degraded` -> `outputs/figures/phase2e_report_visualisation/05_realism/hero_vortex_ideal_realistic_degraded.png`
- `hero_energy_loss_efficiency` -> `outputs/figures/phase2e_report_visualisation/06_energy/hero_energy_loss_efficiency.png`
- `hero_vortex_beam_family` -> `outputs/figures/phase2e_report_visualisation/08_hero_figures/hero_vortex_beam_family.png`
- `hero_vortex_parameter_dependence` -> `outputs/figures/phase2e_report_visualisation/08_hero_figures/hero_vortex_parameter_dependence.png`
- `hero_scalar_vector_objective_benchmark` -> `outputs/figures/phase2e_report_visualisation/08_hero_figures/hero_scalar_vector_objective_benchmark.png`

## Governance

- Cases summarised: 6.
- Figures: 47 PNG/PDF pairs.
- Endpoint checks: 66, all reproduced.
- Unity-loss energy rows are explicitly labelled as assumptions/placeholders, and simulated losses
  below 0.001 are marked numerically rather than rendered as unexplained blank bars.
- Upstream files are SHA-256 checked before and after in-memory reconstruction.
- A normal run refuses to overwrite an existing Phase 2E root; replacement requires explicit
  `--overwrite` and still cannot write into accepted upstream roots.

## Limitations

The parameter sweeps are trend-screening simulations, not calibrated experimental predictions. The
effective-NA sweep is a source-plane spectral cutoff diagnostic and is not a replacement for the
vector Debye objective in Phase 2C. Absolute sample dimensions, pulse fluence, damage thresholds,
nonlinear material modification and experimental validation remain calibration-blocked.
