# Nathan MODE 2S - Source-Scale Lab Realism, Tolerance Audit, Precompensation

**Status:** MODE 2S source-scale tolerance audit and bounded inverse precompensation only. The
clean M2N/M2Q dual-SLM + carrier + 4F + QWP + axicon bench is degraded by explicit, physically
labelled imperfections, swept one at a time and in representative combined cases, with the bounded
M2Q-style precompensator as the recovery tool. No inherited objective/sample geometry, no
microfabrication sample-plane claim, no unconstrained pixel-level hologram optimisation, and six
bright lobes are never called a hexagon unless the strict C3-vs-C6 classifier gate passes.

## Question

When realistic source-scale bench imperfections are included, does the hexagonal Bessel field
survive - and if not, can the M2Q inverse/precompensation framework recover it with physically
meaningful mask corrections?

## Baseline And Model Notes

The clean degraded-operator baseline reproduces the M2N carrier/4F route exactly
(`z = 60 mm` correlation `0.9931`, first-order efficiency `0.949`; test-enforced agreement).

- **SLM aperture fit (reported, never assumed):** the 10 mm source window does NOT fit the
  1920 x 1080 x 8 um panel vertically (active area 15.36 x 8.64 mm; largest valid square window
  8.64 mm). The physical model clips the field by the active area instead of silently rescaling;
  the Gaussian beam itself (1/e radius 2 mm) loses only `1.6e-5` of its power to that clip.
- **Pixelation audit:** the 8 um pitch is finer than the source-scale grid step (26 um at
  grid 384), so per-pixel structure is not resolvable at this grid and the honest pixel-level
  statements are the sampling ratios (carrier period = 20 SLM pixels). Quantisation, LUT gamma,
  stroke, and fill factor are modelled pointwise; dead-space (fill-factor) light is unmodulated,
  carries no carrier, and is rejected by the 4F iris automatically.
- **Camera frame:** a decentred axicon forms its pattern about the cone axis; evaluation happens
  in the beam-axis (camera-aligned) frame via the exact spectral shift theorem, so the reported
  axicon-decentre sensitivity is the genuine mask/axicon structural mismatch, not a trivial
  pattern translation. (An earlier interpolation-based frame shift artificially failed 50 um
  decentres; the spectral shift removed that artefact.)

## Tier 1 - Single-Parameter Tolerances (grid 384, strict gate + 0.90 correlation bar)

| parameter | swept range | result |
|---|---|---|
| phase quantisation | continuous, 10-bit, 8-bit, 6-bit, 4-bit | all pass (even 16 levels) |
| H/V piston | 0 ... 2 pi | all pass (piston = uniform polarisation rotation; the intensity observable is invariant, Stokes metrics expose it) |
| H/V amplitude ratio | 0.8 ... 1.2 | all pass |
| QWP angle error | +/- 2 deg | all pass |
| QWP retardance error | +/- 5 deg | all pass |
| iris radius | 0.24 ... 0.80 x carrier | all pass |
| iris decentre | +/- 1.5 lp/mm | all pass |
| H/V lateral shift | 0 ... 160 um (20 SLM px) | all pass |
| **axicon decentre** | 0 ... 1.0 mm | **passes to 0.2 mm, fails beyond** (0.5 mm is a blurred near-recoverable hexagon; 1.0 mm tips into triangular dark-core; the vector singularity must sit on the cone axis to a fraction of the 64 um radial fringe period) |
| observation z offset | +/- 20 mm | all pass (the Bessel zone is long) |

The only parameter with a finite tolerance, axicon decentre, passes exactly at the typical lab
setting error (0.2 mm), so no tolerance is classified tight.

## Tier 2 - Combined Lab Cases (uncompensated)

| case | z60 correlation | strict class | result |
|---|---:|---|---|
| mild lab (8-bit, ff 0.95, 8 um shift, 0.1 rad piston, 0.25 deg QWP, 0.25 lp/mm iris decentre, 50 um axicon decentre, 1 mm z, defocus 0.1) | 0.9894 | visual_hexagonal_field | **pass** |
| moderate lab (8-bit, ff 0.90, 24 um shift + 0.2 deg rotation, 0.3 rad piston, 1.1 ratio, 1 deg QWP, 2 deg retardance, tighter iris + 0.5 lp/mm decentre, 0.2 mm axicon decentre, 5 mm z, defocus 0.3 + astig 0.2) | 0.9490 | visual_hexagonal_field | **pass** |
| bad lab (everything at 2-10x typical simultaneously, incl. 0.5 mm axicon decentre and axicon tilt) | 0.5822 | triangular_lobed_field | fail (triangular_dark_core) |

## Tier 3 - Bounded Precompensation

One near-recoverable failure existed (the 0.5 mm axicon-decentre sweep point, correlation 0.789):
the bounded 16-variable precompensator (six sector pistons, global V piston, sector rotation, duty
scale, QWP angle correction, defocus/astig corrections, iris recentre, hologram/mask recentre x/y;
all clipped to physical bounds, seeded with the measured beam-axis decentre exactly as a real bench
calibration would) recovered it to **0.9762, strict hexagon pass**. The dominant correction is
physically transparent: hologram recentre `514 um` onto the measured `500 um` axicon axis, plus a
`-1.06 deg` sector rotation and a `0.12 rad` V piston. Compensation recovery: **1/1 attempted**.
The bad-lab combined case (0.582) is below the near-recoverable threshold and was not force-fitted.

## Outputs

Generated in `outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance/` (19 artefacts):
`mode2s_clean_baseline.png`, `mode2s_slm_pixel_grid_fit.png`,
`mode2s_single_parameter_tolerances.csv/json`, `mode2s_tolerance_summary_plot.png`,
`mode2s_qwp_angle_sweep.png`, `mode2s_hv_piston_sweep.png`, `mode2s_iris_sweep.png`,
`mode2s_registration_sweep.png`, `mode2s_axicon_alignment_sweep.png`,
`mode2s_combined_cases.csv/json`, `mode2s_compensation_results.csv/json`,
`mode2s_best_uncompensated_z60.png`, `mode2s_best_compensated_z60.png`,
`mode2s_failure_examples.png`, `mode2s_outcome_report.json`, `simulation_scope_manifest.json`.

## Outcome

**M2S-A.** The realistic degraded source-scale bench remains within tolerance: mild and moderate
combined lab cases pass uncompensated, every single-parameter tolerance is at or beyond the
typical lab setting error (planning estimates, not calibrated lab data), and the one
near-recoverable failure is reliably restored by the bounded precompensator with a physically
interpretable correction. **Source-scale lab implementation is plausible.**

The single alignment that actually matters is the hologram-centre-to-axicon-axis registration
(<= 0.2 mm blind, fully correctable by the standard measure-and-recentre calibration). Everything
else - 8-bit phase, fill factor, channel piston/imbalance, QWP setting, iris placement, H/V
registration at the tens-of-microns level, camera z placement - is forgiving.

## Next Action

The natural continuations are: (i) a Shack-Hartmann-style closed-loop version of the Zernike
precompensation (MODE 2S deliberately used static corrections only); (ii) higher-resolution
pixel-true SLM modelling if a sub-8-um grid is ever justified; (iii) the separate, still-blocked
microfabrication branch (MODE 1C/M1E), which nothing here unblocks or claims.
