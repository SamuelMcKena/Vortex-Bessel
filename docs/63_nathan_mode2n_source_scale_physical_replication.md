# Nathan MODE 2N - Source-Scale Physical Bench Replication

**Status:** MODE 2N source-scale bench replication only. Gaussian input, HWP/dual-SLM/QWP
generation (plus a carrier + 4F first-order filtered dual-SLM route), Nathan source-scale physical
axicon, free-space vector propagation observed around `z = 60 mm`. The inherited objective/sample
microfabrication geometry is **not** used, and no micro-scale sample-plane claim is made or judged
by MODE 1C/M1E constraints. A route succeeds only on the propagated intensity, never on pre-axicon
overlap alone.

## Question

Can a physically realistic optical bench (dual SLMs / HWPs / QWPs / 4F filtering / axicon /
free-space propagation) reproduce Nathan's Fig. 4 source-scale hexagonal Bessel beam?

## Pipeline

```text
Gaussian (1030 nm, w0 = 2 mm)
  -> generation route (2N-A / 2N-B / 2N-C)
  -> physical axicon (n = 1.458, medium n = 1.0, base angle 2 deg, p/s Fresnel)
  -> free-space vector angular-spectrum propagation (z = 0.1 ... 290 mm, ref 60 mm)
  -> I = |Ex|^2 + |Ey|^2 + |Ez|^2 on Nathan's axis-sampled 10 mm grid
```

Everything downstream of generation is byte-identical to the validated V0 source machinery
(`source_parity_grid`, literal segmented RA convention, `_apply_free_space_vector_axicon`,
`_free_space_intensity_stack`), so route-vs-V0 differences measure only the generation hardware.
The V0 reference itself classifies `visual_hexagonal_field` at the confirmation resolution
(1024 grid, 62 z-planes with z = 60 mm inserted exactly).

## Routes (confirmation run, grid 1024)

| route | pre-axicon overlap | z60 correlation | angular corr | x-z corr | class | pass |
|---|---:|---:|---:|---:|---|---|
| 2N-A ideal patterned HWP (`beta = alpha/2`) | 1.000000 | 1.000000 | 1.000 | 1.000 | visual_hexagonal_field | pass |
| 2N-B ideal dual-SLM + QWP (`H: +alpha`, `V: -alpha + pi/2`, QWP `-pi/4`) | 1.000000 | 1.000000 | 1.000 | 1.000 | visual_hexagonal_field | pass |
| 2N-C dual-SLM + carrier + 4F first order + QWP | 0.949279 | 0.993609 | 0.999 | 0.996 | visual_hexagonal_field | pass |

Route 2N-C details (phase-only SLM channels, carrier 6.25 lp/mm, hard iris of radius 2.5 lp/mm
around the first order, analytic carrier removal, lossless recombination, uniform QWP -pi/4):

- first-order efficiency `0.9493` (the 5.1% loss is spectral clipping of the sector-discontinuity
  tails by the iris; an ideal continuous phase-only SLM sends everything into the shifted order);
- zero-order content before the iris `3.0e-4` of total power; zero-order leakage after the iris
  `0.0` (the DC and first-order disks are disjoint at this geometry);
- 4F power ledger closes to relative error `2.7e-16`;
- pre-axicon Stokes RMS `0.52` is dominated by very dim outer-envelope pixels inside the metric
  mask; the propagated beam is nonetheless V0-like at `0.9936` equal-power correlation with a dark
  core (`0.001`) and `c120 - c60 = -0.05` (C6-dominant, not triangular);
- the on-axis intensity trace correlation is not meaningful for this dark-core beam (the on-axis
  signal is ~0 at all z) and is reported but not gated on.

Pixel fill factor, phase quantisation, and aberrations are deliberately not modelled yet: the task
was the basic carrier-filtered case first.

## Outputs

Generated in `outputs/figures/digital_twin/nathan_mode2n_source_replication/`:

- `mode2n_target_pre_axicon.png`
- `mode2n_route_patterned_hwp_pre_axicon.png`, `mode2n_route_dual_slm_qwp_pre_axicon.png`,
  `mode2n_route_dual_slm_4f_pre_axicon.png`
- `mode2n_v0_reference_z60.png`, `mode2n_patterned_hwp_z60.png`, `mode2n_dual_slm_qwp_z60.png`,
  `mode2n_dual_slm_4f_z60.png`
- `mode2n_xz_comparison.png`
- `mode2n_route_metrics.csv/json`
- `mode2n_outcome_report.json`
- `simulation_scope_manifest.json`

The x-z comparison shows the long non-diffracting hexagonal Bessel zone (roughly 10-200 mm with
the declared reference plane at 60 mm inside it) for V0 and all three routes.

## Outcome

**M2N-A.** Ideal patterned-HWP and ideal dual-SLM/QWP routes reproduce V0 after axicon propagation
exactly, and the carrier/4F dual-SLM route also reproduces it with acceptable fidelity
(`0.9936` z = 60 mm correlation, hexagonal class preserved). Experimental source-scale replication
of Nathan's Fig. 4 beam is plausible.

## Next Action

The natural realism ladder for this branch is: SLM pixel fill factor and phase quantisation, iris
misalignment/size sweeps, waveplate retardance/axis errors, and recombination loss - each as a
scoped sweep on route 2N-C. Whether this source-scale bench can be adapted into the
microfabrication architecture remains a separate question that MODE 1C/M1E currently block; this
outcome makes no claim about it.
