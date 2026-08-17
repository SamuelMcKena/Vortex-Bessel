# PHASE 2B - Visual Diagnostics and Beam-Volume Maps

**Status:** visual and interpretation layer only. Outcome **PHASE2B-A**. No Phase 1,
Phase 1R, Phase 2A, MODE 2Y, or MODE 2Z physics contract was changed. No sample-plane or
microfabrication success claim is made.

## Contract

Phase 2B consumes the fixed canonical simulation machinery and writes only to `outputs\figures\phase2b_visual_diagnostics`. Native
fixed-grid arrays are authoritative for every metric. Scalable angular-spectrum (SAS) arrays are
used for physically resampled focus rendering, while bicubic/Lanczos interpolation is display-only.
The 3D maps are pure transverse `I(x,y)` surfaces at z=60 mm: x and y are physical position, and
surface height and colour encode the same normalised intensity. They contain no propagation axis.

## Canonical Cases

| case | family | native N | native dx (um) | z planes | power drift | useful fraction |
|---|---|---:|---:|---:|---:|---:|
| `G0` | Gaussian | 512 | 19.531 | 101 | 1.461e-04 | n/a |
| `B0` | Bessel | 512 | 19.531 | 101 | 3.142e-04 | n/a |
| `V1` | vortex Bessel | 512 | 19.531 | 101 | 3.120e-04 | n/a |
| `V3` | vortex Bessel | 512 | 19.531 | 101 | 2.966e-04 | n/a |
| `H1_REALISTIC` | canonical realistic continuous vector hexagonal field | 1024 | 9.766 | 101 | 9.277e-10 | 0.3674 |
| `H1_CONTINUOUS` | continuous vector hexagonal field | 1024 | 9.766 | 101 | 9.277e-10 | 0.3674 |
| `H1_AVERAGED` | sector-averaged vector hexagonal surrogate | 1024 | 9.766 | 101 | 9.290e-10 | 0.3684 |

The mandatory 3D set is B0, V1, V3, H1 realistic, H1 continuous, and H1 averaged. Each is one
high-resolution z=60 mm intensity surface on a ring-based focus crop, with no isosurface, point-cloud,
envelope, or long propagation dimension. H1 realistic is explicitly the canonical continuous
realistic field, so the H1 realistic and H1 continuous surfaces share the same N=1536 endpoint.

## Hex Comparison

At z=60 mm, continuous local orientation improves edge-gradient sharpness by
48.76%, narrows the
80-20 transition by 33.33%,
and narrows ridge FWHM by 40.00%
relative to the sector-averaged surrogate. Early z=30 mm, canonical z=60 mm, and late z=150 mm
planes use matched crops, equal-power comparison, and shared colour limits. The N=1536 z=60 hero
uses the already justified MODE 2Z-HN sampling and SAS only as the physical focus renderer.

## Route and Energy Views

The route panel compares target/analytic, ideal sequential, realistic sequential, mild realism,
an uncompensated 0.5 mm axicon/mask offset, and its bounded digital-recentring correction. Metrics
remain on the native N=1024 arrays. The energy panel reads Phase 2A's accepted ledger directly;
first-order efficiency is not multiplied a second time, and stage throughput, plane power,
useful-region fraction, peak proxy, and SLM dead-space semantics remain distinct.

## Provenance and Limits

- Figures generated: `27` PNG/PDF pairs.
- Upstream accepted artifacts unchanged: `True`.
- Native endpoint checks reproduced: `34` / `34`.
- Mandatory 3D outputs present: `True`.
- High-N hero present: `True`.
- Display interpolation used for metrics: `False`.
- 3D surfaces show z=60 mm plane-peak-normalised intensity only; use the separate x-z/y-z and
  power panels for axial evolution.
- MODE 2Z-HN's sampled transition width and FWHM remain resolution-sensitive. Phase 2B does not
  convert those plateaus into a stronger convergence claim.

## Conclusion

Outcome **PHASE2B-A**: publication visual diagnostics were generated with native metric provenance, physical SAS focus rendering, and complete mandatory 3D intensity-map coverage. The pack is physically coherent for visual
inspection and publication composition, with all quantitative claims traceable to native arrays.
