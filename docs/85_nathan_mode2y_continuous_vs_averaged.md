# Nathan MODE 2Y - Continuous vs Sector-Averaged Propagation Audit

**Status:** source-scale sequential propagation comparison only. No split-arm architecture and no
microfabrication/sample-plane success claim.

## Question

With the Gaussian envelope, six-sector labels, total input power, axicon and propagation held fixed,
does the true continuously varying local radial/azimuthal field produce a sharper propagated hexagon
than a deliberately piecewise-constant one-line-per-sector surrogate?

## Fixed Comparison

The continuous input follows local `e_r(theta)` or `e_theta(theta)`. The surrogate assigns the
orientation at each 60 degree sector centre to every pixel in that sector. It is a diagnostic
surrogate, not a physically correct radial/azimuthal field. Input powers match to floating-point
precision. Both are propagated through the ideal sequential-equivalent route and the validated
carrier + common-4F + QWP sequential route, then through the same source-scale axicon and vector ASM.

Sampling is native N=1024, z=0 to 200 mm in 2 mm steps. Display interpolation never enters a metric.

## Results

| route | z60 corr | strict | edge grad (mm^-1) | 80-20 width (mm) | corner conc. | ridge FWHM (mm) | best z (mm) |
|---|---:|---|---:|---:|---:|---:|---:|
| `ideal_continuous` | 1.0000000 | False | 10.9102 | 0.12430 | 2.5312 | 0.06905 | 18.0 |
| `ideal_sector_averaged` | 0.9786990 | False | 7.6119 | 0.16573 | 3.4891 | 0.09667 | 122.0 |
| `realistic_continuous_common_4f` | 0.9935508 | True | 10.8060 | 0.12430 | 2.5201 | 0.06905 | 38.0 |
| `realistic_sector_averaged_common_4f` | 0.9750858 | False | 7.2640 | 0.16573 | 3.3995 | 0.09667 | 38.0 |

Ideal median continuous sharpness change: +36.67%.
Realistic median continuous sharpness change: +36.67%.
The predeclared decision requires at least three of four sharpness metrics to improve by 5% in both
routes before calling the continuous field measurably sharper.

The continuous field wins edge-gradient sharpness, 80-20 transition width and bright-ridge FWHM in
both routes. The averaged surrogate has greater corner concentration and slightly greater peak and
useful-region energy, but poorer V0 morphology and it fails the repaired strict gate in the realistic
route. The realistic continuous field remains strict-eligible.

## Conclusion

Outcome **M2Y-A**: continuous is measurably sharper in both ideal and realistic routes. This conclusion concerns source-scale propagated
morphology only. Local-vector truth remains the separate MODE 2X result, and the repaired strict
hexagon gate remains independently reported.

Output root: `outputs\figures\digital_twin\nathan_mode2y_continuous_vs_averaged`.
