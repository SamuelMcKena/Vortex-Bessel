# Nathan MODE 2Z - Orientation-Fidelity Interpolation Sweep

**Status:** source-scale sequential propagation study only. No split-arm architecture and no
microfabrication/sample-plane success claim.

## Question

Does propagated hexagon quality improve systematically as each 60 degree sector changes from one
fixed representative line (`eta=0`) to the true continuously varying local radial/azimuthal field
(`eta=1`), or does an intermediate morphology-energy trade-off perform better?

## Sweep Definition

The interpolation is `alpha_eta = alpha_sector_centre + eta * (theta - theta_sector_centre)` inside
each sector. The in-sector offset is wrapped geometrically before interpolation and never exceeds
30 degrees. This preserves the authoritative sector labels, Gaussian amplitude, total input power,
carrier, common 4F, QWP, axicon, grid and z samples. All decision metrics use native arrays.

Sampling: N=1024; eta=0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0; z=0
to 200 mm in 2 mm steps, including exact z=60 mm.

## Gate Definition Audit

The first generic classifier draft applied a 5% relative floor to every metric and required a high
Spearman coefficient even for native-grid width metrics that evolve as monotone plateaus. That draft
labelled non-reversing plateau-and-jump sequences as mixed. Before interpretation, `M2Z-GATE-FIX-1`
replaced it with metric-aware rules: V0 correlation requires an absolute gain of 0.01; dimensional
sharpness metrics require a 5% relative quality gain; monotonic evidence is either Spearman rho >=
0.70 or at least 90% nondecreasing eta steps with a 1% numerical tolerance. No field, propagation
array or raw metric was changed by this classifier repair.

## Trend Audit

- `ideal`: 4/4 predeclared morphology trends pass; strict onset eta = `none`.
- `realistic`: 4/4 predeclared morphology trends pass; strict onset eta = `0.70`.

The four predeclared trend observables are V0 correlation, edge-gradient sharpness, inverse 80-20
transition width and inverse bright-ridge FWHM. Corner concentration, peak intensity and useful
energy are retained as independent trade-off diagnostics rather than folded into a pass condition.

At z=60 mm, V0 correlation and edge-gradient sharpness rise at every eta, while transition width and
ridge FWHM improve through native-grid plateaus without a material reversal. The realistic strict
gate first passes at eta=0.70 and remains passed through eta=1.00. Useful-region energy decreases
slightly toward eta=1, peak intensity has a shallow interior maximum, and best-z/persistence evolve
nonmonotonically. The sweep therefore shows monotonic transverse morphology improvement alongside a
real energy/axial trade-off, not a superior intermediate morphology optimum.

## Continuous Endpoint

| route | z60 corr | strict | edge grad (mm^-1) | 80-20 width (mm) | ridge FWHM (mm) | best z (mm) |
|---|---:|---|---:|---:|---:|---:|
| `ideal` | 1.000000 | False | 10.9102 | 0.12430 | 0.06905 | 18.0 |
| `realistic` | 0.993551 | True | 10.8060 | 0.12430 | 0.06905 | 38.0 |

## Conclusion

Outcome **M2Z-A**: z60 morphology improves systematically toward the continuous endpoint in both
routes; energy and axial evolution remain trade-offs. MODE 2Z does not replace the independent
MODE 2X local-vector truth gate or the MODE 2Y endpoint comparison. It tests how morphology evolves
between those two physically distinct endpoints.

Output root: `outputs\figures\digital_twin\nathan_mode2z_orientation_interpolation`.

<!-- MODE2Z-HN-START -->
## Targeted High-N Confirmation

The selected z=60 mm N=1536 check is reported in docs/87. Outcome
**M2Z-HN-C**: correlation and edge gains are stable, but the selected onset shifts and sampled width endpoints remain resolution-sensitive. Selected-grid strict onset is
`eta=0.80`. This threshold remains
project-specific and is not a universal experimental tolerance.
<!-- MODE2Z-HN-END -->
