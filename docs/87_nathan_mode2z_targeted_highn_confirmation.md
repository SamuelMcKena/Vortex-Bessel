# Nathan MODE 2Z-HN - Targeted High-N Threshold Confirmation

**Status:** targeted source-scale numerical confirmation only. Realistic sequential common-4F route,
exact z=60 mm, selected eta values. No split-arm or microfabrication/sample-plane claim.

## Scope

This is not another axial sweep. It regenerates an N=1536 ideal-continuous V0
reference and realistic-continuous strict-gate reference, then evaluates only eta =
0.0, 0.4, 0.6, 0.7, 0.8, 1.0 at z=60 mm. Native dx is
6.510 um. The 6.25 lp/mm carrier has
24.58 samples per period, and the
carrier-plus-iris filtered band retains a 8.78x
Nyquist margin. Interpolation is display-only.

## Results

| eta | corr N1024 | corr N1536 | strict N1024 | strict N1536 | edge N1536 | width N1536 | FWHM N1536 |
|---:|---:|---:|---|---|---:|---:|---:|
| 0.0 | 0.975086 | 0.975102 | False | False | 9.1344 | 0.165728 | 0.099437 |
| 0.4 | 0.987541 | 0.987557 | False | False | 10.9498 | 0.132583 | 0.066291 |
| 0.6 | 0.991120 | 0.991136 | False | False | 11.8367 | 0.132583 | 0.066291 |
| 0.7 | 0.992298 | 0.992315 | True | False | 12.3104 | 0.132583 | 0.066291 |
| 0.8 | 0.993089 | 0.993105 | True | True | 12.7540 | 0.132583 | 0.066291 |
| 1.0 | 0.993551 | 0.993567 | True | True | 13.4575 | 0.099437 | 0.066291 |

Selected-grid strict onset: N=1024 `0.70`;
N=1536 `0.80`. At N=1536,
eta=0.7 fails only because `angular profile correlation below threshold`; eta=0.8 passes.

Correlation monotonic: `True`. Edge-gradient monotonic:
`True`. Transition width nonincreasing:
`True`. FWHM nonincreasing:
`True`. Stable endpoint gains:
`False`.

| endpoint diagnostic | N=1024 | N=1536 | convergence interpretation |
|---|---:|---:|---|
| correlation gain | 0.018465 | 0.018465 | stable |
| edge-gradient gain | 48.76% | 47.33% | stable relative gain |
| transition-width narrowing | 33.33% | 66.67% | resolution-sensitive |
| ridge-FWHM narrowing | 40.00% | 50.00% | resolution-sensitive |
| peak change | -2.02% | -1.87% | stable |
| useful-energy change | -0.47% | -0.44% | stable after within-grid normalisation |

Correlation/edge endpoint stability: `True`.
Peak/energy endpoint stability: `True`.
Sampled width endpoint stability: `False`.

Resolved width levels among the selected eta values: transition width N=1024
`3` versus N=1536
`3`; FWHM N=1024
`2` versus N=1536
`2`. Plateaus that remain at high N are reported as plateaus,
not smoothed into artificial continuous measurements.

## Conclusion

Outcome **M2Z-HN-C**: correlation and edge gains are stable, but the selected onset shifts and sampled width endpoints remain resolution-sensitive. The eta threshold is a calibrated,
project-specific simulation threshold, not a universal experimental tolerance.

Output root: `outputs\figures\digital_twin\nathan_mode2z_orientation_interpolation\07_highN_confirmation`.
