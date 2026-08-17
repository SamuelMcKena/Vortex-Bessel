# Nathan MODE 2X - Local Radial/Azimuthal Vector-Field Truth Audit

**Status:** source-scale pre-axicon local vector truth only. No new architecture.
No microfabrication/sample-plane success claim.

## Why This Audit Exists

Sector averaging is useful as a meeting schematic, but one constant arrow cannot prove a radial or
azimuthal field. Each sector is locally linearly polarised, but its orientation varies continuously
with position. A radial sector is not represented physically by one constant average arrow; its local
polarisation follows e_r(theta). An azimuthal sector follows e_theta(theta).

## Method

The native complex Cartesian field is converted point by point using
`Er = Ex cos(theta) + Ey sin(theta)` and `Etheta = -Ex sin(theta) + Ey cos(theta)`.
Radial and azimuthal purity are intensity-weighted local-basis powers. Line orientation is obtained
from `0.5 atan2(S2,S1)` and compared modulo pi. Local linearity is audited from `|S3|/S0` using the
project convention `S3 = -2 Im(Ex Ey*)`.

The central singular neighbourhood, pixels below `1e-6` of peak intensity, and a 0.5 degree guard
on authoritative V0 sector boundaries are excluded. Native samples determine every metric;
interpolation is display-only.

## Results

| route | radial purity | azimuthal purity | angle RMS (rad) | S3 RMS | local pass | final strict pass |
|---|---:|---:|---:|---:|---|---|
| `authoritative_analytic_target` | 1.000000000 | 1.000000000 | 9.627e-17 | 0.000e+00 | True | False |
| `ideal_patterned_hwp` | 1.000000000 | 1.000000000 | 9.627e-17 | 0.000e+00 | True | False |
| `ideal_abstract_dual_slm_qwp` | 1.000000000 | 1.000000000 | 1.133e-16 | 6.066e-17 | True | False |
| `ideal_sequential_dual_slm` | 1.000000000 | 1.000000000 | 1.133e-16 | 6.066e-17 | True | False |
| `realistic_sequential_carrier_common_4f` | 0.978467221 | 0.978112259 | 1.538e-01 | 4.343e-16 | False | True |
| `REALISTIC_4F_HEXAGON_REFERENCE` | 0.978467221 | 0.978112259 | 1.538e-01 | 4.343e-16 | False | True |
| `m2s_combined_moderate_lab` | 0.867344432 | 0.902057683 | 3.493e-01 | 1.345e-01 | False | False |
| `m2s_combined_bad_lab` | 0.252185449 | 0.340242770 | 1.055e+00 | 2.655e-01 | False | False |
| `compensated_axicon_mask_offset_0p5mm` | 0.980285701 | 0.979300010 | 1.483e-01 | 8.503e-05 | False | False |
| `strict_c6.75_i0.40_q-0.25_r+0.0_p0.00` | 0.980193143 | 0.980247934 | 1.467e-01 | 6.171e-03 | False | True |

The pre-axicon local-vector gate and repaired final z=60 mm intensity-hexagon gate are independent.
A route can pass either one without passing the other. No audited ideal implementation uses a
sector-averaged constant orientation: the analytic target, patterned HWP, abstract dual-SLM and
sequential dual-SLM fields all retain the continuously varying local `alpha(theta)` map.

The canonical realistic 4F route remains locally linear (`S3` RMS at floating-point noise), but its
hard common-4F filtering smooths the discontinuous sector boundaries enough to produce about 2.2%
cross-basis leakage and a 0.154 rad intensity-weighted line-angle RMS. It therefore keeps the repaired
final strict hexagon while failing the independently fixed local-vector gate. The ideal-route final
strict entries are false because the repaired candidate gate is calibrated to the immutable realistic
4F reference; they are not failures of ideal local vector truth.

## Meeting Figures

- `outputs\figures\digital_twin\nathan_mode2x_local_vector_truth\01_figures\sector_averaged_polarisation_schematic.png` is explanatory only.
- `outputs\figures\digital_twin\nathan_mode2x_local_vector_truth\01_figures\true_local_polarisation_field.png` is the physically truthful local field.

## Conclusion

Outcome **M2X-B**. Ideal local truth passes, but the realistic sequential route requires revised
filtering or correction before a high-purity local-vector claim is accepted. The accepted architecture
remains one sequential collinear beam through SLM1,
the conditional polarisation swap, SLM2, optional swap-back, the common 4F, QWP and axicon. MODE 2X
adds a local vector-field audit only. It does not authorise or claim microfabrication/sample-plane
performance.
