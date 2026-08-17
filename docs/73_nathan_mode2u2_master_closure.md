# Nathan MODE 2U2 - Master Closure

**Status:** source-scale closure audit only. No microfabrication/sample-plane
success claim is made here, and M2V is not authorised unless the outcome is
M2U2-A.

## Hardware Binding

The Nathan branch is bound to the inherited Digital Twin where available:

- wavelength: `1.029000000e-06 m`
- SLM: `HOLOEYE LCOS-NIR`, `1920 x 1080`, pitch `8.000e-06 m`
- active area: `15.36 x 8.64 mm`
- carrier: `6.250 lp/mm`, `20.00` pixels per period
- nominal 4F focal length used for planning inference: `0.3` m
- Nathan source axicon: base angle `2.000 deg`, n `1.458`

Exact SLM phase stroke, camera scale, and measured 4F stop geometry remain
unknown in the repository evidence registers.

## Conflicts

`hardware_parameter_conflicts.csv/json` records `7` unresolved
scope splits or conflicts. The important ones are wavelength 1029/1030 nm,
axicon n=1.458 versus n=1.5 in the inherited target branch, 100 mm CSLM
placeholder versus nominal F300 geometry, command-domain carrier semantics, and
unknown camera/SLM manufacturer details.

## Native Panel

Native-panel confirmation rasterised the H/V phase masks on the exact
`1920 x 1080` rectangular panel with
`8.0 um` pitch. The 10 mm source window still
does not fit the short panel axis, while the 2 mm Gaussian beam clips only a
small tail. The propagation column is intentionally labelled as a square-grid
bridge; a full rectangular 1920 x 1080 propagation engine is not claimed.

Native rows: `8`. Passing rows:
`7`.

## Energy And Optimisation

The full energy ledger contains `242` rows over the fixed useful
hexagon region. The operating points are:

- best shape: `m2u2_opt_020_c5.75_i0.40_q-0.25_r+0.0_p0.00`
- best peak: `m2u2_opt_015_c5.75_i0.32_q+0.25_r+0.0_p0.10`
- best useful-region energy: `m2u2_opt_032_c5.75_i0.40_q+0.25_r+0.0_p0.00`
- compromise: `m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10`

## Robustness And Correction

Interaction robustness used `latin_hypercube` sampling with seed
`20260709` and `18`
samples. Pass fraction: `0.611`.
Dominant terms are listed in `interaction_robustness_summary.json`.

Blind/semi-blind correction tested `4` cases. It reports the
truth after the fact, but the search metadata states `uses_injected_truth =
false`; correction was chosen from camera/reference metrics only.

## Build Precheck

The strange six-piece segmented polariser/waveplate concept is not required for
the source-scale route. Dual SLMs plus conventional polarisation optics, a QWP,
a 4F first-order filter, and the source-scale axicon remain the recommended
architecture. However, the exact SLM phase response, QWP/HWP axis convention,
physical 4F mapping, order power split, and camera scale are not yet verified.

## Outcome

**M2U2-B.** The source-scale route remains compelling under provenance-bound and native-panel mask checks, but unverified SLM phase stroke/model specifics, physical 4F/camera calibration, and route-scope conflicts must be resolved before final lab prescription.

M2V authorised: `false`.

Output root: `outputs/figures/digital_twin/nathan_mode2u2_master_closure`.
