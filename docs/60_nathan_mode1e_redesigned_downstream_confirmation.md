# Nathan MODE 1E - Actual Redesigned Downstream Confirmation

**Status:** MODE 1E actual-downstream confirmation only. Ideal P2 input, redesigned pre-axicon and
target/surface `k_r`, actual inherited MODE 1 downstream machinery. No patterned-HWP/QWP/dual-SLM
physical routes, no 4F carrier/iris, no panel realism, no waveplate errors, no route ranking.
Proxy-only, source-template, and P2 Jones passes are structurally excluded from `M1E-A`.

## Question

Can a legitimate redesigned downstream configuration, using the physically meaningful changes
identified by MODE 1D, produce a sample-plane micro-scale version of the minimum accepted
Nathan-style hexagonal Bessel target (`N = 12`), when run through the **actual** downstream model
rather than a free-space proxy?

## Design Method

Each candidate is a fully-resolved redesigned `TwinConfig`, not a proxy:

```text
k_r,pre     = 2 pi N / R_P2                      (physical_axicon.axicon_base_angle_deg)
k_r,surface = M_k k_r,pre                        (target.target_core_diameter_m = 2 x j_0,1 / k_r,surface)
NA_required = k_r,surface / k0
R_P2        -> laser.beam_radius_on_slm_m        (vector waist re-derived from the twin)
```

with the audited MODE 1C mapping factor `M_k = 123.899`, objective NA `0.45`, SLM-safe radius
`3888 um`, and SLM phase-period floor `4 x 8 um`. After construction, the candidate is re-audited
through the same `resolve_vector_axicon_parameters` path the downstream run uses; any mismatch is
reported as `surface_kr_fingerprint_mismatch` and the candidate is blocked, never silently run.
The downstream tripwire (`assert_locked_kr_fingerprint`) is retargeted to the candidate's declared
surface `k_r`, so it still fires on any inconsistency; it is never skipped.

## Candidates

Requested `N in {10, 11, 12, 13, 13.5}` at `R in {0.75 x SLM-safe, SLM-safe}`:

| candidate | N target | N actual at P2 | NA required | status | class | N12 angular corr | gate |
|---|---:|---:|---:|---|---|---:|---|
| N10 @ 2916 um | 10.0 | 9.69 | 0.437 | actual F0 | dark_core_structured | 0.089 | fail |
| N11-13.5 @ 2916 um | 11-13.5 | - | 0.481-0.590 | infeasible (NA) | - | - | not run |
| N10 @ 3888 um | 10.0 | 8.80 | 0.328 | actual F0 | triangular_lobed | 0.085 | fail |
| N11 @ 3888 um | 11.0 | 9.68 | 0.361 | actual F0 | triangular_lobed | 0.083 | fail |
| N12 @ 3888 um | 12.0 | 10.56 | 0.393 | actual F0 | dark_core_structured | 0.087 | fail |
| N13 @ 3888 um | 13.0 | 11.44 | 0.426 | actual F0 + F2 | dark_core_structured | 0.090 | fail |
| N13.5 @ 3888 um | 13.5 | 11.87 | 0.443 | actual F0 | dark_core_structured | 0.085 | fail |
| inherited control | 4.11 | 4.11 | 0.263 | actual F0 | triangular_lobed | 0.121 | fail |

All six feasible candidates ran as **actual downstream** runs: requested vs resolved `k_r,pre` and
`k_r,surface` agreed to relative error `<= 2e-16` (zero fingerprint-blocked candidates, zero proxy
runs). The primary `N = 12` candidate reproduces the MODE 1D budget numbers: `k_r,pre = 1.94e4
m^-1`, `k_r,surface = 2.40e6 m^-1`, `NA = 0.393 < 0.45`, base angle `0.364 deg`, target
equivalent-`l0` core diameter `2.00 um`.

`N actual at P2` is lower than the request because the P2 window/SLM-safe aperture truncates the
Gaussian near its 1/e radius (measured 1/e radius `3420 um` for the `3888 um` request); the
realised regime is therefore `N ~ 8.8-11.9`.

## Source Templates

Both templates were built with the MODE 1D validated-source machinery (`grid_n = 384`,
`z_planes = 21`) and both classify as accepted `visual_hexagonal_field`:

- `N = 12` primary template (MODE 1D minimum accepted target), actual ring count `12.0`;
- `N = 31` V0-like stricter reference.

## Result

**No redesigned candidate produces a sample-plane hexagon.** The redesign is a real but
insufficient improvement over the inherited control:

- order-3/order-6 ring content drops from `0.34` (control) to `~0.10-0.13`;
- six-sector energy balance improves to `max/min ~ 1.07-1.09`;
- the core is strongly dark everywhere (`dark-core ratio ~ 0.002-0.008`);
- but the decisive C3-vs-C6 discriminator stays triangular-leaning: `c120 - c60 ~ +0.07...+0.09`
  (a genuine hexagon needs `<= 0.04`), the fields classify as `dark_core_structured_field` or
  `triangular_lobed_field`, no plane in any candidate z-stack reaches a persistent
  `visual_hexagonal_field` (best fraction `0.15`), and the `N = 12` template similarity stays
  at `~0.09` angular / negative XY correlation.

The F2 vectorial pupil-spectrum reference for the shortlisted best candidate (`N13 @ 3888 um`)
agrees with F0 (`equal-power full-field correlation 0.824`), so the non-hexagonal outcome is not an
artefact of the scalar per-component focus bridge.

Honest z-dependence note: the redesigned candidates concentrate their energetic zone at
`z ~ 0-25 um`, short of the declared mid-stack reference plane at `75 um` (the z-span is inherited
from the unchanged target Bessel length). No plane anywhere in the stack classifies as a visual
hexagon, so the conclusion does not hinge on the reference-plane choice, but a future redesign
iteration should re-declare the z-window to the redesigned zone. Two further caveats: the square
P2 window truncation (the SLM-safe aperture) adds axis-aligned diffraction streaks, and the source
template's angular profile is sparsely sampled at `grid_n = 384` (ring radius ~4 px), which
depresses the angular correlations; the class veto (`c120 - c60`) is template-independent and is
the decisive failure.

## Outputs

Generated in `outputs/figures/digital_twin/nathan_mode1e_redesigned_downstream/`:

- `mode1e_design_candidates.csv/json`
- `mode1e_source_template_N12.png`, `mode1e_source_template_N31.png`
- `mode1e_current_inherited_control.png`
- `mode1e_candidate_<id>_f0.png` for all six actual runs
- `mode1e_candidate_mode1e_N13p0_R003888um_f2.png` (shortlisted F2)
- `mode1e_template_comparison_<id>.png` for all six actual runs
- `mode1e_outcome_report.json`
- `simulation_scope_manifest.json`

## Outcome

**M1E-B.** Legitimate actual redesigned downstream configurations (resolved `k_r` fingerprints
matching the request exactly) improve the result - lower C3 ring content, balanced sectors, strong
dark core - but the sample-plane field remains triangular/lobed or dark-core-structured, not a
visual hexagon, and does not match the `N = 12` source-template class. **MODE 2A/2B remains
blocked.**

## Next Action

The k_r/NA/aperture budget itself is now confirmed reachable (`NA 0.39 < 0.45` at `N = 12`), so the
remaining failure is structural to the inherited downstream architecture rather than a pure
ring-count deficit. If this line is pursued further, the next scoped step should be an optical
redesign that changes what MODE 1E deliberately held fixed: the surface/pre mapping factor
(`M_k = 123.9`; MODE 1D says a `2.26x` reduction at the SLM-safe radius would be needed at NA 0.45
for V0-like ring counts), the focus-bridge geometry, or a z-window re-declared to the redesigned
Bessel zone. Do not treat the MODE 1E improvement trend as a sample-plane hexagon claim.
