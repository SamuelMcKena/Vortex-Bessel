# Nathan Visual Reproduction Ladder

**Status:** strict visual gate, not a route-comparison study.  
**Boundary:** no HWP/SLM robustness, route ranking, best-H6 plane selection, paper-resolution focus conclusion, or publication claim is allowed from this report.

## Visual Pass Condition

A result does not pass because H6 is non-zero, a metric peaks at one z plane, or the input-field overlap is close to one. It passes only if the displayed transverse intensity shows a continuous six-sided hollow wall, a dark central region, no dominant six-spot lattice or grid artefact, and coherent structure across adjacent z planes.

All images here use fixed declared reference planes. Metrics are diagnostics only.

## V0 Propagated-Output Parity Control

V0 is an isolated Nathan source-style propagation test. It does not use ObjectiveMap, the scalar focal bridge, Digital Twin geometry, or sample-plane mapping. Input-array parity and propagated-output visual parity are reported separately.

The reference image must be the crop extracted from `Laser_Manufacturing.pdf`, page 7, Figure 4:

- crop: `outputs/reference/nathan_marco_report_figure4_page7_crop.png`
- provenance: `outputs/reference/nathan_marco_report_figure4_page7_crop.provenance.json`
- full page render used for cropping: `outputs/reference/nathan_marco_report_page7_render.png`

The unrelated green "Total Intensity / Intensity after polarizer" montage is not a Nathan Figure 4 reference and must not be used in this notebook.

Frozen V0 source values:

| Parameter | Value |
|---|---:|
| wavelength | `1030e-9 m` |
| Gaussian radius | `2.0e-3 m` |
| axicon index | `1.458` |
| axicon apex angle | `176 deg` |
| interpreted base angle | `2 deg` |
| sector pairs | `3` |
| sector width | `60 deg` |
| grid | `1024 x 1024` minimum for the primary decision run |
| window | `10 mm` |
| reference z | exactly `60 mm` inserted into the z stack |
| z range | `0.1 mm` to `290 mm` |
| z planes | at least `60` plus the inserted reference plane when needed |

The notebook now creates:

- `nathan_visual_ladder_v0_reference_vs_reproduction.png`: four-row comparison with the actual Figure 4 reference crop, V0A literal source-port outputs, V0B project-path outputs, and V0A-versus-V0B difference diagnostics.
- `nathan_visual_ladder_v0_field_views.png`: full transverse field, central 10% zoom, local normalisation, and common normalisation.
- `nathan_visual_ladder_v0_convergence_xy.png`: N=512, N=1024, and optional N=1536 central-crop visual convergence comparison.

The V0 paths are:

- V0A: `vbb_study.digital_twin.nathan_literal_source_port`, an isolated literal source-scale port.
- V0B: `vbb_study.digital_twin.nathan_vector_hexagon`, the current project vector implementation using identical source parameters.

The V0 report function returns separate keys for `input_array_parity`, `source_parameter_parity`, `propagated_output_parity`, `propagated_output_visual_verdict`, and `numerical_resolution_status`. The notebook must not overwrite these with `replace(..., verdict=...)`.

## V1 Inherited Laser/Axicon Pre-Objective Gate

V1 uses the inherited Digital Twin laser and inherited vector-axicon parameters, then propagates by vector free-space ASM only. It does not use ObjectiveMap, scalar focus, or sample-plane mapping.

Inherited V1 axicon facts:

| Quantity | Value |
|---|---:|
| `k_r_pre_m_inv` | `12940.60517266251` |
| `k_r_surface_m_inv` | `1603333.3333333333` |
| base angle | `0.2428511254567136 deg` |
| axicon index | `1.5` |
| medium index | `1.0` |

V1 is reported only as: **Current inherited pre-objective result under its present parameterisation.** It is not evidence that Nathan's mechanism is physically lost before the objective until V0 has passed source-output parity.

## V2 Status

V2 is **not allowed** until V0 passes or has a documented, numerically converged partial mismatch. HWP/SLM sweeps, route-realism comparison, and F0/F1/F2 focus conclusions remain paused behind that gate.

## Required Stopping Result

The stopping outcome must state:

- V0 source-output parity status.
- V1 inherited pre-objective visual status, if V1 was generated.
- Whether V2 is allowed.

The required wording for V1, before V0 passes, is: "Current inherited pre-objective result under its present parameterisation."
