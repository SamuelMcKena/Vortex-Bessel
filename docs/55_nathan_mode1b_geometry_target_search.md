# Nathan MODE 1B - Geometry Target Search

**Status:** MODE 1B only. No physical HWP/QWP/SLM realisation, no panel realism, no 4F carrier/iris
realism, and no publication-final route ranking.

**Question:** can an ideal downstream geometry produce a micro-scale version of the validated V0 Nathan
hexagonal Bessel total-intensity field from the ideal P2 six-sector vector field?

**Plane-radius audit update:** see `docs/56_nathan_mode1b_ring_count_plane_audit.md`. The original
inherited ring-count value `0.033` mixed a sample-plane radius with a pre-axicon `k_r`; it is retained
only as a labelled failed diagnostic. Corrected MODE 1B uses plane-labelled P2/axicon radii.

## Guardrails

We are not trying to make a triangle. A triangular or three-lobed dark-core field is the inherited
failure mode, not the target.

Current inherited MODE 1 remains a triangular/C3 failure at the micro-scale sample. Order-6/H6 content
alone is not a visual hexagon because a C3 triangular field can carry a strong order-6 ring harmonic.

MODE 1B therefore compares candidates against the validated V0 target template with:

- total intensity only, `|Ex|^2 + |Ey|^2 + |Ez|^2`;
- the same C3-vs-C6 visual classifier used by MODE 1;
- a scale/rotation-invariant angular and profile comparison to the V0 template;
- an explicit feasibility report for each candidate.

## Ring Count Gate

The diagnostic ring count is

```text
n_rings = beam_radius_m * abs(k_r_m_inv) / (2*pi)
```

For a thin axicon diagnostic,

```text
k_r ~= (2*pi / wavelength_m) * (n_axicon - n_medium) * tan(base_angle_rad)
```

The validated V0 source has many effective rings and classifies as `visual_hexagonal_field`. The
near-current inherited micro-scale geometry has very low ring count and remains triangular/lobed.
MODE 1B reports the V0 ring count, the inherited estimate, and every candidate ring count.

## Feasibility Split

A candidate is not realistic merely because it looks hexagonal.

Each candidate gets a `Mode1BFeasibility` report with objective NA, phase period, SLM-pixel feasibility,
physical-axicon candidate status, and a class:

- `physically_plausible_existing_model`
- `exploratory_high_angle_redesign`
- `not_feasible_or_not_useful`

The exact wording matters:

- **existing architecture candidate** means the candidate fits the current-model feasibility class.
- **ideal exploratory redesign candidate** means the candidate may show useful physics, but it is outside
  current-lab realism or outside the validated thin/paraxial regime.

If only high-angle candidates work, physical realisation is not approved for the current lab
architecture.

## Outcomes

MODE 1B chooses exactly one outcome:

- `M1B-A-realistic`: an existing-architecture candidate passes the visual/template/feasibility gate.
  MODE 2A/2B may begin for that configuration.
- `M1B-A-exploratory`: an ideal exploratory redesign candidate passes visually, but current-architecture
  MODE 2A/2B remain blocked.
- `M1B-B`: structured fields appear, but they remain triangular/lobed or non-hexagonal.
- `M1B-C`: search inconclusive due to numerical or sampling limitations.
- `M1B-D`: tested downstream design space cannot produce the target.

## Outputs

Notebook:

- `notebooks/digital_twin/10_nathan_mode1b_geometry_target_search.ipynb`

Output directory:

- `outputs/figures/digital_twin/nathan_mode1b_geometry_search/`

Required artifacts:

- `mode1b_target_template.png`
- `mode1b_current_inherited_failure.png`
- `mode1b_search_summary.csv`
- `mode1b_search_summary.json`
- `mode1b_shortlist_<candidate_id>.png`
- `mode1b_outcome_report.json`
- `simulation_scope_manifest.json`

## MODE 2A/2B Gate

HWP/QWP/SLM physical realisation stays blocked unless the outcome is `M1B-A-realistic`.

`M1B-A-exploratory` is scientifically useful, but it only authorises a redesign study. It does not
authorise current-lab physical realisation.
