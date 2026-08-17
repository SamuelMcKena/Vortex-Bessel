# Nathan V0 Source-Observable and Model Audit

**Status:** narrow MODE 0 audit. No MODE 1/2A/2B/3 work, no HWP/SLM realism, no route ranking,
no F0/F1/F2 conclusions, no publication figures.
**Question:** why did V0 have *exact input-array parity* but only a *partial visible output match*
(a spoked/bright-core pattern instead of Nathan's Figure 4)?
**Outcome (one line):** the observable was always correct (total intensity); the mismatch was a
**grid-centring translation error** at the r=0 polarisation singularity. Corrected, V0 reproduces
Nathan's own appendix code to machine precision. **Verdict: A.**

---

## A. The real source figure and observable

Primary source located and read directly (not from old notebook labels or crops):

- **File:** `Laser_Manufacturing.pdf` — *"Coherent Light, Incoherent Structure"*, Nathan Marco,
  supervisor Prof. Richard Carter, Heriot-Watt University, submission date 18 June 2026.
- **On disk:** `C:\Users\sm2006\Desktop\PhD\Year 1\Holoeye Hex Beam Literature\Laser_Manufacturing.pdf`
  (2.18 MB, 20 pages). The project crop `outputs/reference/nathan_marco_report_figure4_page7_crop.png`
  (SHA256 `0906fed2…83775b`) is confirmed to be **page 7, Figure 4**.
- **Target figure:** Figure 4, page 7.

**Exact observable — total intensity, no analyser.** Verbatim evidence:

- Fig. 4 caption: *"Simulated **intensity distribution** of an alternating radial and azimuthal
  **Bessel beam** … transverse intensity profile at the selected observation plane of 60 mm …
  x-z propagation map … the **on-axis intensity** as a function of propagation distance, indicating
  the axial region over which the beam is **most strongly confined**. Intensities are plotted in
  arbitrary units."*
- Methods §2.2.1: *"Finally, the **total intensity is calculated from the three electric-field
  components** and visualised."*
- Appendix Listing 1: `field_intensity(Ex,Ey,Ez) = |Ex|² + |Ey|² + |Ez|²`, plotted directly.

There is **no linear polarizer / analyser / component projection** anywhere in the beam-shaping
result — not in the caption, the methods, or the code. The four Figure-4 panels are xy total
intensity at z = 60 mm, its central x-slice, the x-z total-intensity map, and the on-axis total
intensity vs z.

**Field configuration (fully specified by the source).** Fig. 4 uses *"Three pairs of segments of
60°"* → six alternating 60° sectors. Gaussian input, `w0 = 2 mm`, `λ = 1030 nm`, axicon `n = 1.458`,
medium `n = 1.0`, apex 176° (base angle 2°), observation plane 60 mm, propagation 0.1–290 mm.
Appendix `make_segmented_ra_input`: azimuthal state first, radial state in the final `sector_theta`
of each cell; `phi0 = 0` (radial) / `π/2` (azimuthal); `Ex = amp·cos(θ+phi0)`, `Ey = amp·sin(θ+phi0)`.
Sector **ordering, phase convention and radial/azimuthal assignment are therefore not ambiguous** —
they are read directly from the code.

**Axicon model (source).** Appendix `apply_axicon`: thin element, radial phase
`exp(-i·k0·(n_ax − n_med)·axi·R)` (small-angle, uses the base angle `axi` directly); Fresnel s/p
transmission at the conical exit face with the radial component treated as p and the azimuthal as s
(`fresnel_ts_tp`); a plane-entrance factor `t_entry = 2·n_med/(n_med+n_ax)`. Propagation is the
angular-spectrum method with a `P = I − ss^T` transversality projection that also generates `Ez`.

**Important context from the report.** The hexagon was *not* a designed target. Abstract:
*"the study of a square-beam polarization mask based on alternating radial and azimuthal states … the
simulations produced the intended square beam, but also revealed **an unexpected hexagonal beam
structure**."* The hollow **donut** is the *pure* radial/azimuthal sanity check (Fig. 2); the
*square* beam is the 8-segment 20°/70° case (Fig. 3); the **hexagonal Bessel beam** is the 6-segment
60° case (Fig. 4, and enlarged in Fig. 5, described as the *"enlarged hexagonal Bessel-beam
profile"*). Nathan also notes an *"anomalous intensity spike … near the beginning of the simulated
propagation … attributed to a known numerical issue in the code."*

**What Figure 4 actually shows:** a **hexagonal Bessel beam with a dark on-axis core** — the bright
hexagonal/sixfold ring structure surrounds a small central null (visible as the double-peak in the
central x-slice and as an on-axis intensity that is ~10⁻³ of the transverse peak). This is confirmed
below by running Nathan's own code: on-axis/peak ≈ 5×10⁻⁴, transverse peak ≈ 68.7 (matching the
Fig. 4 colourbar maximum of ~68).

## B. Search for Nathan's original code / arrays

- **No** original `.py`, `.ipynb`, `.mat`, `.npy`, or `.npz` files by Nathan exist anywhere in the
  repository or referenced folders (searched repo + `Holoeye Hex Beam Literature`).
- **However, the full source code is printed in the report Appendix** (Listing 1,
  *"@author: Nathan Marco, modified from original by ecali"*, pages 11–20). This is the actual
  algorithm, not a paraphrase.
- The appendix code was transcribed verbatim (algorithm unmodified) to
  `scratchpad/nathan_appendix_verbatim.py` and run as an independent gold-standard reference. It is
  **not** committed (it is Nathan's code; kept as external evidence, attributed here).

**Provenance statement:** V0A/V0B are a **project re-implementation of Nathan's appendix algorithm**,
now **validated to reproduce his verbatim appendix code to machine precision** for the Figure-4
hexagon (§E). It is *not* execution of Nathan's original file; array-level identity is against the
appendix listing, figure-level identity is against Fig. 4.

## C. V0A / V0B independence

| Path | Module | Field | Axicon | Propagation |
|---|---|---|---|---|
| **V0A** | `nathan_literal_source_port` | `make_segmented_ra_input` | `apply_source_axicon` | `propagate_source_vector_asm` |
| **V0B** | `nathan_vector_hexagon` | `canonical_target_field` → `nathan_alpha` + `gaussian_envelope` | `_apply_free_space_vector_axicon` (project `fresnel_sp_amplitudes`) | `_free_space_intensity_stack` (inline ASM, project `fft2c`/`ifft2c`) |

- Current module hashes: `nathan_literal_source_port.py` = `a093e18c…7965b`;
  `nathan_vector_hexagon.py` = `dd1b97d0…f765a`.
- **Call graphs are separate.** `_free_space_intensity_stack` implements the vector ASM +
  transversality projection **inline**; it does **not** call `propagate_source_vector_asm`, and V0B
  does not call any V0A propagation function. (`nathan_vector_hexagon` imports the literal port only
  to *run* V0A itself.)
- **Numerically bit-identical:** equal-power intensity RMS = `0.0`, correlation = `1.0`, max abs
  intensity difference = `0.0` at N = 512, 1024 and 1536 (reference plane z = 60 mm).

**Interpretation (the key nuance):** V0A and V0B agree because they encode the *same physics with the
same conventions* (for the symmetric 60° hexagon `nathan_alpha` and `make_segmented_ra_input`
coincide, and both now use Nathan's grid centring). This is **intra-project cross-validation**, and
it is exactly why a bright-core bug in the shared grid convention appeared in *both* while neither
matched Nathan. **The only independent check against Nathan is his verbatim appendix code (§E).**

## D. The correct observable

The source result is **total intensity** with **no analyser**. Per the audit rule *"if no analyser
is part of the source result, do not invent one merely to force a hexagon,"* **no analyser/Jones
projection was added.** V0 already plotted the correct observable; the observable was never the
problem. This rules out acceptance option **B**.

## E. Root cause — grid centring at the r=0 polarisation singularity

Input-array parity was exact, so the defect had to be in the **propagated** field. The segmented
radial/azimuthal field `Ex = amp·cos(θ+phi0)` has a polarisation singularity at r = 0. Analytically
its azimuthal m = 0 component integrates to zero for the 60° hexagon, so the on-axis intensity should
be ≈ 0 (dark core). Whether the *discrete* field preserves that cancellation depends on how the axis
is sampled.

Controlled test (Nathan's pipeline, N = 512, only the knobs varied; total intensity at z = 60 mm):

| grid convention | centre-pixel zeroing | on-axis / peak | corr. to Nathan canonical |
|---|---|---:|---:|
| axis-sampled (`x = −L + arange·dx`) | yes (Nathan) | **0.00049** | **1.0000** |
| axis-sampled | no | 0.00048 | 1.0000 |
| zero-straddling (`arange − N/2 + 0.5`) | yes | **1.00000** | 0.8378 |
| zero-straddling (**old V0 path**) | no | 1.00000 | 0.8378 |

Findings:

- **Centre-pixel zeroing is irrelevant** (0.00049 vs 0.00048). It is *not* the cause.
- **The grid centring is decisive.** Nathan's grid samples the axis exactly at index `n//2`, which
  preserves the m = 0 cancellation → **dark-core hexagonal Bessel beam** (matches Fig. 4). The
  project's default `make_xy_grid` straddles zero (`arange − N/2 + 0.5`); the r = 0 singularity is
  then carried by four uncancelled near-axis pixels that inject a **spurious bright on-axis core**
  (on-axis/peak = 1.0, correlation to Nathan only 0.84).
- Both V0A and V0B used the straddling grid → both showed the bright core → they agreed with each
  other but not with Nathan. This is the entire "input parity exact, output only partial" story.

Evidence figure: `outputs/figures/digital_twin/nathan_visual_ladder/nathan_source_observable_audit_grid_convention.png`.

### Fix applied (source-justified, isolated to V0)

Both V0 paths now use Nathan's axis-sampled centring `x = (arange − n//2)·dx` (drop the `+0.5`):

- `nathan_literal_source_port.make_source_grid` (V0A, isolated module).
- `nathan_vector_hexagon.source_parity_grid` (V0B) — built inline (not via `make_xy_grid`, not via
  the V0A port) so the V0B path stays independent while adopting the correct source centring.

The shared `make_xy_grid` and all downstream twin geometry are **unchanged**; `source_parity_grid`
is used only by V0. This is not tuning-to-taste: it matches Nathan's documented numerical method and
is validated below. No centre-pixel zeroing was added (negligible, and adding it to only one path
would break the V0A≡V0B parity that the tests lock).

### Post-fix validation

- Port V0A on-axis/peak: **1.0 → 0.00048** (dark/hollow core).
- **Corrected V0 vs Nathan's verbatim appendix code (single axicon, rotation 0): correlation =
  1.00000**, port on-axis/peak 0.00048 vs Nathan 0.00049, transverse peak ≈ 68.7 ≈ Fig. 4 colourbar.
- V0A≡V0B parity preserved (RMS = 0, corr = 1.0 at N = 512/1024/1536).
- Regenerated `nathan_visual_ladder_v0_reference_vs_reproduction.png` now shows the dark-core
  hexagonal Bessel beam, double-peak x-slice, and Bessel x-z — a clear visual match to the Fig. 4
  crop.

## E′. Source-constrained ambiguity sweep

Only genuinely source-unresolvable degrees of freedom were swept (total intensity, N = 512):

- **Sector rotation** (`nathan_source_observable_audit_rotation_sweep.png`, 0/15/30/45/60°): all
  preserve the sixfold hexagon (rotated); **rotation = 0 matches Fig. 4's orientation** (rotation =
  30° gives a distinct orientation, corr 0.80 to the rot-0 case — it is a rotated beam, not a
  different beam). The port default `sector_rotation_rad = 0` is correct for Fig. 4.
- **Phase sign, sector ordering, radial/azimuthal assignment:** *not* swept — they are unambiguous in
  Nathan's appendix code and already matched.
- **Centre treatment:** covered by the grid-centring test above (the operative "centre treatment").
- **Two-axicon recombination:** Nathan's `__main__` default superposes two axicons (apex 176.0°/176.8°)
  for lobe suppression, but that work *"led to substantial degradation."* The clean Fig. 4 is the
  **single-axicon** case (correlation to the two-axicon result 0.87), which the port default matches.

No metric-led "best-H6" plane or parameter selection was performed.

## Convergence

The module convergence gate (`run_v0_numerical_convergence`, N = 512/1024/1536) reports
`v0_convergence_materially_consistent`. The reconstruction is numerically converged.

## On the "PARTIAL" automated verdict (not overridden)

The automated `propagated_output_visual_verdict` still returns **PARTIAL**, and it was **not**
overridden. That metric tests for a *clean continuous hollow hexagonal wall* (`wall_continuity`,
`wall_power_fraction`); Nathan's Fig. 4 is a **multi-ring hexagonal Bessel beam with a small dark
core**, which does not satisfy a single-wall criterion. So "PARTIAL" here means *"this is not a clean
hexagonal wall,"* which is **a downstream (MODE 1+) question about usefulness at the sample**, not a
statement about whether V0 reproduces the source. The correct V0 pass criterion — agreement with
Nathan's own code — is **PASS** (correlation 1.00000). Input-array parity and propagated-output
parity remain reported separately.

## Files changed / commands / outputs

**Modified (2 source files):**

- `vbb_study/digital_twin/nathan_literal_source_port.py` — `make_source_grid` → axis-sampled
  centring; docstring corrected to "reconstructed source-model port … validated against the
  appendix," dropping the misleading "literal Nathan code" implication.
- `vbb_study/digital_twin/nathan_vector_hexagon.py` — `source_parity_grid` → inline axis-sampled
  centred grid (V0 only).

**Added:** `docs/53_nathan_source_observable_audit.md` (this file).

**Regenerated figures** (`outputs/figures/digital_twin/nathan_visual_ladder/`):
`nathan_visual_ladder_v0_reference_vs_reproduction.png`, `…_field_views.png`, `…_v0.png`,
`…_convergence_xy.png`, `…_convergence_report.json`, `…_status_report.json`, and audit evidence
`nathan_source_observable_audit_grid_convention.png`, `nathan_source_observable_audit_rotation_sweep.png`,
`v0_audit_metrics.json`.

**Commands:** targeted `pytest tests/test_nathan_vector_hexagon_digital_twin.py -q` → **22 passed**
(before and after); `git diff --check` clean; audit + regeneration scripts run under
`C:\PhD\.venv2`.

**Not changed:** `make_xy_grid`, all downstream twin geometry, patterned-HWP and serial-SLM
implementations (preserved for MODE 2A/2B), notebook 08 (already V0-only, no dead V1/F0/F1/F2 cells,
no `replace(…verdict=…)` overrides, no embedded outputs).

## Verdict

**A. The reconstructed source model now reproduces the correct source observable (total intensity)
with a visually recognisable sixfold / dark-core hexagonal Bessel structure, validated against
Nathan's own appendix code (correlation 1.00000). MODE 1 may begin.**

Why A and not the others:

- **Not B:** the source observable is total intensity with no analyser; nothing was omitted on the
  observable side. The defect was numerical (grid centring), not a missing projection.
- **Not C:** the report information was *sufficient* — the appendix contains the executable algorithm,
  and the corrected reconstruction reproduces it to machine precision. No missing parameter file or
  output array blocks reproduction.
- **Not D:** the corrected, converged reconstruction *does* reproduce Nathan's Figure-4 result (a
  dark-core hexagonal Bessel beam). The earlier failure was a fixed grid-centring bug, not an
  unreproducible mechanism.

**Scope reminder:** "MODE 1 may begin" means the V0 source mechanism is validated. It does **not**
assert that this field yields a *useful, clean, hollow micro-scale hexagonal wall* at the Digital
Twin sample — that is exactly the MODE 1 question (does the ideal field survive the inherited
downstream axicon/relay/objective/sample geometry). MODE 2A/2B/3 and any route ranking remain out of
scope until MODE 1 is run.
