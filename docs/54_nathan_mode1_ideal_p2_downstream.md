# Nathan MODE 1 — Ideal P2 Field Through the Inherited Downstream Digital Twin

**Status:** MODE 1 only. No physical HWP/QWP/SLM realisation (MODE 2A/2B), no panel realism (MODE 3),
no robustness sweeps, no route ranking, no publication figures.
**Prerequisite:** V0 source-observable audit resolved (verdict A, docs/53).
**Outcome:** **M1-B** — the ideal field does **not** produce a visually convincing micro-scale hexagonal
beam; it survives as a **dark-core triangular / three-lobed (C3) structure**. **MODE 2A/2B remain
blocked.**

> An earlier revision of this stage reported M1-A ("hexagonal survives; MODE 2A/2B may begin"). That was
> **wrong**: the completion gate accepted a field on order-6 ring content alone, but a C3/triangular
> field carries a strong order-6 harmonic on a ring. A triangular-symmetry veto was added; the corrected
> classification is `triangular_lobed_field` and the outcome is M1-B.

## What MODE 1 tests

MODE 1 injects the *ideal* canonical Nathan six-sector `VectorField` at the common **P2** handoff plane
and propagates it through the **inherited** downstream Digital Twin geometry:

- **F0** (current bridge): vector axicon (thin, p/s Fresnel) → scalar per-component ObjectiveMap/focus
  bridge → vector ASM sample z-stack with `Ez`.
- **F2** (diagnostic): vector axicon → scoped vectorial pupil-spectrum reference at the same z values.

It **bypasses** patterned HWP, SLM1, relay, intermediate HWP, SLM2, final QWP, panel realism,
carrier/iris realism, HWP mosaics and waveplate errors. MODE 1 therefore represents the physical bench
**partially, downstream only; upstream physical generation is bypassed**. Every run writes
`simulation_scope_manifest.json`.

Entry point: `run_mode1_ideal_p2_downstream(...)` → `Mode1Result`. Notebook:
`notebooks/digital_twin/09_nathan_mode1_ideal_p2_downstream.ipynb`. Figures + manifest:
`outputs/figures/digital_twin/nathan_mode1_ideal_p2/`.

## Triangular-symmetry veto (why order-6 content is not enough)

`_mode1_symmetry(...)` + `mode1_symmetry_class(...)` classify each plane into exactly one of
`MODE1_SYMMETRY_CLASSES = (visual_hexagonal_field, triangular_lobed_field, dark_core_structured_field)`
from:

- **ring angular orders** (3, 6, 9, 12) and ratios — including `order3/order6`;
- **rotational self-similarity** of the central crop at 60°, 120°, 180° — the decisive discriminator: a
  hexagon has 60° self-similarity ≥ 120°; a C3/triangle has **120° > 60°**;
- **six-sector energy balance** on the ring — `max/min`, odd-vs-even (three-pair) imbalance;
- **connected bright-island count** on the ring (3 ⇒ triangular).

A plane is `visual_hexagonal_field` only if it is hollow **and** `order6 ≥ order3` **and** 60°
self-similarity ≥ 0.55 **and** `(c120 − c60) < 0.08` **and** six-sector `max/min < 1.6` **and** ≥ 5 ring
islands. It is `triangular_lobed_field` if `(c120 − c60) ≥ 0.08`, or `order3 ≥ order6`, or the six sectors
are strongly imbalanced, or there are exactly 3 ring islands. Order-6 ring content **alone never**
produces a hexagonal classification.

## Result (grid_n = 96–192)

At the declared reference plane (middle of the non-diffracting zone):

- `reference_symmetry_class` = **triangular_lobed_field** (grid_n ≥ 160; `dark_core_structured_field`
  right at the 128 borderline), with **≈ 55–73 % of planes triangular** and only ≈ 20–33 % hexagonal.
- `order3/order6 ≈ 0.36` (order-6 actually *exceeds* order-3), yet **120° self-similarity (≈ 0.83–0.90)
  exceeds 60° (≈ 0.70–0.72)** — a clear C3 signature, and the dark core is visibly triangular.
- Six-sector energies are roughly balanced (`max/min ≈ 1.1`), so this is a *mixed* C3+C6 field whose
  **dominant visual symmetry is triangular**, not a clean hexagon.
- Centre-treatment **robust** (dark core is real, not a P2 grid artefact); F0 vs F2 equal-power
  full-field correlation 0.94–0.97.

**z-dependence (honest observation, not cherry-picking):** the *early* formation planes (small z, near
the bright zone) classify closer to hexagonal, but the field **degrades to triangular** as it propagates;
the declared middle reference plane is triangular. This is the key lead for the next sensitivity study.

## Metrics — three separate families

`mode1_hexagonal_bessel_survival_metrics(...)` reports, with no single "best plane" selected:

- **`symmetry_classification`** — the C3-vs-C6 veto and the per-plane classes (this drives the outcome).
- **`source_like_hexagonal_bessel_survival`** — dark-core ratio, ring radius, sixfold/orientation
  stability, axial persistence (a dark, structured field *does* survive — but structure ≠ hexagon).
- **`clean_single_wall_usefulness`** — wall continuity / power (separate usefulness metric, not a gate).

`mode1_completion_gate(...)` suggests M1-A/B/C/D and sets `mode2_realisation_allowed`; **M1-A requires a
`visual_hexagonal_field` at the reference plane and on ≥ 50 % of planes.** The operator confirms the
final outcome.

## Outcome

**M1-B.** The ideal Nathan six-sector P2 field does not produce a visually convincing micro-scale
hexagonal beam under the inherited downstream Digital Twin geometry; the output is dominated by a
dark-core triangular / three-lobed (C3) structure. **Physical HWP/QWP/SLM realisation remains paused.**

Chosen M1-B (not M1-D) because a *structured dark-core field does survive* — it is the wrong symmetry
class, not an absence of structure. Not M1-A because it is not a hexagon.

## Next scientific step (not started here)

Do **not** proceed to physical HWP/QWP/SLM realisation. The next task is an **ideal MODE 1
geometry/parameter sensitivity study**: can the inherited/downstream geometry be adjusted, using only
physically meaningful existing parameters (effective k_r / cone angle, axicon parameter, beam
radius / pupil fill, relay magnification, sector rotation, apodisation/envelope, z-reference), so the
ideal P2 field becomes `visual_hexagonal_field` rather than `triangular_lobed_field`? Each candidate must
be classified by the same three-way veto; order-6 / H6 content must not select the winner by itself.

## Scope reminder

MODE 1 does **not** simulate HWP/QWP/SLM generation, panel realism, carrier/iris, robustness sweeps or
route ranking. The patterned-HWP and serial-SLM implementations are preserved for MODE 2A/2B. MODE 2A/2B
and MODE 3 remain **blocked** under this M1-B classification.
