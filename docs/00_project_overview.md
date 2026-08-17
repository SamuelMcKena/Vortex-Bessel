# Project Overview — Structured-Beam Simulation Atlas

**Internal project name:** VBB Study / Structured-Beam Atlas
**Folder name:** `Publication_Study/` (kept for compatibility)
**Last updated:** 2026-06-03

---

## What This Project Is

This is a structured-beam simulation atlas for a Yb:KGW / PHAROS-class ultrafast
laser system.  The central theme is:

> **Ideal mathematical beam target vs. lab-realistic implementation** —
> for every beam family, compare the clean analytic ideal with what
> is actually achievable through a specific hardware route, and record
> the gap explicitly.

The atlas is not only a publication-output pipeline.  It is a reference
simulation library, a hardware-route comparison tool, and a planning
resource for future experiments.  The "publication" label in old file
names reflects its origin; the project is now broader.

---

## Core Research Aim

Build a complete simulation atlas that:

1. Defines the ideal mathematical target for each beam family.
2. Models lab-realistic implementations (holographic SLM, physical axicon,
   objective-pupil clipping, interface corrections).
3. Propagates each beam through the relevant optical path.
4. Evaluates standardised metrics (core size, Bessel zone, strict region,
   vortex ring diameter, power drift, symmetry).
5. Connects each beam family to real hardware routes and records which
   routes are currently available vs. proposed.
6. Provides material-response planning proxies for Cr:ZnSe and related
   materials (explicitly labelled as proxies, not calibrated predictions).

---

## Beam Families Covered

| Beam family | Status | Notes |
|---|---|---|
| Scalar Bessel-Gauss (ell=0) | Active, validated | Core study branch |
| Vortex Bessel-Gauss (ell>0) | Active | Topological charge 1–4 |
| Vector / radial-azimuthal | Active (modelled) | Lab feasibility flagged explicitly |
| Holographic SLM axicon route | Active | First-order filtering included |
| Physical refractive axicon route | Active | Mask-separation modelled |
| Through-sample / interface | Active | Correction labelled as ideal |
| Polygonal / hexagonal | Active | Stability claims require metric support |
| Discrete N-fold beams | Active | Phase-only approximation noted |
| Material proxy (Cr:ZnSe) | Planning proxy | Not calibrated |
| Capsule / weld feature design | Planning proxy | Geometry only |

---

## What This Project Does Not Do

- It does not claim every ideal target has a current lab implementation.
  Hardware status is explicitly recorded in every output's metadata.

- It does not convert threshold maps into calibrated ablation, melt, void,
  crack, stress, or refractive-index modification predictions.
  Material outputs are planning proxies until calibrated by experiment.

- It does not model high-NA vectorial focusing.  The scalar paraxial
  approximation is used throughout except where the vector Jones model
  is explicitly invoked.

- It does not import from `reference_kernels/`.  Those files are
  provenance snapshots of older code; they are not active modules.

---

## Study Architecture

```
Source of truth:       .py modules
  bessel_twin_core.py  — scalar physics engine (130 KB)
  vbb_study/           — helper package (equations, metrics, viz, taxonomy)

Study notebooks:       notebooks/
  Organised by topic: scalar/, lab_realism/, vector/, materials/, advanced/
  Import from .py modules; generate figures, tables, holograms.

Documentation:         docs/
  Theory, conventions, validation, taxonomy, hardware routes, limitations.

Generated outputs:     outputs/
  figures/, csv/, holograms/, manifests/ — generated only, not source.
```

The rule is: **equations and physics live in `.py` modules, not in
notebooks**.  If a notebook defines a reusable equation, that equation
belongs in `vbb_study/equations/`.

---

## Key Terms

| Term | Meaning |
|---|---|
| study | This simulation atlas as a whole |
| atlas | Broad parameter / beam-family output set |
| publication | Final paper-ready export / report (subset of outputs) |
| target | Ideal mathematical field definition |
| lab-realistic | Modelled with hardware apertures, SLM encoding, first-order filtering, etc. |
| planning proxy | Model output not calibrated against measured experiment |
| strict Bessel region | Range where peak power and core radius remain within tolerance |
| canonical zone | Axial FWHM of the on-axis intensity |

See `docs/01_conventions.md` for all metric definitions and units.
See `docs/05_study_taxonomy.md` for standard label values.

---

## Navigation

| Want to... | Go to |
|---|---|
| Understand the optical model | `docs/00_theory.md` |
| Look up a metric definition | `docs/01_conventions.md` |
| Check validation record | `docs/02_validation.md` |
| Understand material proxy limits | `docs/03_materials_application.md` |
| Check vector hardware status | `docs/04_actual_lab_vector_case1.md` |
| Use standard taxonomy labels | `docs/05_study_taxonomy.md` |
| Understand hardware routes | `docs/06_hardware_routes.md` |
| Know the model limitations | `docs/04_model_limitations.md` |
| Run the study | `docs/09_running_the_study.md` |
| Understand this refactor | `docs/08_refactor_plan.md` |
