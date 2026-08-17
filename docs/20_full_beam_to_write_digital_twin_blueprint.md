# Full Beam-to-Write Digital Twin — Master Blueprint

**Stage:** 8A  
**Last updated:** 2026-06-18  
**Status:** Blueprint (documentation only — no numerical nonlinear physics implemented)

---

## Purpose

This document is the master architecture reference for the **full beam-to-write digital twin**: a
start-to-finish lab-planning simulator that takes laser and hardware parameters as input and
produces predicted fluence distributions, material-response proxies, and writing-trajectory
estimates as output.

The digital twin is built in stages.  This document defines the architecture, data flow, model
boundaries, and implementation plan.  It does not claim to be a complete implementation — each
claim is labelled with its model status (see `docs/21_digital_twin_model_status_and_claim_boundaries.md`).

---

## Design Principles

1. **Strict claim labelling.**  Every output carries a `model_status` tag that governs what may be
   claimed about it.  No output may be described as a *prediction* unless it has been validated
   against experiment.

2. **Layered engines.**  Three physically distinct engines are implemented in order of maturity.
   A downstream engine never depends on an output its upstream engine has not yet produced.

3. **No fake calibration.**  Calibrated-material outputs are only produced when real experimental
   calibration data is supplied.  Proxy outputs are always labelled as proxies.

4. **No giant simulator.**  Each engine is a thin wrapper around existing `bessel_twin_core` and
   `vbb_study` infrastructure.  No new numerical nonlinear physics is implemented until Stage 8F+.

5. **Additive.**  The digital twin layers on top of the existing study.  The characterisation locks,
   physics engine, and notebook structure are unchanged.

---

## Three-Engine Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT LAYER                                                        │
│  Laser params: pulse energy, rep rate, wavelength, beam waist       │
│  Hardware route: SLM pattern, axicon, objective, sample             │
│  Writing params: scan speed, shot overlap, trajectory geometry      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENGINE 1 — OPTICAL TRAIN                                           │
│  First-principles optical field propagation                         │
│  Sources: bessel_twin_core (ASM, BL-ASM, SLM encoding)             │
│  Outputs: U(x,y,z) at any plane; intensity; far-field               │
│  Model status: optical_prediction (validated by lock)               │
│  Claim level: spatial field, radial profile, core diameter          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  U(x,y), pulse_energy_J, area_m2
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENGINE 2 — EXPOSURE / WRITING / DOSE                               │
│  Energy accounting + fluence + multi-shot accumulation              │
│  Sources: vbb_study.equations (fluence, incubation)                 │
│  Outputs: F(x,y,z) [J/cm²], dose map, threshold-crossing volume    │
│  Model status: fluence_prediction → fluence_threshold_proxy         │
│  Claim level: spatial fluence distribution; threshold comparison    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  threshold_crossing_volume, F_map
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ENGINE 3 — MATERIAL RESPONSE / CALIBRATION                         │
│  Proxy material model + optional calibration data ingestion         │
│  Sources: vbb_study.equations.material (proxy); experiment data     │
│  Outputs: modification volume proxy; calibrated prediction (gated)  │
│  Model status: uncalibrated_material_response_proxy (default)       │
│               calibrated_material_prediction (requires cal. data)   │
│  Claim level: spatial extent proxy only until calibrated            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Engine 1 — Optical Train

### What it does

Engine 1 propagates a laser beam from the SLM plane through the full optical train to the sample
surface using the existing `bessel_twin_core` and `vbb_study` infrastructure.

### Inputs

| Parameter | Symbol | Source |
|---|---|---|
| Wavelength | λ | `laser.wavelength_m` |
| Beam waist at SLM | w₀ | `laser.beam_radius_on_slm_m` |
| Topological charge | ℓ | `target.ell` |
| Target core diameter | 2r₀ | `target.target_core_diameter_m` |
| Blaze period | Λ | `slm.blaze_period_px` |
| Objective NA | NA | `objective.NA` |
| SLM conjugate mode | — | `physical_axicon.slm2_conjugate_mode` |
| Refractive index | n | `material.refractive_index` |

### Outputs

| Output | Key | Model status |
|---|---|---|
| Complex field at surface | `surface_field` | `optical_prediction` |
| Transverse intensity map | `|U(x,y)|²` | `optical_prediction` |
| Longitudinal (x-z) intensity | `xz_intensity` | `optical_prediction` |
| Radial profile | `radial_profile` | `optical_prediction` |
| Axicon result (k_r, design) | `axicon_result` | `optical_prediction` |
| Validity flags | `validity_report` | `optical_prediction` |

### Model status: `optical_prediction`

The optical field is a first-principles prediction validated by the characterisation lock.  Claims
about core diameter, vortex ring radius, Bessel zone, and winding number are supported at this level.

### What is NOT modelled in Engine 1

- Temporal pulse shape / duration
- Nonlinear propagation in the optical path (Kerr, self-focusing)
- Polarisation-dependent SLM reflection
- SLM temporal phase flicker or pixel fill factor loss
- Aberrations beyond pupil clipping and interface phase step

---

## Engine 2 — Exposure / Writing / Dose

### What it does

Engine 2 converts the optical field from Engine 1 into a spatial fluence map and accumulates dose
for a multi-shot writing trajectory.

### Inputs

| Parameter | Symbol | Source |
|---|---|---|
| Pulse energy | E_p | `laser.pulse_energy_J` |
| Repetition rate | f_rep | `laser.rep_rate_Hz` |
| Scan speed | v | `writing.scan_speed_m_s` |
| Spot overlap | η | derived from v, f_rep, r₀ |
| Shot count per site | N | `writing.shots_per_site` |
| Threshold fluence (1-shot) | F_th(1) | `material.threshold_1shot_J_cm2` |
| Incubation exponent | S | `material.incubation_exponent` |

### Outputs

| Output | Key | Model status |
|---|---|---|
| Peak fluence | `peak_fluence_J_cm2` | `fluence_prediction` |
| Transverse fluence map F(x,y) | `fluence_map` | `fluence_prediction` |
| Longitudinal fluence map F(x,z) | `fluence_xz` | `fluence_prediction` |
| Threshold-crossing volume | `threshold_volume` | `fluence_threshold_proxy` |
| Incubation-adjusted threshold | `F_th_N` | `fluence_threshold_proxy` |
| Multi-shot dose accumulation | `dose_map` | `dose_accumulation_proxy` |
| Writing trajectory diagram | `trajectory_fig` | `dose_accumulation_proxy` |

### Model status hierarchy in Engine 2

- **`fluence_prediction`**: Peak fluence and spatial fluence are derived from the validated optical
  field using energy accounting.  Allowed claim: "predicted spatial fluence distribution."
- **`fluence_threshold_proxy`**: Threshold comparison uses literature values for F_th.  Allowed
  claim: "fluence exceeds literature threshold in this region."  Not allowed: claims about actual
  material modification.
- **`dose_accumulation_proxy`**: Multi-shot accumulation uses simple geometric overlap without
  heat accumulation or incubation coupling.  Allowed claim: "estimated shot overlap for this
  trajectory at these parameters."

### What is NOT modelled in Engine 2

- Heat accumulation between pulses
- Nonlinear energy deposition (multiphoton / avalanche scaling)
- Plasma dynamics
- Mechanical stress / cracking
- Material ejection / recast

---

## Engine 3 — Material Response / Calibration

### What it does

Engine 3 converts the dose and threshold output from Engine 2 into a material response estimate.
By default it produces an **uncalibrated proxy**.  When calibration data is supplied it upgrades to
a **calibrated material prediction**.

### Inputs

| Parameter | Symbol | Source |
|---|---|---|
| Threshold crossing map | — | Engine 2 output |
| Literature threshold | F_th | `material.threshold_1shot_J_cm2` |
| Calibration data (optional) | — | `calibration/` experiment CSV |

### Outputs

| Output | Key | Model status |
|---|---|---|
| Modification volume proxy | `modification_proxy` | `uncalibrated_material_response_proxy` |
| Simulated top-view image | `simulated_image` | `simulated_microscopy_proxy` |
| Calibrated prediction (gated) | `calibrated_result` | `calibrated_material_prediction` |

### Gating rule for calibrated outputs

Engine 3 only produces `calibrated_material_prediction` outputs when:
1. A calibration CSV is loaded from `calibration/` with measured threshold values.
2. The material and regime match the calibration data provenance.
3. The calibration data has a valid `calibration_source` field (not synthetic).

Without calibration data, all Engine 3 outputs carry `uncalibrated_material_response_proxy` and the
figures are stamped with "PROXY — NOT A CALIBRATED PREDICTION".

### What is NOT modelled in Engine 3

- Nonlinear absorption cross-sections
- Thermal diffusion or accumulation
- Phase transitions (melt, resolidification)
- Stress / crack propagation
- Long-term photodarkening or drift

---

## Data Flow Summary

```
laser + hardware + writing params
         │
         ▼
    Engine 1: bt.run_case(cfg, preset, path)
         │  surface_field, axicon_result, design
         ▼
    Engine 2: digital_twin.exposure.compute_fluence(field, laser, writing)
         │  fluence_map, dose_map, threshold_crossing_volume
         ▼
    Engine 3: digital_twin.material.compute_response(dose_map, calibration)
         │  modification_proxy  [or calibrated_result if cal data present]
         ▼
    Figure pipeline: digital_twin.figures.render_all(outputs)
         │
         ▼
    outputs/csv/digital_twin/   figures/digital_twin/
```

---

## Integration with Existing Study

The digital twin integrates with, but does not modify, the existing infrastructure:

| Component | Integration | Change? |
|---|---|---|
| `bessel_twin_core` | Engine 1 calls `bt.run_case` | None |
| `vbb_study.equations` | Engine 2 uses fluence/incubation equations | None |
| Characterisation lock | Lock tests remain unchanged | None |
| Notebook outputs | Digital twin figures go to `outputs/figures/digital_twin/` | Additive |
| CSV outputs | Go to `outputs/csv/digital_twin/` | Additive |
| Model status registry | New CSV registry | Additive |

---

## Directory Layout

```
Publication_Study/
├── vbb_study/
│   └── digital_twin/          ← NEW (Stage 8B+): engine implementations
│       ├── __init__.py
│       ├── exposure.py        ← Engine 2 (Stage 8C)
│       ├── material.py        ← Engine 3 (Stage 8D)
│       ├── figures.py         ← Figure pipeline (Stage 8E)
│       └── calibration.py     ← Calibration ingestion (Stage 8G)
├── notebooks/
│   └── digital_twin/          ← NEW (Stage 8E+): digital twin notebooks
├── outputs/
│   ├── figures/digital_twin/  ← NEW: digital twin figures
│   └── csv/digital_twin/      ← NEW: digital twin CSVs + model status registry
├── tests/
│   └── test_stage8a_blueprint_docs.py  ← NEW (Stage 8A): doc verification
└── docs/
    ├── 20_full_beam_to_write_digital_twin_blueprint.md   ← THIS FILE
    ├── 21_digital_twin_model_status_and_claim_boundaries.md
    ├── 22_digital_twin_figure_and_output_spec.md
    ├── 23_digital_twin_math_and_physics_stack.md
    └── 24_digital_twin_stage_plan.md
```

---

## Cross-References

| Topic | Document |
|---|---|
| Model status definitions and claim boundaries | `docs/21_digital_twin_model_status_and_claim_boundaries.md` |
| Figure and output specification | `docs/22_digital_twin_figure_and_output_spec.md` |
| Math and physics stack | `docs/23_digital_twin_math_and_physics_stack.md` |
| Implementation stage plan (8A–8J) | `docs/24_digital_twin_stage_plan.md` |
| Existing model limitations | `docs/04_model_limitations.md` |
| Existing hardware routes | `docs/06_hardware_routes.md` |
| Material proxy caveats | `docs/03_materials_application.md` |
