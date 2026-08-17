# Digital Twin — Model Status and Claim Boundaries

**Stage:** 8A  
**Last updated:** 2026-06-18  
**Status:** Blueprint (documentation only)

This document defines the **model status registry** for all outputs produced by the digital twin.
Every figure, table, and CSV output must carry one of the statuses listed here.  The status governs
what claims may be made about the output in papers, reports, and internal documents.

---

## Why Model Status Matters

Simulation outputs span a wide range of epistemic confidence:

- A first-principles optical field validated against the characterisation lock is a genuine
  *prediction* of the spatial intensity distribution.
- A fluence map computed from that field with an assumed pulse energy is a *prediction* of the
  spatial fluence — but only if the energy accounting is correct.
- A threshold-crossing map that compares predicted fluence to a *literature* threshold is a
  *proxy* — it tells you where the fluence exceeds the number in a paper, not where the material
  actually modifies.
- A modification-volume estimate that mixes proxy threshold with proxy dose accumulation is a
  *planning proxy* only.

Every output in this study must carry an explicit label so that readers (and future-you) know
exactly which category applies.  Upgrading a status requires documented validation evidence.

---

## Status Hierarchy

Statuses are listed in order from most to least trustworthy.  An output carries the status of its
*weakest* input.

| # | Status key | One-line description |
|---|---|---|
| 11 | `experimentally_validated_prediction` | Direct comparison to measured experiment; full validation |
| 10 | `calibrated_material_prediction` | Calibrated threshold from experiment; validated for one material/regime |
| 9 | `simulated_microscopy_proxy` | Proxy modification volume convolved with PSF; no calibration |
| 8 | `uncalibrated_material_response_proxy` | Proxy models combined; no experiment calibration |
| 7 | `thermal_accumulation_proxy` | Heuristic thermal buildup; uncalibrated |
| 6 | `nonlinear_deposition_proxy` | Heuristic nonlinear absorption hook; uncalibrated |
| 5 | `dose_accumulation_proxy` | Geometric multi-shot accumulation; no heat or incubation coupling |
| 4 | `fluence_threshold_proxy` | Fluence compared to literature threshold; not calibrated |
| 3 | `fluence_prediction` | Spatial fluence from validated optical field + energy accounting |
| 2 | `energy_accounting_prediction` | Pulse energy → peak fluence; analytical formula |
| 1 | `optical_prediction` | First-principles optical field; validated by characterisation lock |

---

## Status Definitions

### 1 — `optical_prediction`

**Definition:** The output is a first-principles calculation of the complex optical field using the
scalar paraxial ASM/BL-ASM propagator.  The propagator parameters are validated against the
characterisation lock (`tests/test_characterisation_lock.py`).

**Allowed claims:**
- Predicted spatial intensity distribution |U(x,y,z)|²
- Predicted core diameter, ring diameter, Bessel zone length, and strict region
- Predicted vortex winding number (topological charge)
- Predicted radial profile shape
- Predicted first-order selection fraction (for holographic route)

**Not allowed:**
- Claims about actual energy deposition in the sample
- Claims about material modification
- Claims about absolute fluence without energy accounting

**Required label:** None required for purely optical outputs.  When combined with other model levels
the weaker status governs.

**Validation:** Characterisation lock (`tests/test_characterisation_lock.py`).  Must pass before
any optical output is published.

---

### 2 — `energy_accounting_prediction`

**Definition:** The output derives peak fluence from the validated optical field using the analytical
energy-accounting formula (see `docs/23_digital_twin_math_and_physics_stack.md`, Section 1).
This formula is closed-form and its derivation is cross-checked against the existing
`vbb_study.equations` implementation.

**Allowed claims:**
- Predicted peak fluence at the surface plane for a given pulse energy
- Predicted fluence scaling with pulse energy

**Not allowed:**
- Claims about spatial fluence distribution beyond the peak
- Claims about threshold crossings without threshold data
- Claims about material modification

**Required label:** "Energy accounting prediction."

---

### 3 — `fluence_prediction`

**Definition:** The full spatial fluence map F(x,y) is computed from the validated field |U(x,y)|²
normalised to the total pulse energy.  This is a genuine prediction of the spatial fluence
distribution at the sample surface, assuming perfect energy accounting and no losses beyond
those explicitly modelled (pupil clipping, first-order efficiency, SLM efficiency).

**Allowed claims:**
- Spatial fluence distribution at the sample surface (for the specified pulse energy and route)
- Ratio of peak to mean fluence
- Threshold-normalised fluence ratios (F/F_th) if threshold is labelled as literature value

**Not allowed:**
- Claims about energy deposition inside the sample (requires propagation into material)
- Claims about modification without threshold data

**Required label:** "Fluence prediction (optical_prediction + energy accounting)."

---

### 4 — `fluence_threshold_proxy`

**Definition:** The predicted fluence map is compared to a *literature* threshold value for the
material.  The threshold has not been calibrated for this system, laser, or material batch.
This is a planning proxy for parameter space exploration.

**Allowed claims:**
- "The predicted fluence exceeds the literature threshold of X J/cm² in region Y."
- "For pulse energy E_p, the threshold-crossing area is approximately Z μm²."
- "The overlap ratio F/F_th_lit is R."

**Not allowed:**
- Claims that the material is actually modified in that region
- Claims about the absolute threshold for this system
- Claims about modification morphology (void, crack, melt, etc.)

**Required label on figure:** "PLANNING PROXY — literature threshold (uncalibrated for this system)."

---

### 5 — `dose_accumulation_proxy`

**Definition:** Multi-shot dose accumulation is computed by geometric overlap of the fluence map
along the writing trajectory.  Heat accumulation, incubation coupling, and plasma effects are
not modelled.  Shot-by-shot fluence maps are simply added.

**Allowed claims:**
- "Estimated cumulative fluence for N shots at this trajectory."
- "Estimated shot-overlap fraction for this scan speed and rep rate."
- "Estimated dose distribution for this writing geometry."

**Not allowed:**
- Claims about actual energy deposited per unit volume
- Claims about heat buildup or thermal effects
- Claims about modification morphology

**Required label on figure:** "DOSE PROXY — geometric accumulation only, no thermal or incubation coupling."

---

### 6 — `nonlinear_deposition_proxy`

**Definition:** A heuristic hook that scales the single-shot fluence by an intensity-dependent
factor to approximate multiphoton / avalanche contributions.  The scaling law and exponent are
not calibrated.  This is a placeholder for future calibrated nonlinear models.

**Allowed claims:**
- "Heuristic estimate of enhanced central-peak deposition assuming M-photon scaling."
- "Qualitative comparison of linear vs. M-photon dose distributions."

**Not allowed:**
- Quantitative claims about deposition depth or volume
- Claims about actual threshold modification

**Required label on figure:** "NONLINEAR HEURISTIC PROXY — exponent and prefactor uncalibrated."

---

### 7 — `thermal_accumulation_proxy`

**Definition:** A heuristic hook that models pulse-to-pulse heat buildup as an exponential
thermal tail added to each shot's contribution.  The thermal diffusion length and coupling
coefficient are not calibrated.

**Allowed claims:**
- "Qualitative estimate of shot-to-shot thermal buildup at this rep rate and scan speed."
- "Relative comparison of thermal accumulation for different scan parameters."

**Not allowed:**
- Quantitative temperature predictions
- Claims about heat-affected zone or thermal damage extent

**Required label on figure:** "THERMAL HEURISTIC PROXY — diffusion length and coupling uncalibrated."

---

### 8 — `uncalibrated_material_response_proxy`

**Definition:** Output combines threshold proxy, dose accumulation proxy, and (optionally)
nonlinear and thermal proxies.  This is the default Engine 3 output when no calibration data
is supplied.  It provides spatial extent of the *predicted modification region* in the sense of
"where the dose exceeds the literature threshold" — nothing more.

**Allowed claims:**
- "Proxy modification volume: region where dose exceeds literature threshold."
- "Estimated feature geometry for planning purposes (not a calibrated prediction)."

**Not allowed:**
- Claims about actual modification morphology (void, melt pool, crack, waveguide)
- Claims about refractive index change
- Claims about modification visibility or contrast in any measurement

**Required label on figure:** "PROXY — NOT A CALIBRATED MATERIAL PREDICTION."

---

### 9 — `simulated_microscopy_proxy`

**Definition:** The proxy modification volume is convolved with a simplified optical transfer
function (OTF) or PSF to simulate a top-view or side-view microscopy image.  Neither the OTF
nor the modification contrast has been calibrated.

**Allowed claims:**
- "Simulated top-view image at diffraction limit: qualitative spatial extent."
- "Estimated resolvability of feature at these beam parameters."

**Not allowed:**
- Quantitative comparison to measured microscopy images without calibration
- Claims about actual image contrast or intensity

**Required label on figure:** "SIMULATED MICROSCOPY PROXY — OTF and contrast uncalibrated."

---

### 10 — `calibrated_material_prediction`

**Definition:** Engine 3 output produced when a calibrated threshold CSV is loaded from
`calibration/` directory.  The threshold F_th(1) and incubation exponent S have been measured
for this specific material / laser combination.  Outputs are validated for the material and
regime described in the calibration file.

**Conditions for upgrade to this level:**
- `calibration/` CSV exists with `calibration_source` field set to a measured data reference.
- The material, wavelength, pulse duration, and NA match the calibration provenance.
- The calibration has been reviewed and approved (tracked in calibration provenance record).

**Allowed claims:**
- "Calibrated threshold crossing map for [material] at [regime]."
- "Predicted modification volume for [material] using calibrated F_th = X ± Y J/cm²."

**Not allowed:**
- Transferring calibration to a different material without re-calibration
- Claims about morphology (requires additional characterisation)

**Required label on figure:** "CALIBRATED PREDICTION — [material], [regime], calibration ref: [ID]."

---

### 11 — `experimentally_validated_prediction`

**Definition:** A direct comparison between digital twin output and measured experiment has been
carried out.  The comparison covers the specific material, regime, and parameter range.  Residuals
are documented.

**Conditions for upgrade to this level:**
- Calibrated prediction (status 10) exists.
- Measured experimental data (microscopy images, calibration curves, or profiles) has been
  compared to the prediction with quantified residuals.
- Validation is documented in a `validation/` provenance record.

**Allowed claims:**
- Full quantitative claims for the validated material / regime / parameter range.

**Required label on figure:** "EXPERIMENTALLY VALIDATED — see validation/[ref]."

---

## Inheritance Rule

An output's model status is the *lowest* (weakest) status among all its inputs.

**Example:** If Engine 1 produces an `optical_prediction` and Engine 2 computes a
`fluence_threshold_proxy` using that field, the combined output is `fluence_threshold_proxy`.
It cannot be promoted by the optical field alone.

---

## Status in Metadata

Every CSV row and figure caption produced by the digital twin must include a `model_status` field
set to one of the keys above.  The registry CSV at
`outputs/csv/digital_twin/model_status_registry.csv` tracks all outputs and their statuses.

---

## Not a Status

The following are intentionally **not** model statuses in this system:

- "best estimate" — not specific enough; use the appropriate proxy level
- "conservative prediction" — implies quantitative calibration; use proxy
- "qualitative" — not a status; attach one of the 11 above
- "indicative" — same problem; attach a specific status
