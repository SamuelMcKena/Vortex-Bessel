# Literature Anchor Claim Boundaries

**Stage:** 8A.1  
**Last updated:** 2026-06-18  
**Status:** Blueprint (documentation only)

This document states the **explicit claim boundaries** for literature-anchored model parameters.
Every claim made using a literature anchor must be consistent with the rules stated here.
Violating these boundaries constitutes overclaiming.

---

## Core Principle

**Literature anchors initialise the model.  Lab calibration earns prediction status.**

A literature value that has not been verified against this system's data can only ever raise the
model status to `fluence_threshold_proxy` or `uncalibrated_material_response_proxy`.  It cannot
raise the status to `calibrated_material_prediction` or `experimentally_validated_prediction`.

No matter how many papers support a parameter, the model status for this specific material,
laser, and writing configuration remains a proxy until local calibration data are collected.

---

## Rule 1 — Fused Silica Does Not Calibrate Cr:ZnSe

A literature value measured in **fused silica, borosilicate glass, or any oxide glass** does
not calibrate any Cr:ZnSe or ZnSe-family material parameter.

This rule applies to:

- Single-shot ablation threshold F_th(1)
- Incubation exponent S and the Jee–Becker N-shot law
- Achievable refractive-index change Δn
- Etch selectivity and etch-rate enhancement
- Modification morphology (void vs smooth vs crack)
- Heat accumulation onset temperature and scan speed
- Nonlinear index n2 (fused silica n2 ≈ 2.4×10⁻²⁰ m²/W; ZnSe n2 ≈ 10⁻¹⁷ m²/W — over 2 orders of magnitude different)
- Plasma recombination times and collision cross-sections

**Allowed use of fused-silica anchors:**  
As a reference system for cross-validation of the optical and fluence engines.  Fused silica
data may confirm that the BL-ASM propagation and fluence scaling are computed correctly.
They do not constrain Cr:ZnSe material response at any level.

---

## Rule 2 — Gaussian Focus Papers Do Not Calibrate Bessel / Vortex-Bessel Writing

A paper that used a **conventional Gaussian focus** (single-scan waveguide writing) does not
calibrate a Bessel-beam or vortex-Bessel writing model, even if the material is the same.

Reasons:

- The spatial fluence distribution is fundamentally different: Bessel beam has a long Bessel
  zone with a narrow core, unlike the ellipsoidal Gaussian focus volume.
- For vortex Bessel (ℓ>0) the peak fluence is in an annular ring, not a central spot.
- The interaction between the Bessel zone and the scanning trajectory creates a different
  dose distribution than a Gaussian scan.
- Sidelobe exposure (secondary rings of the Bessel pattern) has no analogue in Gaussian focus.

**Allowed use of Gaussian-focus waveguide papers:**  
As existence proofs that ultrafast waveguide writing is achievable in the material class.
As order-of-magnitude priors for total pulse energy and scan speed ranges to explore.
As references for measurement methodology (loss measurement, RI reconstruction).

---

## Rule 3 — Surface Ablation Thresholds Do Not Apply to Internal Modification

A **surface ablation threshold** (measured by crater formation, transmission loss, or
surface damage) does not directly predict the internal bulk modification threshold.

Key differences:

- Surface ablation may be initiated by surface defects, contamination, or coatings that do not
  exist inside the bulk.
- Bulk modification requires the beam to propagate through overlying material, which introduces
  aberrations, nonlinear effects, and energy deposition over the full propagation path.
- For Bessel beams, the cone angle determines the threshold geometry: the surface intensity
  can be below the surface ablation threshold while the internal intensity is above the
  bulk modification threshold — or vice versa.
- Surface ablation threshold may be significantly lower than bulk modification threshold due
  to surface imperfections; or higher if the surface has a protective coating.

**Allowed use of surface ablation anchors:**  
To estimate the upper bound on surface fluence during internal writing (the surface must stay
below F_th_surface to avoid surface damage in transmission writing).  To check Liu-method
spot-size measurements against predicted beam area.

---

## Rule 4 — A Thresholded Fluence/Dose Model Is Not Damage Prediction

Comparing the simulated fluence map F(x,y,z) to a literature threshold F_th does not predict
what damage the material will exhibit.

The threshold comparison gives:

> "The fluence in region R exceeds the literature threshold value F_th."

It does not give:

> "The material in region R will be ablated / modified / cracked / have index change ΔΔn."

These are separate model layers.  Morphology (void, smooth track, birefringent zone, crack,
surface crater) depends on laser parameters, material properties, and pulse-to-pulse dynamics
in ways that a simple fluence threshold does not capture.

**Allowed claim:** "The fluence in region R exceeds the literature threshold of X J/cm² for
this material class."  
**Not allowed:** "The material in region R will be modified / damaged."

---

## Rule 5 — Heat Accumulation Models Require Material Thermal Constants and Context

A heat accumulation model (whether the heuristic proxy or a physical heat equation) can only
be meaningfully applied if:

- The thermal diffusivity D_th = κ/(ρ Cp) is known for the material.
- The repetition rate f_rep is specified.
- The scan speed v and beam size w₀ are specified (they determine dwell time).
- The sample geometry (bulk crystal vs thin film vs coated surface) is accounted for.

**Using a glass heat accumulation onset rep rate (typically >100 kHz) as a proxy for ZnSe**
is only valid as an order-of-magnitude estimate.  The thermal diffusivity of ZnSe
(D_th ≈ 10⁻⁵ m²/s) differs from borosilicate glass (D_th ≈ 5×10⁻⁷ m²/s) by over an order
of magnitude; the heat accumulation onset is correspondingly different.

**Required thermal constants for a meaningful model:**  
κ, ρ, Cp (for D_th); pulse duration τ (for initial temperature rise estimate); spot size at
focus w(z_focus) (for heat source volume).

---

## Rule 6 — Nonlinear Propagation Models Require Material-Specific Coefficients

A nonlinear propagation or filamentation model (Section 7 of `docs/23_...`) is anchored by:

- Nonlinear refractive index n2 [m²/W]
- Multiphoton ionisation cross-section σ_K (K-photon order)
- Plasma/free-carrier density coupling coefficient
- Plasma recombination time τ_rec

These are all material and wavelength dependent.  A filamentation model calibrated for fused
silica at 800 nm is not valid for Cr:ZnSe at 1030 nm.

**Allowed use of nonlinear literature:**  
As existence proofs that self-focusing and filamentation can occur in chalcogenide materials
at these pulse energies.  To estimate whether the peak power P is above or below the critical
power for self-focusing:

```
P_crit ≈ λ² / (2π n n2)
```

If P << P_crit, the linear propagation approximation (Engine 1) is valid.  This check can be
performed with literature-prior n2 values.

---

## Rule 7 — Cr:ZnSe / ZnSe-Family Anchors Are Priors, Not Calibration

Even within the ZnSe-family (ZnSe, ZnS, Cr:ZnSe, Cr:ZnS), parameters do not transfer
without verification:

- Crystal-grown vs CVD-deposited ZnSe have different impurity levels and defect densities.
- Cr-doping changes absorption, local heating, and potentially the modification threshold.
- Different sample batches from different suppliers will differ.
- Published ZnSe waveguide writing results are scattered across a wide parameter space.

**Allowed claim with ZnSe-family anchors:**  
"The literature suggests modification in ZnSe-family materials is achievable in the range
X–Y J/cm² at these conditions.  We use this as a prior to initialise our threshold proxy.
The prior has not been calibrated for this specific sample."

**Not allowed:**  
"Our threshold proxy uses the literature value of X J/cm², which is calibrated for our system."

---

## Rule 8 — Literature Anchors Initialise; Lab Calibration Earns Prediction Status

This rule summarises all previous rules.

The model status hierarchy in `docs/21_...` defines 11 levels.  Literature anchors can
initialise the proxy-level models (levels 4–8):

```
fluence_threshold_proxy (4)          ← literature F_th, S
dose_accumulation_proxy (5)          ← literature v, N, f_rep context
uncalibrated_material_response_proxy (8) ← literature morphology priors
```

They cannot grant prediction status at levels 9–11:

```
simulated_microscopy_proxy (9)       ← requires PSF calibration
calibrated_material_prediction (10)  ← requires local experiment calibration
experimentally_validated_prediction (11) ← requires direct comparison to measurement
```

To reach level 10 or 11, **local calibration data must be collected from this specific
system (laser + beam + material + sample)** and loaded into the model via the calibration
ingestion module (Stage 8G).

---

## Quick-Reference Table

| What the paper provides | Allowed use | NOT allowed |
|---|---|---|
| F_th in fused silica | Reference system validation | Cr:ZnSe threshold estimate |
| F_th in ZnSe-family material | Initialise fluence_threshold_proxy | Calibrated prediction |
| Incubation S in any dielectric | Prior for Jee–Becker proxy | Calibrated N-shot threshold |
| Δn in glass waveguides | Existence proof | ZnSe Δn estimate |
| Δn in ZnSe-family waveguides | Order-of-magnitude prior | Calibrated prediction |
| Etch selectivity in fused silica | Understanding etch concept | ZnSe etch response |
| n2 in ZnSe literature | Nonlinear proxy check (P vs P_crit) | Quantitative filamentation model |
| Thermal diffusivity of ZnSe | Heat accumulation proxy | Calibrated thermal model |
| Gaussian focus waveguide paper | Writing mode existence proof | Bessel threshold calibration |
| Bessel modification in fused silica | Feature geometry concept | ZnSe modification geometry |
| Surface ablation in ZnSe | Surface damage upper bound | Bulk modification threshold |
