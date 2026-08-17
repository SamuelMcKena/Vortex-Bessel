# Literature and Model Anchors for the Beam-to-Write Digital Twin

**Stage:** 8A.1  
**Last updated:** 2026-06-18  
**Status:** Structured placeholder — specific citations need verification from actual PDFs/Zotero.

Rows in the companion CSV (`outputs/csv/digital_twin/literature_model_anchors.csv`) marked
`source_status = "needs_literature_lookup"` must be verified against actual papers before
any parameter value is used in a simulation output.

---

## 1. Purpose

This document defines the **literature-anchor layer** for the beam-to-write digital twin.
It structures the evidence base that will underpin the simulator's physics proxy models,
identifies which parameters can be borrowed from literature with appropriate caveats, and
makes clear which parameters must be calibrated experimentally.

**This stage does not implement any simulator module.**  It creates the annotated evidence
scaffold so that later implementation stages (8B onward) build on documented physics rather
than bare assumptions.

---

## 2. Relationship to Stage 8A

Stage 8A defined the three-engine digital twin architecture and the model-status hierarchy
(11 levels from `optical_prediction` to `experimentally_validated_prediction`).

Stage 8A.1 populates the **prior knowledge** layer that can initialize Engine 2 (exposure/
writing/dose) and Engine 3 (material response) before any lab calibration is performed.

Without literature anchors, the default model-status ceiling is:
```
uncalibrated_material_response_proxy
```

With literature anchors, the starting model-status is still proxy — but the proxy parameters
are drawn from peer-reviewed sources with documented material/regime/wavelength specificity,
rather than being entirely ad hoc.

After lab calibration against this system's data, the status can be promoted to:
```
calibrated_material_prediction
```

and eventually (after direct measurement comparison) to:
```
experimentally_validated_prediction
```

---

## 3. Literature Anchor Categories

Eleven categories are required.  Each maps to one or more digital-twin engine layers.

| Category | Engine(s) | Primary use in digital twin |
|---|---|---|
| `bessel_beam_material_modification` | 1, 2, 3 | Beam-shape dependent threshold and feature geometry |
| `fused_silica_bessel_processing` | 1, 2, 3 | Reference system for Bessel modification; NOT transferable to ZnSe |
| `ultrafast_waveguide_inscription` | 2, 3 | Fluence regime, Δn achievable, track geometry by writing mode |
| `crznse_or_zns_family_waveguide_inscription` | 2, 3 | Closest-match material anchors for this study's target material |
| `heat_accumulation_scan_speed` | 2 | Thermal dose accumulation proxy; scan-speed / rep-rate context |
| `incubation_multi_pulse_threshold` | 2 | N-shot threshold scaling law (Jee–Becker) |
| `nonlinear_propagation_filamentation` | 1, 2 | Nonlinear propagation proxy hook; n2 and plasma parameters |
| `surface_ablation_modelling` | 2, 3 | Surface threshold and crater geometry; NOT same as internal modification |
| `refractive_index_reconstruction` | 3 | Waveguide Δn measurement methods and achievable magnitudes |
| `waveguide_mode_modelling` | 3 | Mode solver anchors for step-index and graded Δn profiles |
| `microscope_or_etch_observable` | 3 | What can be seen / measured at each processing regime |

---

## 4. How Anchors Map to the Three-Engine Digital Twin

### Engine 1 — Optical Train

Engine 1 is validated by the characterisation lock and requires no literature anchors for its
core physics (BL-ASM propagation is derived from first principles).

Literature anchors relevant to Engine 1:

- **Bessel beam formation** papers confirm that the cone angle → core diameter relationship
  and the Bessel zone length → propagation invariance hold in the regime used here.
- **Nonlinear propagation** papers provide n2 and critical power estimates to confirm whether
  the beam is in the linear propagation regime (required for Engine 1 validity).
- **Surface transmission** papers provide Fresnel / AR-coating transmission factors for
  energy accounting.

### Engine 2 — Exposure / Writing / Dose

Engine 2 uses literature to anchor:

- **Pulse energy → fluence**: analytical formula; anchored by definition.
- **Fluence threshold**: single-shot and N-shot incubation law anchored by Jee–Becker papers.
- **Writing trajectory**: pulse spacing and overlap geometry anchored by scan-speed / rep-rate
  studies in waveguide writing literature.
- **Heat accumulation proxy**: thermal diffusion length anchored by material thermal
  conductivity / diffusivity literature.

### Engine 3 — Material Response / Calibration

Engine 3 uses literature to anchor:

- **Modification feature geometry**: width, depth, and aspect ratio observed in Bessel and
  Gaussian focus experiments, for planning and sanity-checking.
- **Achievable Δn**: range of refractive-index changes observed in chalcogenide and ZnSe
  family materials.
- **Etch-rate enhancement**: HF / KOH etch selectivity observed in fused-silica Bessel
  processing (reference system; must not be transferred to ZnSe without verification).
- **Microscope observables**: DIC / phase contrast signatures of different modification regimes.

---

## 5. What Can Be Borrowed Directly

The following may be used directly as `direct_physical_constant` (no calibration required):

| Parameter | Material | Value | Source |
|---|---|---|---|
| Refractive index n (@ 1030 nm) | Cr:ZnSe | ~2.4 | Material data (e.g., FLIR/II-VI datasheets) |
| Density ρ | ZnSe | ~5260 kg/m³ | NIST / materials handbooks |
| Heat capacity Cp | ZnSe | ~339 J/kg/K | Materials handbooks |
| Thermal conductivity κ | ZnSe | ~18 W/m/K | Materials handbooks |
| Bandgap E_g | ZnSe | ~2.7 eV | Semiconductor databases |
| Bandgap E_g | Cr:ZnSe | ~2.0 eV (Cr d-level) | Literature |
| Speed of light c | — | 2.998×10⁸ m/s | SI definition |
| Planck constant ℏ | — | 1.055×10⁻³⁴ J·s | SI definition |

The following are known analytical results (no empirical calibration required):

- BL-ASM propagation transfer function (first principles)
- Gaussian beam Rayleigh range (z_R = πw₀²/λ)
- Pulse energy → fluence scaling (E_p / ∫|U|²dA)
- Thermal diffusion length (L_th = √(D_th / f_rep)) — formula is first-principles but requires D_th

---

## 6. What Can Only Be Used as Priors

The following parameters appear in peer-reviewed literature but cannot be directly transferred
to this system without lab verification.  Use as `literature_prior_only`:

| Parameter | Material system | Literature range | Transfer caveat |
|---|---|---|---|
| Nonlinear index n2 | ZnSe family | ~1–4 × 10⁻¹⁷ m²/W | Strong function of doping, wavelength, sample source |
| Single-shot ablation threshold F_th(1) | Cr:ZnSe | ~0.1–0.4 J/cm² | Varies with polish, coatings, doping level, pulse duration |
| Incubation exponent S | Dielectrics | ~0.8–0.95 | Measured in fused silica / sapphire; ZnSe may differ |
| Achievable Δn | ZnSe family waveguides | ~10⁻⁴ – 10⁻² | Depends strongly on writing mode and conditions |
| Etch-rate enhancement | Fused silica | 10–100× vs bulk | DO NOT transfer to ZnSe |
| Heat accumulation onset | Borosilicate / ZnSe | >100 kHz typical | Thermal diffusivity and beam size dependent |
| Damage threshold (surface) | ZnSe @ 1030 nm | ~0.05–0.2 J/cm² | Varies with surface quality, pulse duration |
| Plasma recombination time | Dielectrics | ~0.1–10 ps | Material-specific; use only as order-of-magnitude proxy |

---

## 7. What Must Be Calibrated

The following cannot be borrowed from literature at any meaningful precision.  The model must
not produce `calibrated_material_prediction` outputs until these are measured:

| Quantity | Why calibration is mandatory |
|---|---|
| Single-shot modification threshold (internal) | Depends on this laser + objective + sample combination |
| N-shot incubation law (S, F_th(1)) for Cr:ZnSe in this setup | Published S values are for different materials/configurations |
| Waveguide-writing fluence window | Width of the "modification not damage" window is system-specific |
| Track width and depth vs pulse energy | Geometry depends on beam shape, NA, aberrations |
| Achievable Δn and its sign | Cr:ZnSe modification type (type I/II/void) not yet confirmed |
| Side-lobe damage threshold | Vortex Bessel sidelobes create non-uniform exposure; unknown margin |
| Etch response (if used) | ZnSe etch behaviour under femtosecond modification is not well characterised |
| Waveguide mode size and loss | First measurement will establish the baseline |

---

## 8. Application-Specific Notes

### 8.1 Surface Ablation

Surface ablation is governed by the **surface fluence** at the first hit pulse.  Literature
provides ablation thresholds and Liu-method spot-size calibration.  Key distinction:

- Surface ablation threshold ≠ internal modification threshold (typically F_th_surface < F_th_internal)
- Surface craters are directly measurable; internal tracks are not
- Ablation studies in ZnSe exist for CO₂ and mid-IR lasers; 1030 nm NIR data are sparse

Use surface ablation anchors for:
- Confirming the damage-risk upper bound when writing at the surface
- Calibrating the Liu-method approach (log-diameter vs log-energy plots)

Do NOT use surface ablation thresholds to predict internal modification thresholds.

### 8.2 Internal Modification

Internal modification at 1030 nm in Cr:ZnSe requires the beam to deposit energy below the
surface without causing surface damage.  This requires:

- F_surface < F_th_surface (no surface ablation)
- F_focus ≥ F_th_internal (sufficient internal modification)

The ratio F_th_internal / F_th_surface is material and setup dependent.  In fused silica this
ratio is ~1.5–3×; for ZnSe it must be measured.

Bessel beams complicate this because the fluence distribution is extended in z.  The threshold
crossing geometry is different from a Gaussian focus.

### 8.3 Waveguide Inscription

Femtosecond laser waveguide inscription is well-established in oxide glasses (fused silica,
borosilicate).  Extension to mid-IR chalcogenide crystals (ZnSe, ZnS, Cr:ZnSe) is more recent
and less systematically characterised.

Key regime distinctions:
- **Type I**: smooth index increase, single scan, low repetition rate
- **Type II**: stress-induced waveguiding by adjacent damage tracks, depressed-cladding
- **Void / damage**: high-energy material removal — not a waveguide mechanism

Cr:ZnSe waveguide inscription literature exists but is sparse (as of 2025).  Parameters from
Zn-chalcogenide waveguide papers are the best priors but still require local calibration.

### 8.4 Bessel and Vortex-Bessel Writing

Bessel-beam writing differs from Gaussian-focus writing in:
- Extended Bessel zone: the whole interaction length is above threshold simultaneously
- Vortex annular intensity: for ℓ>0 the peak fluence forms a ring, not a central spot
- Sidelobe exposure: secondary rings can cause unwanted modification

Literature on Bessel-beam waveguide writing is limited to a few groups (mainly in fused silica
and borosilicate glass).  Vortex-Bessel writing of waveguides is essentially unexplored.

These represent genuine novelty in this study — no direct literature anchor can substitute for
original measurements.

### 8.5 Cr:ZnSe / ZnSe-Family Materials

Cr:ZnSe is a mid-infrared gain material with:
- Large nonlinear index n2 (factor ~5–10× fused silica)
- Low phonon energy (facilitates mid-IR laser operation)
- Relatively high damage susceptibility compared to oxides
- Anisotropic thermal expansion (birefringent stress effects post-modification)

ZnSe-family anchor parameters are better than fused-silica anchors for this study, but still
require local calibration because:
- Commercial Cr:ZnSe samples vary in Cr doping, crystal quality, and surface treatment
- The combination of 1030 nm + sub-100 fs + NA=0.45 + Bessel beam is not well-characterised
- Waveguide loss and mode confinement are unknown until first measurements

---

## 9. Recommended First Lab Calibration Sweeps

The following eight sweeps are recommended, in roughly increasing complexity.  Each sweep
is listed with what it calibrates, which digital-twin figure it validates, and which
material parameter it informs.

### Sweep 1 — Energy Threshold at Fixed Focus Depth and Writing Mode

**Description:** Static (stopped) or very slow scan at fixed depth.  Vary pulse energy from
below surface-damage threshold to visible internal modification.

**Calibrates:** Single-shot and few-shot modification threshold F_th(1, few).  
**Validates:** FIG-07 (threshold crossing map) — is the predicted crossing energy correct?  
**Informs:** `material.threshold_1shot_J_cm2`; absolute calibration of Engine 2.

### Sweep 2 — Scan Speed at Fixed Pulse Energy

**Description:** Fixed pulse energy just above threshold; vary scan speed from slow (high N) to
fast (low N, approaching single-shot).

**Calibrates:** N-shot incubation behaviour; track continuity threshold scan speed.  
**Validates:** FIG-08 (incubation curve) and FIG-09 (trajectory diagram).  
**Informs:** `material.incubation_exponent` S; minimum scan speed for continuous track.

### Sweep 3 — Repetition Rate / Burst Mode (if available)

**Description:** Vary rep rate from 1 kHz to 200 kHz at fixed scan speed; or use burst mode.

**Calibrates:** Heat accumulation onset; shot-to-shot thermal tail.  
**Validates:** FIG-10 (dose accumulation) and thermal proxy hook in Engine 2.  
**Informs:** Thermal diffusion length proxy parameter; heat accumulation model.

### Sweep 4 — Focus Depth

**Description:** Fixed energy and scan speed; vary focus depth from surface to 500 µm.

**Calibrates:** Aberration onset vs depth; surface damage vs internal modification boundary.  
**Validates:** FIG-06 (fluence x-z map) — does predicted Bessel zone hold with depth?  
**Informs:** Interface correction model; spherical aberration onset depth.

### Sweep 5 — Vortex Charge and Beam Shape

**Description:** Repeat threshold sweep for ℓ=0 (scalar), ℓ=1, ℓ=2, ℓ=3 at matched
peak fluence.

**Calibrates:** Charge-dependent modification geometry; sidelobe damage margin.  
**Validates:** FIG-04 (surface intensity) vs measured track cross-section.  
**Informs:** Whether vortex ring geometry produces annular modification track (novel).

### Sweep 6 — Static Multi-Pulse Exposure

**Description:** Stopped beam; fixed energy; vary N from 1 to 1000 shots.

**Calibrates:** Incubation law parameters (S, F_th(1)) with high N precision.  
**Validates:** FIG-08 (incubation curve).  
**Informs:** Jee–Becker model coefficients for this material.

### Sweep 7 — Surface vs Internal Comparison

**Description:** At the same pulse energy, compare surface modification crater (visible by
optical microscopy) with internal modification track (visible after polishing or etch).

**Calibrates:** F_th_surface vs F_th_internal ratio.  
**Validates:** Distinction between surface ablation and internal modification models.  
**Informs:** Whether surface ablation anchors are useful priors for internal threshold.

### Sweep 8 — Etched vs Unetched Microscope Comparison

**Description:** Etch one half of a written sample; compare optical microscope images before
and after etching.

**Calibrates:** Etch selectivity (etch-rate enhancement at modification sites).  
**Validates:** FIG-11 (modification proxy) vs observed etch depth / width.  
**Informs:** Whether HF or similar etch enhances ZnSe modification contrast; what etch
observable is available.

---

## 10. Recommended Next Stage

Once the literature anchor architecture is stable (Stage 8A.1 accepted), two paths diverge:

**Path A — Fill anchors from actual PDFs:**
> Stage 8A.2 — Fill literature anchors from actual PDFs / Zotero / manual paper extraction

Required before any literature parameter value is used in a simulation output.  The CSVs
currently contain `needs_literature_lookup` placeholders; these must be replaced with
verified citations and parameter values before the anchors are considered reliable.

**Path B — Start Engine 2 implementation:**
> Stage 8B — Optical cockpit + energy accounting

Can proceed in parallel with 8A.2 because Engine 2 uses only well-established analytical
formulas (fluence = E_p × |U|² / ∫|U|²dA) that do not depend on literature material parameters.

**Recommended:** Proceed to Stage 8B with the understanding that all material proxy parameters
in Engine 2 are `literature_prior_only` until Sweep 1 (energy threshold) is completed.
