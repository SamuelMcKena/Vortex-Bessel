# Digital Twin — Math and Physics Stack

**Stage:** 8A  
**Last updated:** 2026-06-18  
**Status:** Blueprint (documentation only)

This document defines the mathematical model for each layer of the digital twin.  All equations
are stated in SI units unless noted.  Notation follows `docs/00_theory.md` and
`docs/01_conventions.md`.

---

## Notation Summary

| Symbol | Meaning | Units |
|---|---|---|
| U(x, y, z) | Complex scalar field amplitude | — (normalised) |
| I(x, y, z) | Intensity = |U|² | W/m² (when normalised to pulse energy) |
| F(x, y, z) | Fluence | J/cm² |
| E_p | Pulse energy | J |
| τ | Pulse duration (FWHM) | s |
| f_rep | Laser repetition rate | Hz |
| w₀ | 1/e² beam radius at SLM | m |
| r₀ | Vortex ring / design ring radius | m |
| ℓ | Topological charge (integer) | — |
| k | Wavenumber = 2π/λ | m⁻¹ |
| k_r | Transverse (radial) wavenumber | m⁻¹ |
| k_z | Axial wavenumber = √(k²−k_r²) | m⁻¹ |
| γ | Axicon half-angle | rad |
| NA | Numerical aperture of objective | — |
| n | Sample refractive index | — |
| F_th | Single-shot ablation/modification threshold | J/cm² |
| S | Incubation exponent (Jee–Becker law) | — |
| N | Shot count (number of pulses per site) | — |
| N_p | Pixel count along one axis | — |
| Δx | Physical pixel pitch at SLM | m |

---

## Section 1 — Energy Accounting

### 1.1 Normalisation convention

The complex field U is normalised so that the total spatial integral gives unit energy:

```
∫∫ |U(x, y)|² dx dy = 1   [m⁻²]
```

The physical intensity at a given plane is then:

```
I(x, y) = E_p · |U(x, y)|²   [J/m²] = [J/cm² × 10⁴]
```

This convention means |U|² is the normalised intensity (probability density in field units).

### 1.2 Peak fluence

After scaling to physical units and converting to J/cm²:

```
F_peak = E_p · |U_peak|² / 10⁴   [J/cm²]
```

where |U_peak|² is the peak of the normalised intensity map [m⁻²].

For a Gaussian beam:

```
F_peak_Gauss = 2 E_p / (π w₀²) / 10⁴   [J/cm²]
```

For a Bessel-Gauss beam the peak is lower by the envelope factor (see `docs/00_theory.md`).

### 1.3 Throughput losses

The effective pulse energy reaching the sample incorporates:

```
E_eff = E_p · η_SLM · η_filter · η_pupil · η_transmission
```

where:
- η_SLM = SLM diffraction efficiency (default 0.70 for phase-only)
- η_filter = first-order selection fraction (computed from field, model status: `optical_prediction`)
- η_pupil = Gaussian pupil clipping efficiency (`vbb_study.equations.objective_pupil`)
- η_transmission = optics transmission (default 1.0 until measured)

The product η_SLM · η_filter · η_pupil is reported as `throughput_or_efficiency` in the study CSV.

---

## Section 2 — Optical Field (ASM / BL-ASM Propagation)

### 2.1 Angular spectrum propagation

The field at plane z is related to the field at z=0 by:

```
U(x, y, z) = IFFT2 { FFT2{U(x, y, 0)} · H(f_x, f_y, z) }
```

where the transfer function is:

```
H(f_x, f_y, z) = exp(i k_z z)
k_z = √(k² − 4π²(f_x² + f_y²))
```

and evanescent components (k_r > k) are set to zero.

### 2.2 BL-ASM bandlimit

The band-limited angular spectrum method (Matsushima 2009) applies a spatial-frequency bandlimit
to avoid aliasing at large propagation distances:

```
f_max = 1 / (2 Δx √(1 + (Δx/Δz_max)²))
```

This is enforced inside `bessel_twin_core._blasm_propagate`.

### 2.3 Bessel-Gauss target field

At the SLM plane the target field for a vortex Bessel-Gauss beam is:

```
U_SLM(r, φ) = exp(−r²/w₀²) · J_ℓ(k_r r) · exp(i ℓ φ)
```

The design k_r is derived from the target core diameter and objective NA.

### 2.4 Interface correction

At the air–sample interface the field is modified by a planar phase step:

```
U_sample(r) = U_air(r) · exp(i Δφ_interface)
Δφ_interface = (n − 1) k Δz_interface
```

where Δz_interface is the correction depth.  Refraction of the cone wavevector direction is
**not** modelled (see `docs/04_model_limitations.md`, Section 6).

---

## Section 3 — Fluence Scaling

### 3.1 Spatial fluence map

The spatial fluence map at a given z-plane is:

```
F(x, y, z) = E_eff · |U(x, y, z)|² / 10⁴   [J/cm²]
```

where E_eff includes all throughput losses (Section 1.3).

### 3.2 Normalisation check

The total energy in the fluence map should equal E_eff:

```
∫∫ F(x, y, z) dx dy ≈ E_eff   [J]
```

Deviation from this check is reported as `propagation_power_drift_fraction` in the study CSV.

---

## Section 4 — Threshold Proxy

### 4.1 Single-shot threshold

A fluence map is compared to the literature single-shot threshold F_th(1):

```
mask_threshold(x, y) = 1  if F(x, y, z) ≥ F_th(1)
                        0  otherwise
```

The area of the threshold-crossing region at the surface plane is:

```
A_crossing = ∫∫ mask_threshold(x, y) dx dy   [µm²]
```

### 4.2 Equivalent core diameter from threshold

The threshold-crossing diameter (equivalent J₀ diameter) is:

```
d_th = 2 √(A_crossing / π)   [µm]
```

This is a planning metric, not a modification diameter.

**Model status:** `fluence_threshold_proxy`

---

## Section 5 — Writing Trajectory

### 5.1 Shot overlap geometry

For a linear scan at speed v with repetition rate f_rep and beam radius r₀:

```
shot_separation = v / f_rep   [m]
overlap_fraction η = 1 − (v / f_rep) / (2 r₀)
```

For η < 0 (no overlap) shots are separated; for η > 0 shots overlap.

### 5.2 Multi-shot fluence accumulation

For N shots at positions (x_i, y_i) along the trajectory, the cumulative fluence is:

```
F_cumulative(x, y) = Σᵢ F(x − x_i, y − y_i)   [J/cm²]
```

This is a simple geometric sum — no heat accumulation, no incubation coupling.

**Model status:** `dose_accumulation_proxy`

---

## Section 6 — Incubation (Jee–Becker Law)

### 6.1 N-shot threshold

The incubation-modified threshold for N pulses follows the Jee–Becker accumulation law:

```
F_th(N) = F_th(1) · N^(S−1)
```

where:
- F_th(1) is the single-shot threshold [J/cm²]
- S is the incubation exponent (S < 1 means incubation lowers the threshold with shot count)
- N is the number of pulses

For Cr:ZnSe the literature value (Hashida et al.) is S ≈ 0.85, F_th(1) ≈ 0.15 J/cm².
These are literature proxies — not calibrated for this system.

### 6.2 Incubation-adjusted threshold crossing

With incubation, the threshold-crossing region for N shots is:

```
mask_incubation(x, y, N) = 1  if F(x, y) ≥ F_th(N)
                            0  otherwise
```

**Model status:** `fluence_threshold_proxy`

---

## Section 7 — Nonlinear Deposition Proxy Hook

This hook is a placeholder for future calibrated nonlinear models.  It scales the single-shot
fluence map by a heuristic intensity-dependent factor:

```
F_nl(x, y) = F(x, y) · [I(x, y) / I_peak]^(M−1)
```

where M is a proxy multiphoton order (default M = 3 for Cr:ZnSe at 1030 nm).  This is NOT a
physical model — it qualitatively concentrates the effective dose towards the intensity peak.

**Model status:** `nonlinear_deposition_proxy`

The nonlinear hook is disabled by default.  It must be explicitly enabled with a flag
`enable_nonlinear_proxy=True` and the figure must carry the proxy stamp.

---

## Section 8 — Thermal Accumulation Proxy Hook

This hook is a placeholder for future heat-diffusion models.  It models pulse-to-pulse thermal
buildup as an exponential decay:

```
T_accumulated(x, y, t_n) = Σₖ F(x − x_k, y − y_k) · exp(−(t_n − t_k) / τ_thermal)
```

where τ_thermal is a heuristic thermal time constant (default: τ_thermal = 1/f_rep for
continuous scan, i.e., one-shot memory).  This is NOT a physical heat-diffusion model.

**Model status:** `thermal_accumulation_proxy`

The thermal hook is disabled by default.  Must be enabled explicitly.

---

## Section 9 — Microscope Proxy (Simulated Image)

The modification proxy volume V(x, y, z) is convolved with a Gaussian PSF to simulate an
optical microscopy top-view image:

```
I_sim(x, y) = V(x, y, z_focus) ⊛ PSF(x, y)
PSF(x, y) = exp(−(x² + y²) / (2 σ_PSF²))
σ_PSF = λ_mic / (2 NA_mic)
```

where λ_mic and NA_mic are the microscope wavelength and NA (defaults: 532 nm, 0.7 NA).

**Model status:** `simulated_microscopy_proxy`

Neither the PSF shape nor the image contrast has been calibrated.  The simulated image shows
only the spatial extent of the modification proxy at diffraction-limited resolution.

---

## Implementation Notes

- All Engine 1 calculations are implemented in `bessel_twin_core` and `vbb_study.equations`.
  Do not re-implement them in the digital twin module.
- Engine 2 calculations (Sections 3–6) belong in `vbb_study/digital_twin/exposure.py` (Stage 8C).
- Engine 3 calculations (Sections 7–9) belong in `vbb_study/digital_twin/material.py` (Stage 8D).
- Nonlinear and thermal hooks (Sections 7–8) are stubs in Stage 8D; they are not numerically
  implemented until Stage 8F+.
- All physical constants use SI throughout.  Conversion to J/cm² happens at the output stage only.
