# Theory — Vortex Bessel-Gauss Beam Simulation

This document derives the key physics implemented in `bessel_twin_core.py` and links each formula to the implementing function. Equation labels match the inline comments in the code.

---

## 1. True Bessel-Gauss target field

A Bessel-Gauss (BG) beam of order ℓ in the focal plane is:

```
U_BG(r, φ) = J_ℓ(k_r r) exp(iℓφ) exp(−r²/w₀²)
```

where `J_ℓ` is the ℓ-th order Bessel function of the first kind, `k_r` is the transverse wavevector (rad/m), and `w₀` is the Gaussian waist.

**Code**: `bessel_twin_core.build_bessel_gauss_field_ideal` evaluates this field on the focal-plane crop grid using `scipy.special.jv`. For â„“ > 0 this true target has near-zero on-axis amplitude.

---

## 1a. Conical axicon-phase input field

The scalar engine also uses a Gaussian-apodized conical axicon input/source field:

```
U_cone(r, Ï†) = exp(âˆ’rÂ²/wâ‚€Â²) exp(i[-k_r r + â„“Ï†])
```

This is not the same object as the true Bessel-Gauss target because it does not contain `J_â„“(k_r r)`. It is an idealized axicon-phase field that can produce a finite-aperture Bessel-like field after propagation.

**Code**: `bessel_twin_core.build_conical_axicon_field_ideal` evaluates this conical input field. `bessel_twin_core.build_sample_field_ideal` is kept only as a backwards-compatible wrapper for the conical field and should not be described as the Bessel-function target builder.

---

## 1b. Propagated finite-aperture Bessel-like field

Realistic and physical-axicon routes apply finite apertures, SLM sampling/quantisation, filtering, alignment assumptions, and propagation. Their output is a propagated finite-aperture Bessel-like field that targets the ideal Bessel-Gauss family but is not identical to an infinite analytic target.

---

## 2. Axicon transmission function

A physical axicon of half-angle γ transmits a plane wave with the phase:

```
T_axicon(r) = exp(−i k_r r)    where k_r = k₀ (n_axicon − n_medium) tan(γ)
```

For the SLM-encoded holographic axicon in air (`n_medium = 1`), this becomes the axicon phase component:

```
φ_axicon(r) = −k_r_SLM × r
```

**Code**: `bessel_twin_core._continuous_phase`, line ~770:
```python
phi_axicon = -design.kr_slm_m_inv * R
```

The axicon cone angle is recovered from `compute_design_from_targets` (line ~595):
```python
gamma = math.atan(kr_slm / (k0 * (n_axicon − n_medium)))
```

---

## 3. Combined axicon + spiral phase plate (SPP)

A vortex Bessel beam adds an azimuthal phase ramp (topological charge ℓ):

```
φ_total(r, φ) = φ_axicon(r) + ℓ φ + φ_blaze(x) + φ_corr(x, y)
```

where `φ_blaze = 2π × carrier_cpm × x` shifts the beam to the first diffraction order, and `φ_corr` is an optional interface correction.

**Code**: `bessel_twin_core._continuous_phase` assembles all four components (line ~780):
```python
phi_vortex = design.ell * PHI
phi_blaze  = TWOPI * config.slm.carrier_cpm * X
phase = phi_axicon + phi_vortex + phi_signum + phi_blaze + phi_corr
```

---

## 4. Geometric Bessel zone z_max

For a Gaussian input of waist `w_SLM` and axicon cone angle giving radial wavevector `k_r` in the sample medium, the geometric Bessel zone extends to (Baliyan 2023):

```
z_max = k_medium × w₀_sample / k_r_sample
```

where `w₀_sample = L × k_r_sample / k_medium` is the design Gaussian waist and `L` is the target zone length.

The FWHM zone (canonical metric) is shorter: empirically `≈ 0.55–0.65 × z_max` for the finite Gaussian aperture (check 5 in the validation suite, ratio 0.61).

**Code**: `bessel_twin_core.analytic_references` (line ~658):
```python
zmax = design.w0_sample_m * k_medium / design.kr_sample_m_inv
```

**Canonical zone**: `bessel_twin_core.bessel_zone_metrics` (line ~1702), FWHM of axial peak trace.

---

## 5. Ring radius = first J'_ℓ zero / k_r

For a vortex Bessel beam of order ℓ, the ring radius (location of peak transverse intensity) equals the first zero of the derivative J'_ℓ:

```
r_ring = jnp_zeros(ℓ, 1)[0] / k_r
```

For ℓ = 0 (Gaussian-Bessel), the on-axis peak replaces the ring, and the effective "core radius" is:

```
r_core = j_{0,1} / k_r    (j_{0,1} = 2.404825557... is the first zero of J_0)
```

**Code**: `bessel_twin_core.compute_design_from_targets` (line ~597):
```python
ring_r = sp.jnp_zeros(ell_abs, 1)[0] / kr_sample
```

`vbb_study.vbb_metrics.peak_plane_radial_metrics` measures this from a simulated intensity plane.

**Validation**: check 6 and 19 in the suite confirm relative error < 2%.

---

## 6. Angular spectrum propagation

Free-space propagation of a paraxial field `U(x, y; z=0)` to `z > 0` uses the angular spectrum transfer function:

```
H(fx, fy; z) = exp(i 2π z sqrt(n²/λ² − fx² − fy²))
```

BL-ASM (band-limited angular spectrum method) applies a hard spectral cutoff at the Nyquist frequency to suppress aliasing wrap-around. The propagator is obtained via:

**Code**: `bessel_twin_core.make_bl_asm_propagator` (line ~1159), returns a callable `prop(z_m)`.

SAS (scalable angular spectrum): zooms the output grid by a magnification factor, trading output pixel size for propagation range. Falls back to BL-ASM when within the BL-ASM validity zone.

**Code**: `bessel_twin_core.scalable_angular_spectrum_propagate` (line ~1259).

**Validation**: checks 1–2 (energy conservation, Gaussian waist law), checks 13–14 (SAS vs BL-ASM agreement).

---

## 7. SLM1 → SLM2-conjugate → physical axicon route

The physical axicon path avoids holographic first-order isolation entirely:

```
SLM1: applies vortex phase  exp(iℓφ)
  ↓  free-space propagation Δz to SLM2
SLM2: applies conjugate of accumulated phase  −φ_accumulated(r)
  ↓  physical refractive axicon  exp(−ik_r r)
  ↓  objective focuses into sample
```

SLM2 "flattens" the wavefront distortion accumulated between SLM1 and the physical axicon. This allows the physical axicon to receive a clean vortex-only wavefront.

**Code**: `vbb_study.vbb_axicon.PhysicalAxicon.generate` orchestrates this pipeline.

---

## 8. Holographic vs physical routes (swappable)

Both routes target the same ideal in-sample Bessel-Gauss vortex field, but they are not physically identical implementations. They differ in aperture limits, SLM quantisation, first-order filtering, efficiency, alignment sensitivity, and aberrations. They are selected via `TwinConfig.generation_method`:

| method | `generation_method` | `path` | first-order isolation needed |
|--------|--------------------|---------|-----------------------------|
| holographic | `"holographic"` | `"realistic"` | yes — carrier + blaze + filter |
| physical | `"physical"` | `"ideal"` | no — physical axicon does it |

The holographic path is limited by the SLM carrier frequency: the axicon cone ring frequency must be less than the carrier minus the isolation filter radius. For a 3 µm core Bessel beam this is satisfied; for sub-2 µm cores at long zones it fails.

**Code**: `vbb_study.vbb_studies.build_beam_to_surface_result` dispatches on method and path (line ~355); `vbb_study.vbb_materials_study.holographic_max_zone_um` computes the analytical carrier constraint.

---

## 9. Air-beam / in-sample separation

The simulation separates two stages:

1. **Air beam (beam-to-surface)**: holographic SLM → first-order isolation → pupil → focal-plane propagation → air propagation to glass surface. Medium is air (n = 1).

2. **In-sample (through-sample)**: refraction at the air/Cr:ZnSe interface → ASM propagation in n = 2.44 medium to the target write depth.

The surface is placed at z = 0 in the stitched coordinate system. Air is z < 0; sample is z > 0.

**Code**:
- Air stage: `vbb_study.vbb_studies._result_from_realistic_holographic`
- Interface + in-sample: `vbb_study.vbb_sample_study.run_through_sample`
- Stitching: `vbb_study.vbb_studies._stitched_volume`
- Full pipeline: `vbb_study.vbb_studies.run_full_source_to_sample`

Power continuity at the interface is reported/checked by the validation pipeline when the relevant through-sample stage is run: `continuity_relative_power_error < 1 × 10⁻⁹`.
