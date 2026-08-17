# Model Limitations

This document records the known limits of the simulation models used in this
study.  Read this before interpreting any quantitative result.

---

## 1. Scalar Paraxial Approximation

All propagation in the main study uses the **scalar paraxial approximation**:

- Polarisation state is not tracked (single scalar complex amplitude).
- The paraxial approximation requires the cone half-angle `gamma` to be
  small (< ~15°).  Results near this limit should be treated with caution.
- No vectorial focusing, no tight-focus corrections, no back-aperture
  polarisation mixing.

**When this matters:** For high-NA objectives (NA > 0.5) or very large axicon
angles the scalar results will underestimate the focal-volume distortion and
overestimate z-invariance.

---

## 2. Angular Spectrum Method (ASM / BL-ASM)

Propagation uses the band-limited angular spectrum method (BL-ASM,
Matsushima 2009).  Key limitations:

- The field is sampled on a finite numerical grid.  Sampling must satisfy
  the BL-ASM bandlimit for the propagation distance and pixel pitch.
- Grid truncation introduces edge artefacts; check that the beam never
  approaches the grid boundary.
- Evanescent components are excluded by design.

Check validity flags reported in `vbb_study.vbb_validation`.

---

## 3. SLM Holographic Axicon Route

The holographic axicon is modelled as:

- Phase-only SLM with 8-bit greyscale quantisation.
- First-order filtering in a 4f relay: the filter passes the desired
  diffraction order and rejects the DC and higher orders.
- Blaze carrier grating is included for order separation.

**Not modelled:**
- SLM pixel fill-factor diffraction loss.
- SLM temporal phase flicker.
- Alignment errors in the 4f relay.
- Non-flat SLM surface distortion.
- Temporal pulse distortion through the SLM.

---

## 4. Physical Axicon Route

The physical axicon is modelled as:

- Ideal conical phase with no aberrations.
- Mask separation to separate the conical field from the Gaussian envelope.
- Surface roughness, tip rounding, and material dispersion are not modelled.

---

## 5. Objective and Pupil

- The objective is modelled as an ideal lens with a hard pupil aperture.
- Pupil radius is derived from the NA and tube-lens focal length.
- Aberrations (spherical, coma, field curvature) are not modelled unless
  Zernike coefficients are explicitly supplied.

---

## 6. Interface / Sample Correction

- The air–sample interface applies a planar phase step for the refractive
  index change.
- **No refraction of the cone wavevector direction** is implemented (first-
  order plane-wave approximation).
- Interface aberration correction is labelled **ideal** unless experimentally
  measured Zernike coefficients are supplied.
- For focusing deep in sample the scalar model neglects spherical aberration
  from the refractive-index mismatch.

---

## 7. Vector Jones Model

The vector Jones model (`vbb_study.vbb_vector`) builds polarised fields by
combining scalar-propagated components with Jones vectors.  Limitations:

- The scalar propagation does not account for polarisation-dependent
  reflection at the SLM or at interfaces.
- True radial/azimuthal vector beams require a spatially varying polarisation
  element (vortex retarder, segmented waveplate, or q-plate).
  **The current Case-1 lab setup does not produce true radial/azimuthal
  beams.** See `docs/04_actual_lab_vector_case1.md`.

---

## 8. Material Response Proxy

All material outputs are **planning proxies**:

- Fluence and incubation calculations use literature threshold values for
  Cr:ZnSe unless experiment-calibrated values are supplied.
- Threshold-crossing maps show *where the fluence exceeds the threshold*,
  not *where the material is actually modified*.
- The incubation law `F_th(N) = F_th(1) * N^(S-1)` is applied with
  literature coefficients; calibration has not been performed for this
  system/sample combination.
- XZ line-fluence maps are non-energy-conserving visualisation proxies.
  They are useful for spatial planning but do not represent actual
  energy deposition.

See `docs/03_materials_application.md` for the full proxy warning.

---

## 9. Hexagonal and Polygonal Beams

- Hexagonal patterns that look good in the focal plane are **not
  automatically z-stable** propagating channels.
- Do not claim a hexagonal beam is a propagation-invariant Bessel-like
  structure unless the strict Bessel region and accepted propagation depth
  metrics demonstrate it.
- Phase-only approximations (using only the phase of the target field) are
  clearly less accurate than complex-amplitude targets; both are implemented
  but must not be reported interchangeably.

---

## 10. Discrete N-fold Beams

- A discrete N-fold superposition of plane waves has a finite interference
  field that is only approximately periodic in z.
- The accepted depth depends on the number of plane waves and their angular
  separation.  The wider the angular cone the shorter the interference zone.
- Side-lobe level is a function of N; low-N beams have significant side lobes.

---

## 11. What Is Not a Limitation of This Study

The following are **intentional scope choices**, not model deficiencies:

- No temporal pulse dynamics: this is a CW-equivalent beam-shaping study.
- No nonlinear optics: the study is linear optics only.
- No thermal or mechanical response: material response is threshold-only proxy.
- No detector/camera model: outputs assume ideal field measurement.
