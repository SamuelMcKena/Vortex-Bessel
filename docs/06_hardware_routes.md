# Hardware Routes

This document describes each hardware route modelled in the study, which
elements are required, which routes are currently implemented in the
modelled lab stack, and what future hardware would be needed for the
routes not yet available.

For the formal distinction between `ideal_math`, `lab_proxy`,
`case1_existing_hardware`, `proposed_hardware`, and `unsupported` see
`docs/05_study_taxonomy.md`.

---

## Optical Planes Convention

Every route uses the same plane labelling:

| Plane label | Physical location |
|---|---|
| SLM plane | Spatial light modulator surface |
| Fourier / filter plane | Back focal plane of the first 4f lens (first-order filter sits here) |
| Objective back-pupil plane | Entrance pupil of the objective lens |
| Focal / surface plane | Front focal plane of the objective (sample surface) |
| In-sample plane | Depth z_s below the surface inside the medium |

Propagation is tracked sequentially through these planes.  Any model that
skips a plane (e.g., no 4f relay) must note this explicitly.

---

## Route 1 — Holographic SLM Axicon (current modelled lab route)

**Taxonomy:** `generation_method = holographic_slm`,
`hardware_status = case1_existing_hardware`

### Elements required
1. Spatial light modulator (SLM) — phase-only, ≥ 1024×768, 8-bit.
2. 4f relay with spatial filter at the Fourier plane.
3. Objective lens.
4. Optional: vortex phase mask or spiral phase written onto the SLM.

### What is modelled
- Gaussian beam incident on the SLM.
- SLM encodes: conical axicon phase + spiral phase (for ell > 0) + blaze
  carrier grating for first-order separation.
- Phase is 8-bit quantised, optionally wrapped.
- 4f relay propagates the encoded field to the Fourier plane where a hard
  circular aperture retains only the first diffraction order.
- The filtered field is propagated to the objective back pupil and then to
  the focal plane via the objective model.

### Current limits
- Phase-only encoding: amplitude information is discarded.
- Blaze efficiency loss (typically 80–90 % for an optimised grating).
- First-order filter radius must be chosen to exclude DC and +2 orders;
  undersized filter clips the Bessel ring in k-space; oversized filter
  admits DC contamination.

---

## Route 2 — Physical Refractive Axicon

**Taxonomy:** `generation_method = physical_axicon`,
`hardware_status = lab_proxy`

### Elements required
1. Physical refractive axicon (glass cone, base angle γ).
2. Imaging relay or direct beam propagation.
3. Objective lens.

### What is modelled
- Gaussian beam passes through the axicon conical phase.
- Optional mask separation step: the axicon output is propagated to a
  plane where the Bessel ring and Gaussian envelope separate, then a
  hard mask selects the Bessel ring only.
- Field is propagated to the focal plane.

### Current limits
- Tip rounding and surface roughness of a real axicon are not modelled.
- Dispersion of the axicon glass is not modelled.
- No alignment tolerance model.

---

## Route 3 — Ideal Target (no hardware encoding)

**Taxonomy:** `generation_method = ideal_target`,
`hardware_status = ideal_math`

The true Bessel-Gauss field `J_ell(k_r r) exp(i ell phi) exp(-r^2/w0^2)`
is placed directly on the grid.  This is a mathematical reference only;
no physical element produces this exact field.

Use this for:
- Metric validation (a perfect Bessel-Gauss should report exactly its
  design parameters).
- Comparison baseline for lab-realistic routes.

---

## Route 4 — Objective Pupil and First-Order Filtering (shared sub-route)

The objective back-pupil model is used by both Routes 1 and 2:

- Pupil radius `R_pupil = f_obj * NA / n_medium`.
- A hard circular aperture at the pupil clips the field.
- Clipping is a significant source of sidelobe energy; the fraction of
  power clipped is reported in the output metadata.

First-order filtering is used in Route 1:
- Filter transmits a ring-shaped region in k-space centred on the Bessel
  ring radius `k_r`.
- Filter inner and outer radii should be set to pass the main Bessel ring
  but reject DC and higher-order rings.

---

## Route 5 — Through-Sample Propagation and Interface Correction

**Taxonomy:** `hardware_status = lab_proxy` (interface correction is ideal)

### Planes involved
1. Focal / surface plane: field at z = 0 (sample surface).
2. In-sample propagation: BL-ASM propagation using `k_medium = n * k_0`.
3. Interface correction: phase ramp for planar air–sample interface.

### What is modelled
- Refractive index step at the flat air–sample interface is applied as a
  planar phase.
- In-sample propagation uses the medium wavenumber `k = n * 2pi/lambda0`.
- The Bessel cone angle in the sample differs from the air angle by
  Snell's law: `sin(theta_s) = sin(theta_air) / n`.

### What is not modelled
- Spherical aberration from a curved interface.
- Scattering in a turbid or inhomogeneous sample.
- Depth-dependent aberration growth.

---

## Route 6 — Vector Beam Routes

### Current modelled route (Case 1)

**Taxonomy:** `generation_method = ideal_target` or `holographic_slm`,
`hardware_status = case1_existing_hardware` (scalar encoding only),
`vector_lab_realizable = False` for true radial/azimuthal.

The current modelled lab setup uses:
- A standard linear-polarisation SLM.
- A single linear polariser after the SLM.
- No spatially varying polarisation element.

This setup **does not produce true radial or azimuthal polarisation**.
It produces a scalar (single-polarisation) vortex beam.  The vector field
must be synthesised from two co-propagating scalar modes with orthogonal
polarisations, which requires either:

1. A vortex retarder (q-plate) in the beam path.
2. A segmented waveplate.
3. A second SLM path with orthogonal polarisation.
4. A Sagnac or Mach–Zehnder interferometric route.

None of these are currently in the modelled lab stack.

### Proposed routes (future hardware)

**Taxonomy:** `hardware_status = proposed_hardware`

| Route | Required element | Notes |
|---|---|---|
| q-plate (half-integer charge) | q-plate in beam path | Converts circular to radial/azimuthal |
| Segmented waveplate | Segmented λ/2 element at pupil | Converts linearly polarised ring to vector |
| Dual-SLM Mach–Zehnder | Two SLMs + PBS | Full control of both polarisation components |

Any output claiming a true vector beam must use `hardware_status = proposed_hardware`
or `ideal_math`, not `case1_existing_hardware`.

---

## Route 7 - Hexagonal, Polygonal, And Discrete N-Fold Beams

**Taxonomy:** Stage 8 uses `generation_method = phase_only_slm`,
`complex_amplitude_proxy`, `amplitude_phase_target`,
`discrete_superposition`, `holographic_phase_mask`,
`future_hardware_required`, or `simulation_only`; and
`hardware_status = current_lab_realizable`, `future_hardware_required`,
`simulation_only`, or `diagnostic_only`.

Hexagonal and polygonal outputs are separated into distinct optical cases:

1. **Focal-plane polygon targets**: filled or hollow polygonal target masks.
   These are target definitions only and are labelled `focal_plane_only`.
2. **Hollow polygonal outlines**: geometry-proxy outline fields with explicit
   outline fidelity, edge uniformity, core suppression, and side-lobe metrics.
3. **Discrete N-fold superpositions**: finite plane-wave constellations on one
   transverse-k ring. These are lattice/kaleidoscope fields, not localized
   hollow polygonal rings.
4. **Phase-only approximations**: current-lab-realizable only when the row is
   generated by a tested phase-only SLM or holographic phase-mask route.
5. **Complex-amplitude targets**: labelled `future_hardware_required` or
   `simulation_only` unless a tested phase-only encoding route is provided.

A hexagonal or polygonal focal-plane pattern is not automatically a
propagation-stable Bessel-like beam. Propagation stability must be measured
with z-dependent metrics such as accepted depth, symmetry retention, outline
fidelity, core suppression, and side-lobe contamination.

Phase-only SLM compatibility is not assumed for complex-amplitude polygonal
targets. If complex amplitude is required, the case is labelled
future_hardware_required or simulation_only unless a tested encoding route is
provided.

---

## Summary Table

| Route | Hardware status | Currently modelled | Future required |
|---|---|---|---|
| Ideal Bessel-Gauss target | `ideal_math` | Yes | — |
| Holographic SLM axicon | `case1_existing_hardware` | Yes | — |
| Physical axicon | `lab_proxy` | Yes | — |
| Objective pupil clipping | shared sub-route | Yes | — |
| First-order filtering | part of holographic route | Yes | — |
| Air beam propagation | `lab_proxy` | Yes | — |
| Surface/interface correction | `lab_proxy` (ideal) | Yes | Measured Zernike |
| In-sample propagation | `lab_proxy` | Yes | — |
| True vector (radial/azimuthal) | `proposed_hardware` | Model only | q-plate / dual-SLM |
| Hexagonal/polygonal focal-plane target | `future_hardware_required` or `simulation_only` | Yes | Complex-amplitude or tested encoding |
| Phase-only polygonal approximation | `current_lab_realizable` | Yes, focal-plane only | Propagation testing for stable use |
| Discrete N-fold phase-only proxy | `current_lab_realizable` | Yes | Material calibration for writing claims |
