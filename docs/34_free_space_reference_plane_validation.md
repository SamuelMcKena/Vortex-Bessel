# Stage 8C.3R.1 — Free-Space Reference-Plane Validation

Status: validation of the Stage 8C.3R component-plane upstream model at a
**free-space reference plane**, before any material/interface model exists.

This stage is a **free-space / reference-plane optical study only**. There is **no
sample material model**, no interface, no chosen material, thickness, surface
position, focus depth, or objective/interface geometry. The propagation medium is

```
n = 1.0  (free-space reference propagation)
```

and that is stated explicitly in the notebook, figures, metrics, and this document.

All outputs remain `optical_prediction` / `fluence_prediction` / `diagnostic_only`
with `final_export_allowed=False`. No material modification, absorbed/deposited
energy, plasma, ablation, voids, index change, nonlinear propagation, thermal
accumulation, or calibrated material prediction is computed or claimed.

---

## 1. Plane meanings (the only allowed labels)

The component-plane pipeline carries a complex field through these planes. Each is
labelled with exactly one of the allowed Stage 8C.3R.1 labels.

| pipeline state | label | physical meaning (n = 1.0) |
|---|---|---|
| `input_state` | `input_complex_field` | Gaussian entrance field; input decentre / ellipticity / rotation / aperture / tilt applied here to the complex field |
| `slm_state` (phase) | `SLM_phase_mask_generation` | vortex + axicon phase with **independent** centres; phase quantisation / phase noise |
| `slm_state` (amplitude) | `SLM_amplitude_or_active_area` | device amplitude: dead pixels, fill factor, finite active area; coherent zero-order carrier |
| `pupil_state` | `objective_pupil_plane` | circular pupil clip / decentre and low-order Zernike phase (waves over the pupil normalisation radius) |
| `sample_entrance_state` | `free_space_reference_plane` | **intended sample-entrance reference plane, n = 1.0** — the field handed to free-space propagation. NOT an in-material plane. |
| propagated stack | `free_space_reference_plane` (volume) | band-limited angular-spectrum propagation in free space to a z-stack of reference planes |
| legacy `apply_lab_perturbations_to_stack` | `post_propagation_diagnostic_only` | the old intensity-stack transform; `diagnostic_visual_transform`, never physics-active |
| camera crop / detector noise / autoscale | `display_only` | display operations; never physical metrics |
| 4F filter, relay imaging, mask rotation, jitter/drift, cone-angle, apex defect | `not_represented_by_current_engine` | honestly warning-only; no faked plane |

The endpoint of this stage is the **post-objective free-space reference plane**
(equivalently the *intended sample-entrance reference plane in air*). It is **not**
`sample entrance`, `sample plane`, `in sample`, or `in-material`, and is never
labelled as such in C3R.1 outputs. The future sample / interface / material
controls are retained but marked `future material/interface study`.

---

## 2. Validation questions and how each is answered

1. **Zero-control reproduces the clean free-space reference?** — `zero_control_equivalence`
   compares the no-perturbation pipeline against an independent canonical free-space
   construction + propagation (`canonical_free_space_reference`) and against the
   analytical Bessel-Gauss expectation (ring radius `j'_ell,1 / kr`, on-axis null for
   `ell>0`). Tolerances in §4.
2. **Passive losses preserved without hidden renormalisation?** — the energy audit
   (`compute_energy_audit`) reports raw field power before/after every component, the
   transmitted fraction, and `renormalisation_factor` (must be 1.0). `energy_accounting_valid`
   is a first-class boolean.
3. **Beam tilt produces the expected steering?** — `validate_beam_tilt` compares the
   measured centre-trajectory slope to the analytical free-space slope
   `dx/dz = kx/kz`, `kx = k0 sin(theta_x)`.
4. **Translation vs deformation vs clipping vs core contamination vs FOV failure?** —
   `classify_translation_vs_deformation` plus the crop/FOV reliability state
   (`numerically_reliable` / `caution_crop_limited` / `invalid_out_of_frame`).
5. **What does each individual upstream error do in isolation?** — the individual
   sensitivity atlas (one control family per sweep), never a combined stress test as
   primary evidence.
6. **Are the outputs numerically reliable?** — `fov_convergence_check` compares standard
   vs expanded grid/FOV and assigns a reliability label; metrics from an
   `invalid_out_of_frame` run are not presented as optical-tolerance predictions.

---

## 3. Energy / normalisation rules (enforced)

- Passive apertures, SLM active-area clipping, and pupil clipping cannot increase total
  transmitted power. `energy_accounting_valid=False` if any component shows gain.
- A local peak may rise after clipping **only** when the surviving field genuinely
  redistributes through propagation/interference. This is gated by
  `peak_rise_supported_by_energy_redistribution`, which is `True` only when: energy
  falls correctly, `renormalisation_factor == 1.0`, the encircled/core/annular/side-lobe
  energy redistribution is consistent with the peak change, and the FOV/crop check
  passes. Otherwise the rise is reported as `peak_rise_unvalidated`.
- No lost energy is restored by amplitude, intensity, fluence, or per-plane
  normalisation. The fluence scaling uses the genuinely transmitted reference-plane
  energy `reference_plane_pulse_energy_uJ = input * transmitted_fraction`.
- Display scaling is never used for physical metrics; any per-plane normalisation is
  reported separately from the raw power/energy audit.

---

## 4. Documented numerical tolerances

| quantity | tolerance | rationale |
|---|---|---|
| zero-control vs canonical complex-field cosine similarity | ≥ 1 − 1e-9 | same locked propagator; only orchestration differs |
| zero-control vs canonical intensity / fluence cosine similarity | ≥ 1 − 1e-9 | as above |
| zero-control vs canonical raw field-power relative difference | ≤ 1e-9 | no hidden renormalisation |
| zero-control peak-position difference | ≤ 1 pixel | identical fields |
| Bessel ring radius vs analytical `j'_ell,1 / kr` | ≤ 12 % | finite Gauss apodisation + discrete grid shift the ring slightly inward |
| on-axis core null (ell>0) core/peak ratio | ≤ 0.15 | hollow core for a charged vortex |
| measured vs analytical tilt steering slope | ≤ 15 % relative | centroid estimate on a structured beam over the valid z-fit range |
| standard vs expanded FOV peak-fluence convergence | ≤ 5 % | `numerically_reliable` threshold |

---

## 5. Controls still warning-only / future (not faked)

`not_represented_by_current_engine` (warning-only): first-order (4F) filter
radius/decentre/clipping, unwanted diffraction-order leakage, relay
magnification/decentre/tilt/aperture, SLM/mask rotation, physical-axicon cone-angle
error, axicon apex defect, pointing/stage/focus jitter.

`future material/interface study` (retained, disabled): sample tilt, surface offset,
focus-depth marker, refractive-index uncertainty, sample thickness, and every
material-response toggle. None of these alter the field in C3R.1.
