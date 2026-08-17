# SurfaceField Energy-Scaled Optical Cockpit

**Stage:** 8C
**Last updated:** 2026-06-18
**Model status:** `fluence_prediction` (optical fluence only)
**Final export allowed:** False

---

## 1. Purpose

Stage 8C connects the Stage 8B energy-accounting cockpit to the **real simulated
optical field** produced by the existing repository engine
(`bessel_twin_core`). It converts an optical-field intensity array — a single
transverse plane (`SurfaceField`) or a propagation volume — into an
**energy-conserving transverse fluence map** in J/cm², using the pulse energy at
the sample computed by the Stage 8B energy ledger.

The data flow is:

```
existing repo optical field (SurfaceField / propagate_volume)
  → canonical OpticalFieldPlane / OpticalFieldStack       (field_coupling)
  → energy-conserving transverse fluence scaling          (field_fluence)
  → peak fluence / peak intensity estimates
  → field-derived cockpit metrics + diagnostic figure     (field_figures)
```

Stage 8C is still **optical/fluence only**. It does not implement material
response, thresholds, dose accumulation, nonlinear deposition, or thermal
accumulation.

---

## 2. Relationship to Stage 8B

Stage 8B (`01_optical_cockpit_energy_accounting.ipynb`) computes the pulse energy
at the sample from the optical chain, plus a quick "Mode A" fluence estimate that
divides that energy by an **assumed effective beam area**.

Stage 8C keeps the Stage 8B energy ledger unchanged and *reuses* it:
`ledger.energy_at_sample_uJ` is the input energy for the Mode B scaling.
The Stage 8B conversions are reused directly:

- `scale_intensity_to_fluence_j_cm2` — the energy-conserving array scaler.
- `peak_intensity_w_cm2` — the approximate `I ≈ F / τ` conversion.

Stage 8C replaces the *assumed area* with the *actual field shape*.

---

## 3. Why Mode B is better than effective-area fluence

The Mode A estimate (Stage 8B) assumes the energy is spread uniformly over an
assumed effective area. For a Bessel / vortex-Bessel beam this is wrong in two
ways: the energy is concentrated in a narrow core (or annulus) far smaller than
any "spot", and a large fraction of the energy sits in the side-lobe rings.

Mode B scales the **measured intensity distribution** of the simulated field, so:

- the peak fluence reflects the real core/ring concentration, not a flat disk;
- the spatial fluence map is available for downstream diagnostics;
- the total integrated energy is conserved exactly (to floating point).

Mode B is therefore a faithful **optical** fluence prediction, while Mode A is
only an order-of-magnitude planning estimate.

---

## 4. Field conventions

The canonical containers fix the axis order explicitly so downstream code never
guesses:

| Container | Array | Convention |
|---|---|---|
| `OpticalFieldPlane` | `intensity` | `intensity[y, x]` |
| `OpticalFieldStack` | `intensity_zyx` | `intensity[z, y, x]` |

Coordinates:

- `OpticalFieldPlane`: scalar `dx_um`, `dy_um`, optional `z_um`.
- `OpticalFieldStack`: 1D `x_um`, `y_um`, `z_um` arrays, each strictly monotonic;
  `intensity_zyx.shape == (len(z_um), len(y_um), len(x_um))`.

The repository `SurfaceField` stores complex components `Ex/Ey/Ez` and a `grid`
dict whose `x`/`dx` are in **metres**. The adapter forms intensity as
`|Ex|² + |Ey|² + |Ez|²` over the available components and converts the sampling
to microns (`× 1e6`). The `propagate_volume` result stores `intensity_stack`
already in `[z, y, x]` order with `z` in metres and a square `crop_grid`.

---

## 5. Energy-conserving scaling equation

For a transverse plane with intensity `I(x, y)` and pixel area `dA`:

```
F(x, y) = E_sample · I(x, y) / ( Σ I · dA )        [J/cm²]
```

with `dA = dx_um · dy_um` converted to cm² (`× 1e-8`) and `E_sample` in J. This
guarantees energy conservation over the plane:

```
∫ F(x, y) dA  =  E_sample
```

The integrated-energy check `integrated_energy_uJ_from_fluence` recovers
`E_sample` to floating-point precision, and the notebook reports the residual.

Peak intensity reuses the Stage 8B approximation:

```
I_peak ≈ F_peak / τ        [W/cm²]
```

labelled approximate (no nonlinear reshaping, no plasma, no thermal feedback).

---

## 6. Why the 3D volume integral is NOT the total pulse energy

For a stack, each transverse plane is the same single pulse observed at a
different propagation distance `z`. Summing the intensity over the **whole 3D
volume** and treating that volume integral as the pulse energy would count the
same pulse once per plane — it is physically meaningless for a single pulse and
would scale with the (arbitrary) number of sampled z-planes.

Stage 8C therefore uses **per-plane transverse-energy normalisation**: every
z-plane is independently scaled so that its transverse integral equals
`E_sample`. The volume is a stack of transverse fluence maps, **not** a
deposited-energy volume. This is recorded in the stack caveat.

---

## 7. What propagation power drift means

The propagator conserves total power only over the full (uncropped) field. Within
the **cropped diagnostic window**, the transverse intensity integral varies with
`z`: near the Bessel zone the captured power is high, while far from focus the
field spreads beyond the crop and the captured power drops.

`propagation_energy_drift_fraction = (max − min) / max` is computed from the
**raw** transverse intensity integrals (before per-plane renormalisation). A
large drift does not mean energy is lost physically; it flags that the crop
window captures a shrinking fraction of the field away from the Bessel zone, so
per-plane fluence comparisons across very different `z` should be read with that
in mind.

---

## 8. What the output can claim

- "The pulse energy at the sample is `E_sample` µJ" (from the Stage 8B ledger).
- "The energy-scaled **optical** peak fluence of this field is `F_peak` J/cm²."
- "The approximate peak intensity is `I_peak` W/cm² (flat-top, no nonlinearity)."
- "The transverse fluence distribution has this shape" (the map itself).
- "The captured transverse power drifts by X% across the scanned z-range."

These are `fluence_prediction` / `optical_prediction` claims.

---

## 9. What the output cannot claim

**Claim boundary:** Energy-scaled fluence maps are *optical* fluence
predictions. They are not absorbed-energy maps, not a dose map, not dose,
not material-modification maps, and not a damage prediction.

Specifically, Stage 8C must not be used to claim:

- that any fluence value crosses a material threshold (no threshold model here);
- material modification, **void prediction**, crack formation, refractive-index
  change, **ablation prediction**, **waveguide prediction**, or welding;
- absorbed or deposited energy (the field is incident optical intensity);
- any **calibrated material prediction** or experimentally validated result.

The 3D fluence stack is explicitly **not a deposited-energy volume**.

---

## 10. How Stage 8D will use this

Stage 8D (3D beam-to-sample visualiser) will consume the canonical
`OpticalFieldStack` and the `FluenceStackResult` to render the beam-to-sample
journey in 3D (isosurfaces, XZ/YZ slices, peak-fluence-vs-z traces). Because the
containers and conventions are fixed here, Stage 8D needs no new physics — it is
a visualisation layer over the Stage 8C outputs and keeps the same
`fluence_prediction` status and claim boundary.

---

## 11. Failure modes

The Stage 8C layer fails loudly rather than fabricating data:

| Condition | Error |
|---|---|
| No field passed to an extractor | `MissingOpticalFieldError` |
| Field object cannot be adapted | `UnsupportedSurfaceFieldError` (lists what was missing) |
| Intensity has NaN/inf | `InvalidOpticalFieldError` |
| Intensity has negative values | `InvalidOpticalFieldError` |
| Wrong array dimensionality | `InvalidOpticalFieldError` |
| Stack shape ≠ coordinate lengths | `InvalidOpticalFieldError` |
| Non-monotonic coordinates | `InvalidOpticalFieldError` |
| Zero transverse integral on a plane | `InvalidOpticalFieldError` / `ValueError` |
| Invalid pulse energy (negative / NaN) | `ValueError` |
| Unsupported stack normalisation | `ValueError` |
| Saving a figure with caveats disabled | `CaveatsRequiredError` |

In the notebook, if no real field is available and the synthetic demo is disabled
(`require_real_field=True`, `allow_synthetic_demo_field=False`), the notebook
raises `MissingOpticalFieldError` instead of drawing a placeholder beam.
