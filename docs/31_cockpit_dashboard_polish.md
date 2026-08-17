# Integrated Cockpit Dashboard — Stage 8C.2 Polish

**Stage:** 8C.2
**Last updated:** 2026-06-18
**Model status:** `diagnostic_preview` (optical / fluence only)
**Final export allowed:** False

---

## 1. Purpose

Stage 8C.2 is a visual and usability rescue of the integrated beam-to-write
cockpit produced in Stage 8C.1. The backend (energy accounting, field coupling,
energy-conserving fluence, lab-realism report, peak diagnostics) is unchanged;
this stage rebuilds `plot_integrated_cockpit_dashboard` so the output reads like
a scientific cockpit instead of a backend debug grid.

It changes **presentation only**. No optical physics, propagation, axicon,
scalar/vector equations, or lock baselines are touched, and no material response
is introduced.

---

## 2. Dashboard layout

The dashboard uses a `GridSpec` hierarchy (8 rows × 12 columns) with these
regions, top to bottom:

| Region | What it shows |
|---|---|
| Header band | Title, "optical / fluence diagnostic only" subtitle, route summary (method, ell, wavelength, pulse duration, rep rate, energy in / at sample, selected vs target depth), and a large **DISPLAY TRUST** badge |
| Beam-path strip | 13 chips (laser → … → material-response-disabled), each colour-coded by stage status with an arrow flow |
| Warning cards | High-visibility cards for captured-power drift, crop-edge peak, first-order geometry, pupil clipping, power limit, pulse overlap |
| Experiment / Energy / Exposure | Readable cards plus an energy-ledger bar chart with the energy-at-sample line annotated |
| XY intensity / XY fluence / Central ROI | Annotated field panels at the selected plane with clean colourbars |
| XZ propagation (dominant) | The largest panel: XZ fluence with surface / target / selected / global-peak markers and a crop-boundary caution banner when relevant |
| Peak-vs-z and drift | Peak fluence vs z (global + central ROI, selected/target markers) and raw captured-power fraction vs z |
| Interpretation / Future-disabled | Plain-language claim boundary, and the intentionally disabled future-physics modules |

The XZ propagation panel spans two grid rows and eight columns so the
beam-to-sample journey is the visual centre of gravity.

---

## 3. Annotations

Key quantities are annotated directly on or beside the panels: selected plane
depth, target depth, global-peak depth, central-ROI peak depth and value,
selected peak fluence, energy at sample, route type, ell, scan mode and speed,
and the captured-power drift fraction. The header always shows wavelength, pulse
duration, repetition rate, and energy before optics / at sample.

---

## 4. PASS / CAUTION / FAIL

The **DISPLAY TRUST** badge summarises whether the current optical/fluence
display can be trusted, via `compute_overall_status`:

- **PASS** — no caution/fail signals from warning flags, peak diagnostics, or
  the lab-realism stages. Unavailable hardware metrics ("missing") are surfaced
  in the cards but do not by themselves downgrade the badge.
- **CAUTION** — at least one caution signal, e.g. large raw captured-power drift
  (> 20%) or a global peak sitting near the crop boundary.
- **FAIL** — at least one fail signal, e.g. average power above the configured
  limit or a target depth outside the sample.

A large drift or a crop-edge global peak is the honest, expected state for a
tightly cropped real Bessel volume, so a default real-field run typically shows
**CAUTION** — the figure makes that obvious rather than hiding it.

---

## 5. What is still NOT modelled

This dashboard shows optical fluence only. It does **not** show absorbed energy,
a deposited-energy volume, material modification, a written feature, plasma,
nonlinear propagation, or thermal accumulation. The 3D fluence volume is a stack
of per-plane transverse fluence maps, not deposited energy.

These remain disabled future stages (shown in the "Future physics - disabled"
card): dose accumulation, threshold maps, calibrated material response, surface
ablation, nonlinear propagation, thermal accumulation, and a microscope proxy.

---

## 6. Honesty rules baked into the figure

- The selected display plane never silently promotes a crop-edge global peak; a
  red banner appears on the XZ panel when the global peak is near the boundary.
- Raw captured-power drift is plotted explicitly and reported as a percentage;
  it is not disguised by the flat per-plane conserved energy.
- The claim-boundary card restates that the output is optical fluence only.
- Saving is refused when caveats are hidden (`CaveatsRequiredError`), and demo
  fields are never saved as governed outputs.

---

## 7. What remains for later

Once the cockpit is accepted as usable, the recommended next stage is the 3D
beam-to-sample visualiser (a visualisation layer over the existing
`OpticalFieldStack` / `FluenceStackResult`, still optical/fluence only). Material
response, dose, thresholds, and calibration remain later stages that require
material constants and measured calibration data.
