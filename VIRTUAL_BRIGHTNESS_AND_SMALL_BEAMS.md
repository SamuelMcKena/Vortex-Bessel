# Virtual brightness and small-beam design

Launch from the repository (or extracted FULL_SYSTEM) root with `python lab_gui/run_lab_control.py`.

## Compare propagation fairly

1. Select Virtual Lab and cast your masks as usual.
2. Leave exposure and gain unchanged while moving camera z or changing the beam.
3. View → **sensor log** is the default fixed logarithmic display; **sensor range**
   uses a fixed linear scale. Fading is real numerical fading, not corrected away.
4. The older **log**, **percentile** and **full range** views still auto-adjust
   contrast. Use those to inspect shape, not compare brightness.
5. If the image saturates, reduce exposure; if it is faint, increase exposure,
   but record that change when comparing captures. Normal noise/fluctuations remain.
6. **Auto-expose** (beside Snapshot) sets the exposure so the peak at the current
   plane sits near 75 % of full scale, then takes a frame. At fixed sensitivity the
   beams differ enormously: at 1 ms a q=0 focus clips (peak ~9400 counts at
   z=15 mm) while q=20 at the same plane reads 2840 and z=45 mm reads 47. The frame
   chip says CLIPPED when saturated, and saturation also costs score on the
   correction tab, so expose before correcting.

Recorded virtual counts use a fixed relative sensitivity, independent of frame
peak, camera z, correction and resolution. This is not calibrated radiometry.
Changing Gaussian radius or ellipticity conserves incident power before finite
window/aperture losses. It does not magically restore power lost by clipping.
Existing captures from the older per-frame-normalised engine are not numerically
comparable to new captures. New metadata says `fixed_relative_sensitivity_v1` and
`per_frame_peak_normalisation: false`. Reacquire a baseline and rerun correction.

The native view is noise-free; in sensor range/log it uses the captured frame's
sensitivity, not a separately peak-normalised display. It remains a model view,
not extra detector resolution or recorded camera data.

## Explore the earlier tiny-beam design

Open **1 Beam → Small-beam designer…**. This reuses `vbb_study.design` rather than
introducing a different correction model. Enter target diameter, reference Bessel
length, charge, wavelength, SLM input radius and sample refractive index.

**Load old repository target** restores the repository default: 3 µm equivalent
q=0 first-null diameter, 150 µm reference length, q=3, 1029 nm and n=2.44.
This is a recovered default, not proof it was your final optimised setting.

The tool reports required transverse sample/SLM magnification, sample input
radius, cone strength and NA. For q≠0 it reports the actual bright-ring diameter
separately. A first-null diameter is not FWHM, and a reference length is not a
measured axial FWHM. Non-propagating targets are refused; high-angle designs are
flagged as requiring vector/nonparaxial validation.

Calculate shows an ideal Bessel XY/radial reference at its own resolved sample
scale, not a propagated physical bench image. Export saves a 400-dpi figure and
JSON design. Edited inputs invalidate previous results until recalculated.

This tool does **not** cast masks or feed its ideal image to the virtual
correction algorithm.

## The bench route to those targets (added 2026-09-17)

The designer now also reports what *this* bench needs, and **Set this objective
on the virtual bench** fills in Settings → Objective (you still press Apply
geometry). The route is exact rather than a new propagation: the operator-reported
relay is a symmetric 4F (SLM–300–L1–300–aperture–300–L2–300–axicon, f=300 both),
so it is 1:1 and the axicon's k⊥ fixes the ring size. An ideal demagnifying
objective after the axicon scales the whole post-axicon field exactly — rings
×M, z ×M², cone angle ÷M for M = 1/N — because (k⊥·w) and (λz/w²) are both
invariant. The simulated frame therefore *is* the sample-plane pattern in new
units; `labcontrol/sample_plane.py` converts, and the camera metrics gain a
SAMPLE PLANE line.

Consequences worth remembering:

* **Ring size follows the objective; Bessel length follows the input beam
  radius.** A wider input beam lengthens the region and lowers the peak at the
  same ring size and the same total power.
* NA = N × 0.079 for the bench axicon, so past about 1:12 in air there is no
  propagating cone; such a request is refused, not drawn.
* The old 3 µm / 150 µm target maps to a **1:3.3 objective with a 0.13 mm input
  beam radius**.
* Not modelled: the objective's pupil truncation and aberrations, vector/high-NA
  effects, and any imaging path used to view the sample plane. A high-NA answer
  is a design target, not a prediction of that bench.

The 4F aperture is modelled too (**4F stop Ø**, Settings): a stop of diameter D
passes SLM detail down to D/(2λf). Default is ideal. For q=20 on a 2 mm beam,
10 mm passes 98 % of the light, 3 mm 81 % and 1 mm only 14 %, while a plain
Gaussian passes all of them untouched — a pure-phase vortex winds q times around
its core, so it carries structure finer than a tight iris transmits. Verified
against grids 256–4096 and reproduced by the engine's own frame metadata.
