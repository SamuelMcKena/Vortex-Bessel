# 15 September 2026 — q=20 system commissioning and rapid-recovery plan

This session has two purposes:

1. **Commission the current optical system properly** so we know which measured features come from beam pointing, axicon placement, low-order aberration, detector sampling and the recovered higher-order phase.
2. Turn that knowledge into a **fast recovery workflow** so that a routine disturbance such as removing/replacing the physical axicon does not require repeating the full commissioning procedure.

The intended end state is not “measure everything every time”. The intended end state is:

> coarse mechanical recovery → a few camera captures → automatic/short sensorless correction → verification → process.

The dense 15-plane retrieval is the escalation route when the quick recovery fails, not the everyday starting point.

---

## A. What is already reusable and should NOT be re-calibrated every session

These are commissioning/static properties unless the corresponding hardware or software path changes:

- HEDS direct phase path at 1030 nm (`direct_phase_array` / `showPhaseData`)
- SLM panel identity, orientation and native pixel coordinates
- SLM pixel pitch and panel dimensions
- locked 20-pixel carrier convention
- camera pixel pitch / BMG decoding
- relay geometry and SLM-to-system coordinate mapping, provided the relay optics are not moved
- phase-LUT/interferometric calibration once it has been measured properly

A moved/reseated axicon may change the beam axis, centring, symmetry and residual phase. It does **not** automatically invalidate all of the above.

---

## B. Today’s commissioning sequence

### B1 — Freeze the starting optical state

Before changing anything:

- use the current normal q=20 arrangement
- connect HEDS in `direct_phase_array` mode at 1030 nm
- cast both SLM baselines
- keep camera exposure, gain, attenuation and BeamGage export settings fixed
- record the physical axicon position/orientation and take a phone photo of the mount/translation readout if useful
- record the current 4F pinhole state

Do not begin by applying the old retrieved correction. Establish the present system first.

### B2 — Separate propagation-axis walk from q=20 shape distortion

Use three widely separated camera positions for the axis check:

- 31 mm
- 38 mm
- 45 mm

Capture at least 3 repeats at each position for:

1. **q=0 with the physical axicon present** — bright central Bessel spot makes centre tracking easy.
2. **q=20 with the same alignment** — gives the vortex-Bessel ring-centre walk.

If it is easy and repeatable to remove/reinsert the axicon without disturbing other optics, also capture a simple Gaussian/reference beam at the same three camera-stage positions. This is the best way to estimate camera-stage runout independently of the axicon.

Interpretation:

- q=0 and q=20 show the same walk → primarily common optical-axis / camera-stage / axicon pointing geometry, not a vortex-specific aberration.
- Gaussian reference shows the same walk → camera translation axis/runout contributes strongly; record a per-plane camera reference axis and do **not** steer the laser merely to follow the camera.
- Gaussian stays fixed but q=0/q=20 walk → real post-axicon beam-axis tilt; correct mechanical axicon alignment first, then consider a small SLM tip/tilt residual.
- q=20 walks differently from q=0 → investigate vortex/SLM coordinate or asymmetric phase effects.

The 14 September stack gave an observed relative walk of about 8.85 µm/mm in camera x and 4.31 µm/mm in camera y (about 9.84 mrad total relative to the camera travel direction). Treat this as a measured relative slope, **not yet as a proven beam angle**.

### B3 — Mechanically minimise gross axicon pointing before software correction

If B2 shows that the post-axicon beam is genuinely tilted relative to the reference beam:

- use the q=0 Bessel beam for fast alignment
- use near/far camera positions rather than a single image
- adjust axicon lateral position/tilt so the centre changes as little as practical with z
- use back-reflection / existing alignment references where available
- re-capture the three axis-check planes

Do not use coma to correct a beam that is simply pointing in the wrong direction.

A future dedicated SLM tip/tilt layer can remove the small residual once the mechanical alignment is close. Keep that separate from the 20-pixel order-selection carrier.

### B4 — Start the measured sensorless correction session

Launch:

```powershell
cd C:\PhD\Code\Vortex-Bessel
git fetch origin
git switch lab/measurement-correction-workflow
python -m pip install -r requirements-lab.txt
python run_lab_gui.py
```

Create a new lab session. For low-order correction use three strong planes spanning the useful region, initially:

- 34 mm
- 38 mm
- 42 mm

Use 3 repeats per plane for screening/optimisation.

Capture at least 3 fresh dark BMGs with the same camera settings and start the session only after the GUI baseline is confirmed as the phase actually displayed through HEDS.

### B5 — Low-order correction order

Run the signed sensorless sweeps in this order:

1. `astig_x`
2. `coma_y`
3. `coma_x`
4. `astig_xy` only if there is still a clear residual

Do not start with spherical/defocus. The 14 September coarse perturbations showed that coma directions were much more informative than the large spherical probe.

For every mode:

- capture the prescribed start control
- capture all shuffled signed candidates
- capture the end control
- evaluate
- if a candidate is recommended, capture the fresh control → candidate → control verification groups
- accept only if the verification passes
- otherwise restore the accepted control and move on

The score uses the rotational average of the measured baseline as a symmetry reference, not an ideal simulated Bessel beam. This is deliberate because the fine q=20 outer rings are currently close to/under the camera sampling limit.

### B6 — Freeze the accepted low-order phase

At the end of the sensorless rounds, the accepted SLM2 correction is the experimentally demonstrated low-order correction for this hardware state.

Record/export:

- complete accepted `total_phase_rad.npy`
- additive `correction_phase_rad.npy`
- session report
- exact HEDS/camera settings
- the measured q=0/q=20 axis slopes

Do not add `total_phase_rad.npy` on top of the ordinary GUI phase a second time; it is already the complete command for that session state.

### B7 — Dense corrected validation stack

Only after the low-order correction is frozen, capture the full corrected q=20 stack:

- z = 31, 32, ..., 45 mm
- 4 independent BMG repeats per z
- unchanged exposure/gain/attenuation/export settings

This is the dataset to compare directly against the 14 September 15×4 baseline.

Use it to answer separately:

- Did the principal ring become more azimuthally uniform?
- Did the dark core stay dark?
- Did the propagation-axis walk change?
- Did the radial profile / k_perp remain physically sensible?
- Which apparent outer-ring structure remains after genuine optical correction?
- What residual phase does the multi-plane Miao retrieval recover now?

The new residual retrieval is the next higher-order correction iteration. Do not simply keep increasing the gain of the original residual map.

---

## C. Camera-aliasing diagnostic — later, not required to commission today

The direct Beamage data remain useful for relative correction even while some fine outer rings are undersampled. Do not spend today rebuilding the camera path solely for this.

Later, add a 2×–4× imaging relay between a chosen Bessel plane and the Beamage. The optical beam is unchanged; only its image on the sensor is enlarged. If the fourfold/checkerboard outer structure collapses into concentric rings with adequate detector sampling, classify that component as a detector artifact and do not try to correct it with the SLM.

Anything that persists in laboratory coordinates after adequate sampling remains a candidate real optical aberration.

---

## D. Intended rapid-recovery workflow after commissioning

Once the system response is commissioned, a routine event such as someone removing and replacing the axicon should use the following short path.

### Level 1 — quick recovery (normal case)

1. Load the golden SLM/HEDS preset.
2. Coarsely recenter/reorient the axicon using the established physical marks/back reflection.
3. Capture q=0 at three z positions and compare axis slope against the commissioned reference.
4. Capture q=20 at the same three positions.
5. Run only the low-order modes flagged by the measured error signature (usually tip/tilt if implemented, then coma/astig).
6. Verify the winning correction with fresh captures.
7. If all quick metrics are inside the commissioned tolerances, process immediately.

This should be a few captures plus a short correction sweep, not a new PhD project every morning.

### Level 2 — short recovery (quick recovery fails)

Run a denser 3–5-plane capture and a residual phase retrieval / calibrated correction update. Verify it experimentally and then process.

### Level 3 — full re-commissioning (rare)

Use the full 15-plane or denser procedure only if:

- relay optics/SLMs/camera geometry changed
- the axicon replacement causes behaviour outside the existing correction basis
- q=0/q=20 propagation no longer matches the commissioned system model
- the quick and short recovery paths fail verification
- the hardware phase path or phase response changed

---

## E. What success looks like today

A successful commissioning session does **not** require a visually perfect BeamGage screenshot.

Success means we leave with:

- a measured reference for q=0 and q=20 propagation-axis walk
- confidence about whether the dominant walk belongs to camera geometry, axicon pointing or vortex-specific behaviour
- at least one low-order phase term accepted (or experimentally demonstrated unnecessary)
- a reproducible accepted phase state and report
- ideally a fresh corrected 31–45 mm ×4 q=20 stack
- a clear residual-error list split into **optical correction**, **mechanical alignment**, and **detector-sampling** components

That dataset is the foundation for the later one-button / few-capture recovery workflow.
