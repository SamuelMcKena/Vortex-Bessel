# Fu et al. oscillating-polarization Bessel route: lab adaptation

Reference: Fu, Zhang and Gao, *Bessel beams with spatial oscillating
polarization*, Scientific Reports 6, 30765 (2016), DOI 10.1038/srep30765.
The paper's Fig. 4 has two **sequential** reflective SLMs. SLM1 displays a
holographic axicon HA1 plus an order-l spiral phase; a 22.5 degree HWP gives
equal H/V amplitudes; SLM2 applies HA2 plus order-2l spiral phase to H only;
a 45 degree QWP precedes the analyzing polarizer and camera. The second
holographic axicon causes the analyzer lobes to rotate along z. This route
does not use the Nathan hexagonal sector mask, split arms, or physical axicon.

The maintained model is `vbb_study/digital_twin/fu_oscillating_vector_bessel.py`.
Its specialist runner is `tools/render_fu_oscillating_vector_bessel.py`.
Run from `Vortex-Bessel/`:

```powershell
..\.venv\Scripts\python.exe tools\render_fu_oscillating_vector_bessel.py --grid-n 1024 --ell 3
```

Outputs are under `outputs/figures/digital_twin/fu_oscillating_vector_bessel/`:
the z montage, fixed-plane HA2-off/on comparison, rotation curve, per-plane CSV,
manifest, and native 1920 x
1080 phase-design arrays. The default uses the recorded 1029 nm source scope,
2 mm Gaussian radius, 8 um SLM pitch, HA1 radial period 160 um (20 pixels),
and HA2 radial period 1200 um (150 pixels). These are **lab-scaled design
choices**, not the paper's 1550 nm, 880 um and 2184 um settings. The selected
periods place a nominal ~175 mm polarization cycle inside the simulated
40-220 mm camera scan. The order-3 six-lobe case extends the paper's measured
orders 1 and 2.

An HA2-off control keeps the same H/V winding and propagation while removing
HA2's extra radial phase. Its analyzer orientation is stationary. The HA2-on
prediction rotates by about 60 degrees over the 180 mm scan, while the
total-intensity annulus remains broadly round. The model propagates complex
transverse Jones components with angular spectrum; analyzer images come from
linear projections. The output CSV records the 2l harmonic orientation and
contrast at each plane. The manifest records numerical power drift.

This calculation uses the current `Vortex-Bessel` hardware resolver for the
nominal HOLOEYE panel identity, geometry and pixel pitch, and the current
source-scale wavelength and native panel grid. The separate CSLM inventory is
a diagnostic demo profile with unmeasured placeholders; its 1030 nm value is
not promoted to a measured wavelength here. The blank measured-bench template
does not yet support a calibrated prediction. No SLM command is sent by this
runner, and the native arrays remain phase designs pending the measured LUT.

The two SLMs are treated as conjugate planes, with nominal reflection parity
written in one receiver frame. This has not been matched to a measured
SLM1-to-SLM2 relay, fold-mirror count, or panel orientation. The HWP/QWP and
analyzer are ideal, and neither per-panel phase LUT nor camera response is
calibrated. Native arrays are phase designs in radians and are **not
drive-ready**. A real bench trial needs the measured relay/spacing, panel
phase stroke/LUT and parity, waveplate orientation and retardance, and camera
scale/z origin before uploading masks or claiming the predicted rotation.
