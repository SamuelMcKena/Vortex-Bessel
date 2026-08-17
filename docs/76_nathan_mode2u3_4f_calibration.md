# Nathan MODE 2U3 - Physical 4F Calibration Bridge (docs/76)

**Status:** lab procedure. The 4F focal length is nominal (300 mm from the bench description, not
bench calibrated; the 100 mm CSLM value is a removed placeholder), so the carrier-to-displacement
mapping is confirmed on the bench before the iris is fixed.

Nominal numbers at 1029 nm, f = 300 mm, carrier 6.25 lp/mm (20 px on the 8 um panel):
+1 order displacement `x = lambda * f * carrier` = 1.929 mm;
required iris radius 0.772 mm (diameter 1.543 mm);
simulated selected-order efficiency 0.9495; simulated
zero-order leakage 0.00e+00.

## Procedure

1. Display a simple blaze grating (20 px period, full panel) on SLM-H only.
2. Observe the Fourier plane on a card/camera at the nominal focal distance behind lens 1.
3. Locate the zero order (display a flat mask to identify it).
4. Locate the +1 order (record which physical side the carrier sends it to).
5. Measure the physical zero-to-first separation with the camera scale or a translation stage.
6. Infer the actual carrier-to-displacement mapping `x_measured / (lambda * carrier)` -> actual f.
7. Place the iris centred on the +1 order.
8. Sweep the iris radius from ~0.5 mm diameter upward.
9. Measure selected-order power versus radius (power meter after the 4F output).
10. Measure zero-order leakage (block the +1 order; residual power through the iris).
11. Compare with simulation: efficiency plateau ~0.95 with zero leakage at radius ~0.77 mm
    (0.40 x carrier separation); the M2S audit shows the whole 0.24-0.80 fraction range passes.
12. Update the hardware binding (actual f, iris centre, iris radius, measured efficiency) in the
    M2V build package.

The full per-focal-length geometry table is stored in `02_4f/physical_4f_closure.csv/json`.
