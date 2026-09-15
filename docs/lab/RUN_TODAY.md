# Run today — 15 Sep 2026

Use the commissioning branch for today's bench session:

```powershell
cd C:\PhD\Code\Vortex-Bessel
git fetch origin
git switch lab/commissioning-and-rapid-recovery
git pull
python -m pip install -r requirements-lab.txt
python run_lab_gui.py
```

Full plan: `docs/lab/2026-09-15_COMMISSIONING_AND_RAPID_RECOVERY.md`.

## First captures: axis diagnostic

Before Zernike optimisation, capture q=0 and q=20 at 31, 38 and 45 mm, at least 3 repeats per plane. Put each beam state in its own folder and use names such as:

```text
axis_check\q0\z031_r01.bmg
axis_check\q0\z031_r02.bmg
axis_check\q0\z031_r03.bmg
axis_check\q0\z038_r01.bmg
...
axis_check\q20\z045_r03.bmg
```

Analyse each state directly from the checkout:

```powershell
python tools\analyze_beam_walk.py axis_check\q0 --pixel-um 5.5
python tools\analyze_beam_walk.py axis_check\q20 --pixel-um 5.5
```

The output reports x/y walk in µm/mm (numerically mrad for the small relative angle) and R². Treat it as beam direction **relative to camera travel**, not absolute pointing, until camera-stage runout is independently checked.

If practical, repeat the same three-plane measurement with the physical axicon removed and a simple Gaussian/reference beam. If that Gaussian shows the same walk, store the resulting per-plane camera reference axis and do not correct the laser merely to follow the camera rail.

## Then run the correction workbench

Once gross axicon pointing is mechanically satisfactory, create a fresh Lab measurements session in the GUI. Use 34, 38 and 42 mm with 3 repeats for the sensorless low-order sweeps. Start with `astig_x`, then `coma_y`, then `coma_x`; use `astig_xy` only if still useful. Verify every winning candidate with fresh control → candidate → control captures before acceptance.

If time remains, freeze the accepted phase and capture a full corrected q=20 stack from 31–45 mm in 1 mm steps with 4 repeats per plane. That becomes the new input for residual Miao retrieval and direct comparison with the 14 Sep baseline.
