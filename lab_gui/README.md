# Dual-SLM and unified optical lab control

This is the September 14 v0.5 phase-safe GUI with an integrated manual-camera
correction workbench. Launch from the **repository root**:

```powershell
python -m pip install -r requirements-lab.txt
python run_lab_gui.py
```

See [the complete lab guide](../docs/lab/START_HERE.md) and
[source provenance](../docs/lab/SOURCE_AUDIT.md).
The existing HEDS vendor SDK must be installed on the lab PC. The SDK and raw
measurement archive are not included in this source directory.

The compact GUI preserves its prior carrier and panel configuration. Choose
**direct_phase_array**, establish the baseline, then open **Lab measurements**.
The session panel casts complete native phase arrays to SLM2 and does not add an
external LUT. BMG acquisition and stage positioning remain manual.

The **reference pupil diameter** in each SLM editor is not an aperture or output
mask. It only normalises pupil-local additive terms (Zernike, sample-interface
and N-fold terms). Changing it never blanks the carrier, wavefront, vortex,
axicon, focus or correction phase outside the reference circle. Legacy presets
which contain the removed `circular_pupil` gate flag are migrated safely with
that flag disabled.

Two state-sharing front ends are now available:

```powershell
# Familiar dual-SLM editor
python lab_gui\run_slm_gui.py

# Unified state, live quantitative camera, metrics, recovery and sessions
python lab_gui\run_lab_control.py
```

Headless status, formal capture, beam-walk analysis and q20 replay recovery use:

```powershell
python lab_gui\run_labcontrol_cli.py --help
```

The unified framework uses the same v0.6 phase composer and HEDS backend as the
compact GUI. See [lab-control quick start](../docs/lab_control/LAB_QUICK_START.md)
and [architecture](../docs/lab_control/ARCHITECTURE.md). Physical Beamage live
integration remains explicitly hardware-unverified until the official Gentec
named-pipe bridge is bound and validated on the Windows lab PC.
