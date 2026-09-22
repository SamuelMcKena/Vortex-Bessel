# SLM GUI v0.6: measurement sessions

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

The normal GUI preserves its prior carrier and panel configuration. Choose
**direct_phase_array**, establish the baseline, then open **Lab measurements**.
The session panel casts complete native phase arrays to SLM2 and does not add an
external LUT. BMG acquisition and stage positioning remain manual.
