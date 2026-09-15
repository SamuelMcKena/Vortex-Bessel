# Formal measurement provenance

Live preview is observational. `FormalCaptureService` creates scientific records
only after the required complete-phase hashes match the last successful casts.
It can cast a stale configured state when the caller explicitly permits
`auto_cast`; otherwise it refuses capture.

## Formal capture order

1. Generate complete dual-SLM phases through the shared composer.
2. Compare configured hashes with successful cast hashes.
3. Cast required stale panels or fail closed.
4. Snapshot `ExperimentState`.
5. Wait the declared settling interval.
6. Request fresh quantitative frames.
7. Save untouched numeric arrays and file/pixel hashes.
8. Calculate state-aware metrics from those arrays.
9. Seal the measurement manifest and register it in the session.

## Layout

```text
session/
  session.json
  objects/phase/<complete-phase-sha256>.npy
  trials/trial_0001/
    state.json
    phase_hashes.json
    camera/frame_001.npy
    camera/frame_002.npy
    metrics.json
    measurement.json
  reports/session_report.md
```

Complete phase objects are content-addressed and deduplicated. Trials are never
overwritten. Each measurement records:

- state revision and snapshot;
- exact phase hashes and required panels;
- SLM serials and wavelength;
- camera provider/status, settings id, exposure, gain, scale, full scale and dimensions;
- actual z and z reference;
- raw-frame timestamps, data kind, file hash and pixel hash;
- physical configuration;
- recipe, trial role, repeats and settling time;
- calibration ids and analysis version.

`SYNTHETIC`, `REPLAY` and `EXPERIMENT` labels propagate from the provider. An
offline replay can retain `EXPERIMENT` only when its provenance establishes that
the source arrays are experimental; the acquisition route remains recorded as
replay.

Use `build_session_report()` to create a readable Markdown report without
changing raw evidence.
