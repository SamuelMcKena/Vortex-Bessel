# Physical change events

Use the **System / readiness** page or
`CalibrationRegistry.apply_physical_change()` immediately after a known bench
change. The registry marks only the direct targets and their dependency closure.

| Event | Directly invalidated | Examples preserved unless downstream dependency requires otherwise |
|---|---|---|
| `AXICON_REMOVED_REINSERTED` | axicon placement | SLM serial/geometry/phase path, carrier, camera pixel scale |
| `AXICON_XY_CHANGED` | axicon placement | Same stable device facts |
| `CAMERA_REMOUNTED` | sensor geometry, travel axis | SLM and carrier calibration, camera pixel pitch |
| `CAMERA_SETTINGS_CHANGED` | camera response, q=0/vortex references | SLM identity and phase path |
| `PINHOLE_CHANGED` | q=0/vortex references | Camera intrinsic scale and SLM identity |
| `RELAY_OPTIC_MOVED` | relay geometry | Camera intrinsic scale and SLM serial identity |
| `SLM_REPLACED` | SLM serial identity | Camera intrinsic calibrations |
| `LASER_ALIGNMENT_CHANGED` | q=0/vortex propagation references | Device identities and intrinsic pixel scales |
| `WAVELENGTH_CHANGED` | SLM phase path/response and q=0 reference | Camera intrinsic geometry |

Downstream invalidation is transitive. For example, axicon reinsertion also
makes q=0 propagation, vortex propagation, low-order correction, retrieved
axicon-specific residuals, golden vortex reference and verification stale. Tests
explicitly confirm it does **not** invalidate SLM serial identity, HEDS phase path,
camera pixel scale or carrier convention.

`UNKNOWN` remains `UNKNOWN` after a change; the event records its stale reason
but does not pretend that a calibration existed previously.

After a routine axicon reinsertion, the normal recommendation is quick recovery.
A moved relay, replaced SLM or failed phase-path check should escalate to broader
commissioning instead.
