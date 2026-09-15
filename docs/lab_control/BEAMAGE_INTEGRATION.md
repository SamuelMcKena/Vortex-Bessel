# Gentec Beamage integration

## Chosen route

The current official Gentec Beamage product resources expose a **BEAMAGE named
pipe example for .NET**, and state that PC-Beamage must be installed and running.
The implementation therefore defines `BeamageBridge` around that official route
and injects it into `BeamageCameraProvider`.

Primary vendor references checked on 15 September 2026:

- [Beamage-4M product and downloads](https://www.gentec-eo.com/products/beamage-4m)
- [Beamage-4M-FOCUS product and downloads](https://www.gentec-eo.com/products/beamage-4m-focus)
- [PC-Beamage user manual](https://www.gentec-eo.com/Content/downloads/user-manual/User_Manual_Beamage_V14.pdf)
- [Gentec-EO download centre](https://www.gentec-eo.com/resources/download-center)

No undocumented command names were guessed. No screenshot/screen-scrape route
is present.

## Implemented contract

The injected bridge must supply:

```python
connect()
disconnect()
start()
stop()
configure(exposure_us, gain)
read_quantitative_frame(timeout_s) -> (numpy_2d_array, metadata)
```

The provider validates the returned array through `CameraFrame`, carries exposure,
gain, full scale and z metadata, and labels it `EXPERIMENT`. The advanced GUI can
display it through the same live path used by dummy/replay providers.

## Current status

| Question | Answer |
|---|---|
| Provider interface implemented? | Yes |
| Official integration family selected? | Yes: PC-Beamage .NET named pipe |
| Full quantitative matrix required? | Yes; the bridge contract will not accept a screen image |
| Live image available with dummy/replay? | Yes |
| Live image available from physical Beamage now? | No; vendor bridge is not bound in this repository |
| Hardware verified? | No — `HARDWARE_UNVERIFIED` |
| Existing `.bmg` ingestion? | Yes; the strict v1 reader was previously replay-validated on 52 archived Beamage-4M files |

## Lab-PC completion steps

1. Install/run the current PC-Beamage version supported by the camera and obtain
   Gentec's matching named-pipe example from the product/download centre.
2. Confirm the example exposes a full numeric frame and acquisition/exposure
   operations for the installed version. Do not infer commands from older code.
3. Implement `BeamageBridge` in a Windows-only module using either `pythonnet`
   or a small signed .NET sidecar. Keep the protocol adapter isolated.
4. Record exact PC-Beamage, camera firmware and bridge versions in metadata.
5. Verify frame dimensions, orientation, pixel type, full-scale/saturation,
   exposure/gain readback and freshness against a saved numeric export.
6. Exercise start/stop/disconnect recovery and close the GUI during acquisition.
7. Mark the provider physically validated only after the results and raw frames
   are attached to calibration evidence.

If the current vendor API cannot return the matrix directly, the permitted
fallback is a documented automated **numeric** export followed by strict
ingestion. A rendered preview is never an acceptable scientific fallback.
