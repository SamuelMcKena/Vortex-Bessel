# Device providers

## SLM

`SlmProvider` exposes connection, status, complete-phase cast, blank and
disconnect operations. One provider owns both panels, preventing two in-process
clients from competing for HEDS handles.

| Provider | Status | Notes |
|---|---|---|
| `DummySlmProvider` | SOFTWARE-TESTED | Persists exactly what would be transferred; never calls hardware |
| `HedsSlmProvider` | HARDWARE-UNVERIFIED after refactor | Wraps the preserved v0.6 backend without changing phase conversion or transfer semantics |

The HEDS provider refuses to claim phase mode unless `setWavelength(1030)`
succeeds and `getWavelength()` agrees within 0.5 nm. Strict direct-phase mode
does not silently fall back to generic image data. `auto` is the only mode that
permits the documented fallback behavior.

## Camera

All camera providers return a finite, real two-dimensional `CameraFrame`.
Rendered screenshots and RGB/palette images are rejected as quantitative data.

| Provider | Status | Data label |
|---|---|---|
| `DummyCameraProvider` | SOFTWARE-TESTED | `SYNTHETIC`; deterministic camera-like development frames, not optical propagation |
| `ReplayCameraProvider` | REPLAY-VALIDATED | Caller declares `REPLAY`, `EXPERIMENT`, or `SYNTHETIC`; original source path is retained |
| `BeamageCameraProvider` | SOFTWARE-TESTED / HARDWARE-UNVERIFIED | Official PC-Beamage pipe client; vendor BMP is `LIVE_PREVIEW_ONLY` and blocked from formal evidence |

Replay accepts strict BMG plus numeric NPY/TXT/CSV and unscaled grayscale
BMP/PNG/TIFF. It does not turn a colour screenshot into data.

The PC-Beamage client uses the supplied C++ command set over
`\\.\pipe\pipe_beamage`. It keeps timeouts and cleanup explicit and reports
vendor measurements/positions. Exposure and gain are read/controlled in
PC-Beamage because setter commands are absent from the supplied example.

## Stage

`StageProvider` separates recipe motion from beam analysis:

- `ManualStageProvider` returns an operator prompt and records a confirmed z;
- `DummyStageProvider` moves immediately for deterministic tests/demo;
- a future motorised implementation can implement the same `request_move()` and
  `confirm_position()` contract without rewriting capture or beam-walk code.

## Failure behavior

Provider failures raise a recoverable `ProviderError` or `ProviderUnavailable`.
They update state to `ERROR` with the actual message. They never manufacture a
connected status, experimental frame, phase-mode verification or hardware cast.

## Future broker

The provider/controller boundary is the future broker seam. Move
`LabController` into a single service process, serialise commands/state events,
and retain the existing clients. Do not put HEDS ownership separately into each
GUI or CLI process.
