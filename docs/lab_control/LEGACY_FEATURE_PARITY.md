# Legacy SLM feature parity

## Audited sources

The v2 implementation was compared directly with:

- the attached phase-safe v0.5 GUI (`slm_lab_control/app.py`, `config.py`,
  `phase.py`, hardware backends/profiles and presets);
- repository v0.6 at `lab/commissioning-and-rapid-recovery` (`4376e93`);
- the preceding unified branch (`feature/unified-lab-control`);
- the supplied Gentec `NamedPipeClient-V1.00.12` C++ example.

The authoritative phase composer and HEDS backend remain shared code. The new
advanced SLM pages embed the complete `SLMControlPanel`; they are not a reduced
reimplementation.

## SLM feature matrix

| Feature | v0.5/v0.6 | v2 location | Decision |
|---|---:|---|---|
| Locked geometry, serial, wavelength, bit depth | Yes | SLM detail header/locked card | Retained unchanged |
| Locked order-selection carrier | Yes | Composer + locked card | Retained; never repurposed |
| Beam centre and phase-term rotation | Yes | SLM1/SLM2 Details | Retained |
| Reference pupil diameter | Yes | SLM1/SLM2 Details | Retained as a non-gating coordinate/support reference |
| Vortex enable and charge | Yes | Home quick card + Details | Retained and shared |
| Dedicated tip/tilt steering | Unified addition | SLM1/SLM2 Details | Retained; separate from carrier/coma |
| Digital axicon period/sign | Yes | SLM1/SLM2 Details | Retained |
| Focus focal length/sign | Yes | SLM1/SLM2 Details | Retained |
| Measured wavefront map/gain | Yes | SLM1/SLM2 Details | Retained |
| Zernike defocus, astigmatism, coma, spherical | Yes | SLM1/SLM2 Details | Retained |
| Sample-interface spherical correction | Yes | SLM1/SLM2 Details | Retained as an additive pupil-local term |
| Retrieved correction map/gain | Yes | Home enable + Details path/gain | Retained |
| Custom phase map/gain | Yes | SLM1/SLM2 Details | Retained |
| N-fold term order/amplitude | Yes | SLM1/SLM2 Details | Retained |
| Background gray/global gain | Yes | SLM1/SLM2 Details | Retained |
| Generate and complete-phase preview | Yes | Home + Details + compact | Retained; one composer and hash |
| Connect/cast/blank | Yes | Home + Details + compact | Retained; explicit cast only |
| Load/save dual-SLM preset | Yes | Presets + Details + compact | Retained with migration |
| HEDS and dummy backends | Yes | System | Retained |
| PNG/phase-file/direct gray/direct phase/auto transfers | Yes | System | Retained |
| Manual correction workbench | Yes | Compact GUI | Retained; advanced recipes are additional |
| Whole-panel circular pupil gate | Short-lived regression | Nowhere | Removed; old preset flag loads inert and is saved false |

## Shared-state and usability checks

- Home, both detailed pages and the optional compact window use the same
  `ExperimentStore`, `LabController`, phase bundle and HEDS handles.
- A programmatic update appears in every open view.
- Only the SLM named by an event is refreshed; camera frames and SLM2 status do
  not rewrite an in-progress SLM1 edit.
- Model-to-widget refreshes block Qt signals, preventing feedback loops.
- Spin boxes and combo boxes ignore wheel events but retain normal buttons,
  keyboard steps and typed entry.
- Configured and last-cast hashes remain visibly distinct so editing never looks
  like a hardware command.
