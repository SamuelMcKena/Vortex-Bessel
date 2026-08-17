# PHASE 2A Error-Injection Plane Registry

**Status:** authoritative plane/operation registry for current laboratory perturbations.

A physical upstream perturbation acts on the complex field at its declared plane before downstream propagation. Camera noise and display shifts remain post-propagation diagnostics and cannot stand in for beam tilt, decentre, or misalignment.

| error_id | injection plane | operator | field? | timing | class | routes | status |
|---|---|---|---:|---|---|---|---|
| `input_beam_decentre` | before SLM1 | translate input amplitude E(x-dx,y-dy) | true | before | physical | scalar,vector | active |
| `input_tilt` | before SLM1 | multiply by exp(i(kx*x+ky*y)) | true | before | physical | scalar,vector | active |
| `hologram_offset` | corresponding SLM phase plane | translate phase-mask coordinates | true | before | physical | scalar,vector | active |
| `slm_phase_error` | SLM phase plane | add deterministic phase error delta_phi(x,y) | true | before | physical | scalar,vector | active |
| `slm_quantisation` | SLM phase plane | nearest allowed phase level | true | before | physical | scalar,vector | active |
| `iris_offset_radius` | 4F Fourier plane | translate/resize hard spectral aperture | true | before | physical | scalar,vector | active |
| `axicon_decentre` | axicon plane | translate conical phase origin | true | before | physical | scalar,vector | active |
| `axicon_tilt` | axicon plane | multiply by linear phase ramp after conical phase | true | before | physical | scalar,vector | active |
| `objective_pupil_clipping` | objective pupil | multiply by circular pupil mask | true | before | physical | scalar,vector | active |
| `low_order_aberration` | physical pupil/aberrating plane | multiply by exp(i*2*pi*sum(c_j Z_j)) | true | before | physical | scalar,vector | active |
| `sample_interface_tilt` | sample surface | tilted interface coordinate/phase operator | true | before | physical | scalar through-sample | calibration_limited |
| `camera_noise` | camera plane | seeded detector-noise operator on measured intensity | false | after | diagnostic | all observation routes | active |
| `post_processing_display_shift` | display only | translate rendered intensity array | false | after | diagnostic | all display routes | active |

Machine-readable source: `outputs/validation/phase2a/error_injection_registry.csv`.
