# Nathan MODE 2U3 - SLM Phase Calibration Bridge (docs/75)

**Status:** calibration bridge only. No calibration values are fabricated here; the phase stroke
and command-to-phase response of the two PLUTO-2.1-class panels at the actual laser wavelength are
`unresolved_requires_calibration` until this procedure is executed.

## Why

The repository documents 8-bit addressing for the PLUTO family but contains no wavelength-specific
phase-stroke or LUT record (manufacturer register M_SLM1_SPEC/M_SLM2_SPEC: unknown; bench register
B_SLM_PHASE_RESPONSE: unknown). The operating wavelength is the actual PHAROS value (1029 nm; the
Nathan source model rounds to 1030 nm). An exact 2*pi stroke must NOT be assumed.

## Method A - interferometric phase calibration

1. Illuminate the panel with the aligned linear polarisation at the bench incidence angle.
2. Display a two-zone mask: left half fixed at command 0, right half swept over the full drive
   domain (0..255 for uint8).
3. Interfere the two halves (Michelson arm or common-path lateral shear onto the camera).
4. Track the fringe shift of the swept half versus command: `phase(command)` directly.
5. Repeat per panel (SLM-H, SLM-V); record temperature and incidence angle.

## Method B - binary-grating diffraction-efficiency calibration

1. Display a binary grating alternating command 0 and command c (period >= 8 px).
2. Measure first-order power in the Fourier plane versus c.
3. Invert `eta(c) proportional to sin^2(delta_phi(c)/2)` for the phase difference (resolve branch
   by monotonicity from small c).
4. Cross-check against Method A near half-stroke.

## Target mapping

`desired phase (rad) -> calibrated hardware command`, wrapped over the measured usable stroke.
If the usable stroke at 1029 nm is below 2*pi, the wrapped mapping must be validated explicitly
(display a known 0..2*pi ramp and confirm first-order efficiency) before any mask is exported.

## Record schema

The machine-readable schema (no fabricated values) is stored at
`outputs/figures/digital_twin/nathan_mode2u3_hardware_closure/01_phase_calibration/slm_phase_calibration_schema.json`:

```json
{
  "schema": "nathan_mode2u3_slm_phase_calibration",
  "version": 1,
  "status": "unresolved_requires_calibration",
  "fabricated_values": false,
  "record_fields": {
    "slm_identity": {
      "make": null,
      "model": null,
      "serial": null,
      "panel_role": "SLM-H | SLM-V"
    },
    "wavelength_m": null,
    "timestamp_utc": null,
    "calibration_id": null,
    "command_domain": {
      "kind": "uint8 grey level | float phase",
      "min": null,
      "max": null
    },
    "measured_phase_response_rad": "array of (command, phase_rad) samples",
    "usable_phase_stroke_rad": null,
    "wrapped_mapping": "phase_rad -> command lookup covering [0, 2pi) after stroke check",
    "interpolation_method": "monotone cubic (PCHIP) or linear; recorded, not assumed",
    "residual_phase_rms_rad": null,
    "environment": {
      "temperature_C": null,
      "incidence_angle_deg": null,
      "polarisation_alignment_note": null
    }
  },
  "methods": {
    "A_interferometric": "split the panel into a static reference half and a swept half; interfere both halves (Michelson or common-path shear); fringe shift vs command gives phase(command) directly",
    "B_binary_grating": "display a binary grating alternating command 0 and command c; first-order diffraction efficiency eta(c) = sin^2(delta_phi(c)/2) inverts to the phase difference; sweep c over the full drive domain; appropriate for phase-only panels at near-normal incidence"
  },
  "acceptance": {
    "usable_stroke_requirement_rad": "greater than or equal to 2*pi at the operating wavelength, else wrapped mapping must be validated",
    "residual_rms_target_rad": 0.05
  }
}
```

Acceptance: usable stroke >= 2*pi at 1029 nm (or validated wrapped mapping) and residual phase RMS
<= 0.05 rad. M2S showed 8-bit quantisation and even 16-level phase pass the strict gate, so the
calibration target is comfortable.
