# Stage 8C.3R.5.3 — Measured Bench Inventory, Coordinate Conventions, and Physical-4F Readiness

Extends the R5.2 control/profile system (no parallel config universe) into a disciplined
bench record, a coordinate-convention contract, and a four-level physical-4F readiness gate.

**No optical transform is implemented**: no thin-lens, no Fourier propagation, no +1-order
field, no camera physics. Boundary unchanged: `n=1.0` free-space; `fourier_filter_physics_
available=False`; `diagnostic_only`; `final_export_allowed=False`.

## Bench-inventory schema (`bench_inventory.py`)

`BenchInventoryItem`: `component_id`, `display_name`, `component_type`, `route_location`,
`value`, `unit`, `provenance`, `source_type`, `source_reference`, `recorded_date`,
`uncertainty`, `coordinate_frame`, `status`, `required_for_initial_4F_model`,
`required_for_measured_bench_prediction`, `notes`. Provenance reuses the R5.2 vocabulary
(no new classes). Items are derived from the R5.2 control registry plus an evidence overlay
loaded from `configs/hardware/cslm_physical_axicon_bench_inventory.json`. Placeholder values
that exist in the repository demo config are recorded with provenance `diagnostic_placeholder`
and `source_type = repository_demo_config_default` — they are **not** measured. Every genuinely
unknown hardware value stays `value=null`, `provenance=unknown`.

## Coordinate-convention contract (`coordinate_contract.py`)

Ten declared frames: `lab_beam_frame`, `SLM1_pixel_frame`, `SLM1_phase_map_frame`,
`SLM2_pixel_frame`, `SLM2_phase_map_frame`, `physical_axicon_local_frame`,
`Fourier_plane_spatial_frequency_frame`, `Fourier_plane_physical_position_frame`,
`camera_sensor_pixel_frame`, `camera_object_plane_frame`. Nine inter-frame transforms.

The model's own frames (`lab_beam_frame`, the phase-map frames, the FFT spatial-frequency
frame) use an explicit **declared_model_convention** (centred grid, +x right, +y up, +z
propagation, right-handed; DC at array centre via fftshift). The physical **SLM pixel
frames**, the **Fourier-plane physical-position frame**, and the **camera frames** are
`unknown`: their orientation / flip / rotation / pixel-to-physical scale relative to lab is
not calibrated.

### Why coordinate mapping is mandatory

The Fourier-plane carrier-order position (where the +1 order lands and where a stop must be
centred) is set by the SLM2 carrier frequency *in physical SLM coordinates* and by the
spatial-frequency → physical-mm mapping (which needs lens focal length and wavelength). An
X/Y orientation, sign, flip, or rotation error mis-places the order and any later camera
comparison. The `SLM2_pixel → lab` and `Fourier-frequency → physical-position` transforms are
declared but **not modelled** and require calibration; until then they block measured-bench
readiness.

## Readiness levels (`evaluate_physical_4f_readiness`)

- **A — active CSLM diagnostic:** READY (executes with diagnostic placeholders).
- **B — ideal physical-axicon benchmark:** READY (executable; values are placeholder/estimated,
  provenance reported).
- **C — initial scalar 4F-model:** BLOCKED. Hard blockers are the 15 control values plus the
  Fourier-plane coordinate convention and the SLM2 transverse coordinate scale. C is false
  whenever any hard-blocker value is null *or* the Fourier-plane physical-position mapping is
  unknown — even if all values were present.
- **D — measured-bench / camera comparison:** BLOCKED. Additionally requires measured
  provenance, the declared `SLM2→lab` / `Fourier→lab` / `camera→lab` transforms, axicon
  location/orientation, camera pixel pitch / magnification / orientation, a defined reference
  plane, and a beam-profile calibration capture (or declared absent).

Each level reports `ready`, `blocked_by`, `warning_items`, `measured_items`,
`placeholder_items`, `unknown_items`, `next_required_measurements`. A prioritised
`measurement_checklist` orders the C blockers (priority 1) before the D extras (priority 2).

## Hard blockers for physical-4F implementation

`wavelength_nm`, `slm2_pixel_pitch_um` (SLM2 transverse scale), `slm2_carrier_frequency_cpm`
(in physical SLM coordinates), `slm2_to_lens1_distance_mm`, `fourier_lens1_focal_length_mm`,
`fourier_lens1_clear_aperture_mm`, `lens1_to_fourier_plane_distance_mm`,
`fourier_filter_centre_x_um`, `fourier_filter_centre_y_um`, `fourier_filter_radius_um`,
`fourier_filter_shape`, `fourier_plane_to_lens2_distance_mm`, `fourier_lens2_focal_length_mm`,
`fourier_lens2_clear_aperture_mm`, `lens2_to_output_plane_distance_mm` — plus the Fourier-plane
coordinate convention. In the current demo, the lens clear apertures and the SLM2 pixel pitch
are `unknown`, and the Fourier-plane physical-position mapping is `unknown`.

## Measured vs placeholder interpretation

A value is only `measured` if it carries provenance `measured` with evidence. Demo placeholder
geometry (distances, focal lengths) is `diagnostic_placeholder` and must not be read as a
measured bench. The demo inventory has **zero** measured items.

## Profile files

- `configs/hardware/cslm_physical_axicon_demo_profile.json` (R5.2 demo).
- `configs/hardware/cslm_physical_axicon_measured_bench_template.json` (blank; all null/unknown).
- `configs/hardware/cslm_physical_axicon_bench_inventory.json`
  (`profile_status = diagnostic_demo_inventory_not_measured_bench`; placeholder values with
  evidence fields; unknowns null).

## Claim boundary

Free-space `n=1.0` optical/fluence diagnostic; no physical 4F field is generated; changing 4F
inventory values updates readiness only; `final_export_allowed=False`.

## Required bench measurements (prioritised)

P1 (unblock initial scalar 4F): Fourier-plane physical-position coordinate convention; SLM2
pixel pitch / continuous transverse scale; lens-1 and lens-2 focal lengths and clear apertures;
SLM2→lens1, lens1→Fourier-plane, Fourier-plane→lens2, lens2→output distances; Fourier-stop
centre / radius / shape; carrier frequency in physical SLM coordinates; wavelength provenance.
P2 (unblock measured bench/camera): SLM2→lab and Fourier→lab and camera→lab transforms; axicon
location/orientation; camera pixel pitch / magnification / orientation; reference-plane
definition; a beam-profile calibration capture.
