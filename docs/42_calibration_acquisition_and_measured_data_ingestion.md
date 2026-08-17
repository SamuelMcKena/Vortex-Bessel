# Stage 9A — Calibration Acquisition and Measured-Data Ingestion

Structured acquisition, storage, ingestion, and pixel-space analysis tooling to collect the real
data that will later support a calibrated 4F forward model, effective-aberration inference, SLM
correction, and fused-silica pilot experiments.

**No optical physics is added here.** No thin-lens / physical 4F propagation, no +1-order field,
no Fourier-stop transmission / order efficiency / zero-order claims, no camera-imaging model, no
inverse correction, no neural network, no material response. Boundary: `n=1.0` free-space
optical/fluence diagnostics; `fourier_filter_physics_available=False`; `camera_model_enabled=False`;
`material_model_enabled=False`; `diagnostic_only=True`; `final_export_allowed=False`.

## Calibration families (`cslm_physical_axicon_calibration_campaign_v1.json`)

Stage 9A direct Fourier-plane carrier mapping requires a temporary diagnostic method at or
conjugate to the Fourier plane. The installed downstream final-focus camera is not direct
Fourier-plane access; it supports Stage 9A.1B empirical carrier-and-stop response
characterisation only.

| family | kind | identifies |
|---|---|---|
| F0 dark/background reference | `dark_frame` | camera background, noise floor, saturation reference |
| F1 input-beam record | `input_beam` | beam centre/diameter/ellipticity/rotation in **pixels** (no calibrated physical radius) |
| F2 SLM2 carrier / Fourier mapping | `fourier_plane_carrier_sweep` | **highest priority**: carrier sign, carrier→order-position mapping, stop centring, Fourier-plane coordinate calibration (x and y carrier sweeps; record zero/+1/−1) |
| F3 Fourier-stop scan | `fourier_stop_scan` | practical stop placement / clipping / selected-order operating point (measured only; no stop model) |
| F4 Gaussian-through-axicon baseline | `post_axicon_z_stack` | axicon Bessel formation for a Gaussian input (pixel z-stack) |
| F5 vortex-through-axicon atlas (ℓ=1,2,3) | `post_axicon_z_stack` | charge-dependent annular Bessel formation (only the charge varies) |
| F6 future controlled perturbations | `manual_observation` | tip/tilt, defocus, astigmatism, coma, trefoil, mask-centre, carrier, axicon offsets — **`planned_future_calibration` / `not_implemented_in_current_stage`** |

## Raw-versus-derived data policy

Raw camera files are **immutable source evidence**: ingestion copies the source byte-for-byte into
`<run>/raw/`, verifies the SHA256, and never transforms (crop/rotate/normalise) raw data. Generated
raw data is **not committed by default** (`.gitignore` in `data/calibration_runs/` and
`outputs/calibration_runs/`; only `.gitkeep`/README tracked). Any background/dark subtraction,
normalisation, crop, or rotation must be explicitly requested, saved as a *derived* artefact in
`<run>/derived/`, and recorded in `derived/processing_manifest.csv` — never written over raw.

## Acquisition package (`outputs/calibration_runs/<run_id>/`)

`run_manifest.json` (run id, timestamp, git commit, profile name/status, route/handoff mode, all
control values + units + status + provenance, governance flags, physical-4F status, camera status,
material-model status, claim boundary), `acquisition_plan.csv`, `capture_manifest_template.csv`,
`hardware_profile_snapshot.json`, `bench_inventory_snapshot.json`,
`coordinate_contract_snapshot.json`, and `experiment_package/` (bench setup sheet md+csv, camera
capture checklist, energy log, axicon alignment log, fused-silica pilot observation template,
operator notes). Raw subdirs (`raw/`, `manifests/`, `derived/`, `figures/`) live under
`data/calibration_runs/<run_id>/`.

## Capture metadata requirements

Each capture carries: id, kind, run id, file path, raw SHA256, status, timestamp, camera id +
camera-frame declaration, image units, z position, exposure/gain, saturation fraction, background
reference id, profile name, route/handoff mode, SLM mask ids, topological charge, carrier
frequency, axicon-enabled, notes. The capture status pipeline is
`planned → acquired_unverified → ingested → quality_checked → coordinate_calibrated → analysis_ready`
(or `rejected`). `validate_capture_manifest` rejects missing critical fields, invalid
kind/status, and duplicate ids.

## Camera-coordinate limitations & metric validity rules

Supported formats: PNG, TIFF, NumPy `.npy`. Pixel-space metrics (centroid, ring centre/radius,
dark-core fraction, azimuthal uniformity, radial profile, saturation/valid-pixel fractions, FOV
margin) are valid before camera calibration. **Physical-unit (µm/mm) metrics are blocked**
(`blocked_coordinate_uncalibrated`) until a declared camera scale **and** a named reference-plane
relation exist. The tooling never labels pixel values as µm/mm without a valid transform, never
infers optical phase from intensity, never infers aberration coefficients, and never claims a
material outcome. Annular ring fits are not forced on non-annular captures (`is_annular` /
`not_annular` flags).

## Comparison boundary

`compare_measured_to_model` performs an absolute, like-for-like physical comparison only when a
declared camera scale **and** a named reference-plane relation exist (and the capture is
coordinate-calibrated); otherwise it returns `comparison_not_physically_calibrated`. Normalised
shape descriptors are offered for exploratory use, explicitly labelled
`shape_only_diagnostic_comparison` / `not_absolute_physical_validation`. No fitting, inverse
correction, neural network, or model calibration occurs in this stage.

## Path from carrier sweep to 4F readiness

Family 2 measures the SLM2-carrier → Fourier-plane order position, which provides the
Fourier-plane physical-position coordinate convention and the carrier sign — the exact items that
block level C (initial scalar 4F readiness, see docs/41). Once that mapping plus the lens focal
lengths / apertures / distances are measured, level C can be revisited.

This statement applies only to direct Fourier-plane or conjugate-plane access. The installed
downstream final-focus camera is Stage 9A.1B evidence: it can record empirical carrier-and-stop
response for operating-point selection and repeatability checks, but it does not by itself
establish physical Fourier-plane coordinates or physical 4F readiness.

## Path from z-stack to later effective-aberration correction

Families 4 and 5 record Gaussian and vortex (ℓ=1,2,3) post-axicon z-stacks under a fixed camera
convention. With a calibrated camera scale these become the measured reference for a future
calibrated 4F forward model and effective route-error inference; nothing is fitted here.

## Fused-silica pilot logging boundary

The fused-silica observation template captures only **neutral observations** (sample id, material
grade, dimensions, surface prep, energy/rep-rate/scan settings, focus/position, observed track
continuity / feature symmetry / morphology / surface effect / void-crack presence / etch response
/ weld appearance, microscope file path, operator notes). **No calculated material predictions.**
