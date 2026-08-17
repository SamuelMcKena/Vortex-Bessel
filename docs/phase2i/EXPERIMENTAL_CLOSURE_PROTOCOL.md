# Phase 2I experimental closure protocol

Status: implementation / measurements pending

The purpose of Phase 2I is to replace synthetic-only comparison with a traceable laboratory dataset that can be compared to the calibrated digital twin without fitting coordinate systems until the pictures look similar.

## 1. Dataset unit

One experimental dataset is one controlled bench state.  It should have a unique dataset ID and a calibration bundle that identifies the hardware state used during acquisition.

Do not mix different SLM masks, axicon poses, analyzer configurations, camera gains/exposures or optical realignments inside one longitudinal comparison unless the change is explicitly represented and physically corrected.

Every quantitative file is SHA-256 hashed in the dataset manifest.

## 2. Quantitative camera export

Preferred quantitative formats are lossless numeric/radiometric exports:

- NPY / NPZ;
- CSV / TXT numeric arrays;
- TIFF when the exported TIFF values are the camera measurement values.

PNG/JPEG screenshots, GUI screen captures and colour-rendered beam maps are qualitative provenance only.  They must not enter numerical sim-to-experiment metrics.

The original raw/exported measurement file should be retained unchanged.  Derived crops and plots are separate outputs.

## 3. Camera coordinate calibration

Before absolute spatial comparison, populate the calibration bundle with independently measured:

- object-plane metres per camera pixel;
- camera rotation relative to laboratory x/y;
- laboratory-origin/reference pixel on the sensor;
- saturation level, where applicable;
- exposure time;
- attenuation state.

These quantities are not tuned by maximizing image correlation to the simulation.

## 4. Background and saturation

Acquire a background frame at the same camera exposure/readout configuration and, where relevant, the same attenuation state.

Every quantitative intensity frame references its background-frame ID in the dataset manifest.

Saturated pixels are invalid comparison samples.  Saturation must not be clipped and then treated as measured intensity.

## 5. Canonical reference plane

A real Phase 2I dataset declares one physical `canonical_z_ref_m` before comparison.

Once a real camera/sample plane is defined, this experimental plane supersedes a synthetic convenience plane for absolute bench comparison.  A different `z_ref` is not selected separately for each perturbation in order to maximize agreement.

At this plane the primary closure evidence is:

- measured 2-D intensity;
- simulation resampled onto calibrated camera pixels;
- signed residual;
- calibrated measured/simulated line profiles;
- correlation, normalized L2, centroid error and covariance error.

No agreement pass/fail threshold is invented by the ingestion layer.  An acceptance policy must be defined separately using repeatability and uncertainty evidence.

## 6. Measured longitudinal stack

A measured `x-z/y-z` stack requires multiple camera frames at distinct physical z positions for **one unchanged optical case**.

For the stack to preserve steering/decentre, all planes must share one laboratory transverse coordinate system.  Acceptable approaches include:

- a camera translated on a verified straight rail whose transverse registration is independently checked;
- an independently measured fixed fiducial/reference transformation for each z plane.

Not acceptable:

- manually centering the camera on the beam at every z and then calling the result a fixed-lab propagation map;
- using the measured beam centroid itself as the registration fiducial;
- mixing different exposures and independently peak-normalizing each z plane.

For an intensity-comparable stack, camera exposure must be identical across the stack unless an independently calibrated radiometric correction is supplied.  Background evidence is required for every plane.

Primary longitudinal heatmaps use the repository canonical convention:

- fixed laboratory coordinates;
- linear `turbo` intensity;
- one common stack peak;
- no per-z normalization;
- the canonical `z_ref` marked only as a reference guide.

## 7. Analyzer measurements

For vector analyzer closure, acquire the same beam case at nominal analyzer states 0, 45, 90 and 135 degrees while recording the **actual calibrated analyzer angles**, transmission and extinction ratio in the calibration bundle.

Do not substitute the nominal rotation-stage label for a calibrated angle if the experiment is intended to support quantitative vector claims.

For full Stokes including S3, a calibrated QWP/analyzer sequence is additionally required; four linear analyzer images alone do not determine S3.

## 8. Shack-Hartmann / SLM correction provenance

If a camera dataset is acquired after wavefront correction, the dataset must identify the correction iteration/map used.  The corresponding Shack-Hartmann reference/latest measurements and SLM registration belong to the same calibration state.

This prevents comparing a measured corrected beam to a simulation of a different correction state.

## 9. Recommended first laboratory closure sequence

Start with the least ambiguous scalar cases before the full vector study:

1. Gaussian, SLM structured phase disabled where possible;
2. ordinary Bessel / axicon case;
3. scalar vortex-Bessel case;
4. six-sector vector case;
5. radial/azimuthal analyzer cases only after the required polarization hardware is verified and calibrated.

At each stage, first close the camera coordinate/calibration and nominal-case morphology before fitting or interpreting perturbation sweeps.

## 10. Claim boundary

A successful software readiness check means the dataset has the information required to run a calibrated comparison.  It does **not** mean simulation and experiment agree.

Agreement metrics, uncertainty, repeatability and model discrepancy remain separate evidence and must be reported explicitly.
