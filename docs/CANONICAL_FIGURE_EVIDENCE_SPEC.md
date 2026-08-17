# Canonical optical-field figure and evidence specification

Status: **repository-wide presentation/evidence contract**

This document defines the minimum evidence package for comparative structured-beam sweeps.  It exists to keep a figure's numerical meaning fixed when the beam family, perturbation or renderer changes.

## 1. Required evidence pair

Every comparative optical-field sweep must emit, at minimum, two primary figures:

1. **fixed-laboratory longitudinal propagation**: `x-z` and `y-z` intensity maps;
2. **one fixed physical reference plane** `z_ref`: transverse 2-D intensity plus exact one-dimensional profiles.

Neither figure replaces the other.  The longitudinal map establishes formation, steering and axial behaviour; the reference-plane figure establishes transverse morphology and quantitative cross-sections at one common physical location.

## 2. Choosing `z_ref`

`z_ref` is selected **before inspecting the perturbation sweep** and is shared by every comparison case in that study.

Priority order:

1. a real experimental observation/sample/camera plane, once known and calibrated;
2. a physically declared design plane that the study is testing;
3. for a synthetic sensitivity study, one nominal physics plane chosen a priori and recorded in the manifest.

It is forbidden to choose a different `z_ref` for each perturbation in order to make every case look its best.  A supplementary best-focus/best-morphology metric may be reported separately, but it does not redefine the canonical plane.

For the current synthetic Phase 2H vector-tilt study:

- `z_ref = 30 mm` downstream of the Phase 2H post-axicon boundary field;
- the same `z_ref` is used for the six-sector and cylindrical-vector tilt studies;
- this is a synthetic study convention, not yet the location of a verified laboratory plane.

## 3. Primary longitudinal heatmaps

Primary longitudinal evidence must obey all of the following:

- coordinates are fixed in the laboratory frame;
- no row-by-row centring, tracker warp or z-dependent coordinate transformation;
- `x-z` and `y-z` are both supplied;
- one fixed physical crop is used across comparable cases;
- intensity is displayed on a **linear** scale;
- intensity colormap is **`turbo`**;
- one common physical peak is used for all comparable heatmaps, displayed as `0 -> 1`;
- no per-case peak normalisation;
- no per-z normalisation;
- no gamma or logarithmic remapping in the primary figure;
- the common `z_ref` may be marked by one unobtrusive horizontal guide because it directly links the longitudinal and transverse evidence pair.

A full-field context map may accompany a tighter beam-detail crop.  The full raw numerical array remains authoritative regardless of the display crop.

Log-intensity maps are permitted only as explicitly labelled supplementary diagnostics.

## 4. Primary `z_ref` transverse map

The transverse map at `z_ref` must:

- use the same physical x/y limits for every comparison case;
- use a linear **`turbo`** intensity heatmap;
- use one comparison-family-wide 2-D peak for the heatmap scale, displayed as `0 -> 1`;
- not peak-normalise every panel individually;
- not recenter the physical coordinates for the primary laboratory-frame map;
- keep centroid/morphology markers out of the clean primary image unless they are the quantity being explicitly illustrated.

A morphology-centred crop is useful and allowed as a supplementary diagnostic, but it must be identified as beam-following/centred and may not replace the fixed-lab primary map.

## 5. Exact 1-D profiles at `z_ref`

For scalar fields the complex scalar field is evaluated on the requested physical line.  For vector fields each of `Ex`, `Ey`, `Ez` is evaluated and intensities are summed:

`I = |Ex|^2 + |Ey|^2 + |Ez|^2`.

The primary profile sampler uses direct discrete Fourier-series evaluation of the **complex field**, including sub-pixel coordinates.  It must not interpolate a rendered intensity image.

Two profile frames are retained where meaningful:

- **laboratory frame**: fixed lines through `x=0` / `y=0`, preserving steering and decentre;
- **morphology/beam frame**: lines through a declared physical beam axis/centroid, separating shape change from translation.

Primary comparative profiles are divided by the **nominal case's 2-D peak at the same `z_ref`**.  Therefore a perturbed line profile is allowed to exceed 1.  That is real intensity redistribution on the common nominal scale and must not be clipped away.

Shape-only own-peak normalisation is supplementary only.

## 6. Signed vector quantities

Signed normalized Stokes-like quantities use a diverging **`coolwarm`** map with a symmetric fixed range, normally `[-1, +1]`.

Ratios such as `|Ez|^2/I` must be masked below a declared intensity floor.  Undefined/dark-region ratios must never be painted as visually meaningful saturated structure.

## 7. Resolution and rendering provenance

A higher-resolution visual figure must come from a higher-resolution numerical field, not from image resampling presented as extra physics resolution.

Every evidence manifest must record, where applicable:

- computational `N`;
- physical window and `dx`/`dy`;
- wavelength and medium;
- longitudinal z range and `dz`;
- `z_ref` and its selection policy;
- display crop;
- heatmap normalisation value and policy;
- profile normalisation value and policy;
- whether any displayed crop follows a beam axis;
- PNG DPI;
- whether rendered-image interpolation was disabled or used only for display.

A high-resolution display rerun may coexist with a lower-cost systematic metric sweep.  The two must be labelled separately and the high-resolution rerun must preserve the same physical model and pass its sampling/flux/transversality gates.

## 8. Raw evidence and stable filenames

A canonical sweep should emit these stable outputs:

- `<family>__longitudinal_fixed_lab.png`
- `<family>__profiles_zref_<z>mm.png`
- `<family>__metrics.csv`
- `<family>__evidence.npz`
- `<family>__manifest.json`

The PNG is presentation.  CSV/NPZ numerical arrays and the manifest are the auditable evidence.

The manifest must contain the fidelity/calibration status and `report_figures_authorised` state.  A visually polished synthetic figure is not thereby a laboratory-calibrated figure.

## 9. Supplementary figures

The following are useful but supplementary unless a specific study promotes them:

- arbitrary-z transverse/profile sheets;
- log-intensity longitudinal maps;
- beam-following/morphology-centred heatmaps;
- centroid trajectory plots;
- axial peak and line-integrated intensity curves;
- Stokes maps and analyzer atlases;
- ray/eikonal geometry diagnostics;
- own-peak-normalised shape-only comparisons.

Supplementary figures may never silently change the normalization or coordinate convention of the primary evidence pair.

## 10. Phase 2H application

For Phase 2H the canonical primary pair is therefore:

- fixed-lab `x-z` + `y-z` from `0--60 mm`, with the same physical coordinates for every tilt;
- 2-D intensity + exact lab/centred line profiles at `z_ref = 30 mm`.

The dedicated analyzer-spot atlas is supplementary morphology evidence.  Its close-up crop may follow the beam centroid so the 2-spot / 6-spot structure is legible, but its source field must be recomputed at the declared higher numerical sampling rather than bicubic-upscaled from the lower-resolution systematic study.
