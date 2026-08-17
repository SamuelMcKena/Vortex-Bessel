# Phase 2H canonical evidence presentation sign-off

Date: 2026-08-12

This sign-off is presentation/evidence governance only.  It does not change the previously validated Phase 2H optical model or the synthetic study results.

## Canonical contract

Repository-wide convention is defined in `docs/CANONICAL_FIGURE_EVIDENCE_SPEC.md` and enforced by `vbb_study/reporting/evidence_conventions.py`.

For a comparative sweep the required primary pair is:

1. fixed-laboratory `x-z` and `y-z` longitudinal propagation;
2. transverse 2-D intensity plus exact 1-D profiles at one a-priori fixed `z_ref` shared by the sweep.

Primary intensity heatmaps use:

- linear intensity;
- `turbo` colormap;
- one common comparison-family peak shown as 0--1;
- fixed physical coordinates;
- no per-z or per-panel peak normalisation;
- no beam-following recentering in primary longitudinal evidence.

At `z_ref`, 1-D comparison profiles are divided by the nominal case's 2-D peak at the same physical plane.  Profiles may therefore exceed 1.  Vector line intensity is synthesized from the complex `Ex/Ey/Ez` fields rather than interpolating rendered intensity images.

Signed normalized vector quantities use a symmetric diverging scale.  Beam-following crops, log maps and own-peak-normalized views are supplementary diagnostics only.

Current synthetic Phase 2H reference plane: **`z_ref = 30 mm`**.

## Canonical Phase 2H evidence artifact

Workflow: `Phase 2H canonical figure evidence`

Run: **31585861094**

Validated head: **2ebfb26b954fdd1a322a2c9cfccdab78cbd8f6e2**

Artifact:

- name: `phase2h-canonical-evidence-pair`
- artifact ID: **9137102003**
- SHA-256: **946929843bbd0df35e3aec6e6a8c21c26849eec744dda155e37687e2e7f1bd71**

The artifact contains, for rotations about x and y:

- canonical fixed-lab longitudinal PNG;
- canonical `z_ref=30 mm` 2-D/profile PNG;
- profile CSV;
- raw NPZ numerical evidence;
- machine-readable manifest.

The render uses a fixed +/-1.2 mm primary detail crop while retaining the full raw field arrays in NPZ.

## High-resolution analyzer morphology artifact

The full 224-frame cylindrical-vector tilt metric study remains the validated systematic evidence.  A separate display-only morphology atlas is recomputed at higher numerical sampling rather than image-upscaled.

High-resolution configuration:

- source grid: `N = 192`, 3.0 mm window;
- output grid: `N = 768`, 4.5 mm window;
- output sampling: approximately **5.86 um**;
- display tilts: `-2, 0, +2 deg`;
- analyzers: `0, 45, 90, 135 deg`;
- radial and azimuthal states, `ell=1,3`;
- `z_ref = 30 mm`;
- PNG DPI: 300;
- displayed pixels use nearest rendering; no bicubic claim of extra physics resolution.

The analyzer annulus gate is held fixed in **physical units** at approximately 170 um minimum radius.  This preserves the physical meaning of the earlier 12-pixel / 7.2-mm-at-N512 rule when changing numerical sampling.  A first high-resolution attempt that changed this gate in pixel units was rejected because it selected a different inner annulus for radial `ell=3`; the accepted rerun fixes the physical-radius contract rather than relaxing the expected 6-petal result.

Accepted artifact:

- name: `phase2h-highres-analyzer-atlas`
- artifact ID: **9137093575**
- SHA-256: **f5ec2b6f074f5b754afbbb36c6694e67ca98678c7e5879fb26106a2de4c7d76d**

The accepted high-resolution atlas preserves the expected 2-petal (`ell=1`) and 6-petal (`ell=3`) analyzer structure for the displayed cases.

## Exact-head regression status

On the same optical-code head, the original workflow run **31585861082** passed all five Phase 2H jobs:

1. vector two-surface Snell/Fresnel/eikonal gates;
2. calibrated vector route / objective regressions;
3. radial/azimuthal analyzer-spot tilt study;
4. synthetic preview / arbitrary-z profiles;
5. systematic six-sector vector tilt study.

Therefore the canonical-presentation additions do not replace or bypass the underlying Phase 2H physics/regression gates.

## Claim boundary

These figures remain synthetic and `report_figures_authorised = false` for absolute laboratory claims.  The next stage is measured bench ingestion and simulation-to-experiment closure; laboratory calibration, not further cosmetic figure refinement, is now the limiting dependency.
