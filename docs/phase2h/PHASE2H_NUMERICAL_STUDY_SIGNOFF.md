# Phase 2H numerical vector-tilt study sign-off

Status: **NUMERICAL / SYNTHETIC STUDY COMPLETE — LAB CALIBRATION PENDING**

This sign-off applies to the Phase 2H macroscopic flat-entrance -> conical-exit vector refractive axicon model and the numerical sensitivity studies run from it. It does **not** authorize absolute predictions for the laboratory optic or report figures based on assumed hardware values.

## Accepted physical model

The authoritative solver is `vbb_study/digital_twin/vector_refractive_axicon_eikonal.py`.

Snell refraction is driven by the common eikonal / canonical wavevector `k = grad(Phi)`. Total structured-field Poynting is used for energy-flux accounting, not as the ray direction. The earlier Poynting-directed prototype was rejected by the Fermat/eikonal consistency gate and remains provenance only.

The accepted path contains two physical refracting surfaces, local s/p Fresnel transport, optical path, ray-tube Jacobian transport, fixed-laboratory-plane remapping, Maxwell transversality enforcement, flux closure, common-eikonal validity, TIR/fold/interpolation/sampling rejection and downstream vector-wave propagation.

## Canonical evidence convention

The numerical study uses a fixed physical reference plane `z_ref = 30 mm`, chosen a priori and shared by the systematic six-sector and cylindrical-vector analyzer studies. The required evidence pair is:

1. fixed-laboratory `x-z` and `y-z` longitudinal propagation, with no z-dependent recentering;
2. transverse 2-D intensity plus exact one-dimensional line profiles at the common `z_ref`.

Line profiles are evaluated componentwise from the complex `Ex`, `Ey`, `Ez` fields by direct discrete Fourier-series synthesis and then summed in intensity; rendered intensity-image interpolation is not used. Comparative profiles use a common nominal zero-tilt 2-D peak normalization.

## Systematic six-sector vector tilt study

Renderer: `tools/render_phase2h_systematic_vector_tilt_study.py`.

Synthetic fixture: dual-SLM six-sector segmented-vector route; 1029 nm; synthetic 2 degree base-angle plano-conical axicon; refractive index 1.458; clear radius 3 mm; centre thickness 3 mm; output N=512 over 7.2 mm; longitudinal range 0--60 mm in 2 mm increments; canonical `z_ref = 30 mm`.

Axis sweep:

- primary signed tilts: `0, +/-0.1, +/-0.25, +/-0.5, +/-1, +/-2 deg`;
- stress tilts: `+/-5, +/-10 deg`;
- rotations about x and y;
- 30 axis-sweep cases total.

Directional closure:

- fixed tilt magnitudes 1 and 2 deg;
- tilt-vector azimuths `0, 30, ..., 330 deg`;
- 24 additional cases.

Key numerical results for the primary +/-2 deg range:

- rotation-about-x cross-axis centroid response at z_ref is approximately `-0.02610 mm/deg`;
- rotation-about-y cross-axis centroid response is approximately `+0.02765 mm/deg`;
- zero-tilt-subtracted signed cross-axis steering is approximately `-0.3169 mrad/deg` for rotation about x and `+0.3260 mrad/deg` for rotation about y;
- primary sweep power ratio remains approximately `0.99869--1.00104` of the zero-tilt case;
- primary second-moment ellipticity is approximately `1.276--1.293`;
- peak intensity at the common z_ref varies substantially with tilt/direction (`~0.993--1.270` for the primary axis sweep) while total power remains nearly invariant, showing spatial redistribution rather than power creation.

The tilt-azimuth scan shows nearly direction-independent centroid-steering magnitude but a real directional dependence in transverse morphology / peak redistribution. At 2 deg, the z_ref peak spans approximately `1.127--1.353` of nominal as the tilt-vector azimuth changes.

Across the complete axis sweep, representative solver maxima are:

- `|final_flux_closure_ratio - 1| <= ~4.4e-16`;
- final transversality residual `<= ~1.1e-17`;
- required Nyquist fraction `<= ~0.622`;
- common-eikonal component disagreement `<= ~1.45e-3`;
- reconstructed-gradient error `<= ~1.36e-3`.

Authoritative artifact from workflow run 31539304891:

- `phase2h-systematic-vector-tilt-study`
- artifact ID `9120098839`
- SHA-256 `043d14e2bccc4911a0bd2b783ef00ba175644528c7345b15387cd022a52f0106`.

## Radial / azimuthal analyzer-spot tilt study

Renderer: `tools/render_phase2h_cylindrical_analyzer_tilt_study.py`.

The cylindrical-vector source is a Gaussian-envelope generalized polarization state with local linear-polarization angle `|ell|*phi + delta`; no axicon/Bessel phase is pre-applied. The physical Phase 2H refractive axicon therefore creates the downstream Bessel propagation exactly once.

Study matrix:

- radial and azimuthal states;
- `ell = 1, 3`;
- rotations about x and y;
- tilt `-2, -1, -0.5, 0, +0.5, +1, +2 deg`;
- ideal linear analyzers at `0, 45, 90, 135 deg`;
- canonical `z_ref = 30 mm`;
- 224 analyzer frames total.

Angular harmonic analysis is performed on the strongest sufficiently resolved Bessel annulus beyond 12 output pixels rather than blindly using an under-sampled first ring. This is an observable-resolution policy only; it does not alter the optical field or refractive solver.

Results:

- **224/224 analyzer frames retain the expected dominant harmonic**;
- `ell=1 -> 2` analyzer petals;
- `ell=3 -> 6` analyzer petals;
- no harmonic switching anywhere in the full +/-2 deg sweep;
- maximum symmetry-wrapped analyzer-pattern orientation shift is ~0.92 deg;
- resolved analysis-ring radii are ~183--201 micrometres with ~372--466 grid samples in the annulus;
- family power remains within approximately `0.999956--1.000064` of its own zero-tilt reference;
- maximum required Nyquist fraction is ~0.442;
- maximum common-eikonal component disagreement is ~1.2e-5;
- maximum reconstructed-gradient error is ~3.9e-5;
- final flux closure remains at machine precision and transversality residuals remain ~1e-17.

These cylindrical-vector states remain **simulation-only target states** unless the laboratory branch contains the required calibrated polarization-converting hardware. The study does not silently claim that two phase-only SLMs with a shared director axis can synthesize arbitrary radial/azimuthal states unaided.

Authoritative artifact from workflow run 31539304891:

- `phase2h-cylindrical-analyzer-tilt-study`
- artifact ID `9120076053`
- SHA-256 `0324f29abe7bb722cf2cc55d7d79a7a6e06c89e104e86c08d152f16f64c0d3a0`.

## Final CI

Workflow run: **31539304891**

Validated study head: **84d48703b3b72256002ddc4e367782e41619d6de**

All five jobs passed:

1. Vector two-surface Snell/Fresnel/eikonal gates;
2. calibrated vector route / objective regressions;
3. synthetic preview and arbitrary-z profiles;
4. systematic six-sector vector tilt study;
5. radial/azimuthal analyzer-spot tilt study.

The final physics job reports 19 Phase 2H surface/reference/study tests passed, plus 8 existing scalar two-surface refractive reference tests. The independent vector/scalar benchmark reports `hard_pass = true`.

Additional final-run artifacts:

- vector/scalar reference: artifact ID `9120062131`, SHA-256 `6af1cfae68d96e4ff116a629e31abc8f5a46debe0ccfdfe326d57a5ef9b2c1b6`;
- arbitrary-z profile planes: artifact ID `9120073044`;
- Phase 2H preview figures: artifact ID `9120072017`.

## Claim boundary

The following statement is authorized:

> The **numerical/synthetic Phase 2H vector refractive axicon-tilt study is complete within the implemented flat-first macroscopic common-eikonal model**.

The following are **not** authorized yet:

- treating the synthetic 2 degree axicon as the laboratory axicon;
- absolute bench-error magnitudes before real axicon geometry is measured or vendor-verified;
- report-authorized experimental figures based on assumed hardware values;
- simulation/experiment agreement before camera/analyzer data are ingested;
- material-modification predictions;
- cone-first axicon orientation;
- full-volume Maxwell, ghost/reflection-stack, nonlinear, plasma or thermal material physics.

`report_figures_authorised = false` remains intentional until the real axicon and upstream vector bench are calibrated and experimental evidence is bound to the model.
