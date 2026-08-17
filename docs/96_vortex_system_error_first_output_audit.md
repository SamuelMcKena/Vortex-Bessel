# Vortex system-error first-output audit

## Status

The first full N=1536 B0/V1/V3 system-error bundle executed successfully, but it is **not report evidence**.  The audit separated physically sensible signatures from numerical/model-fidelity failures before any publication-facing interpretation.

## Behaviours retained as sensible screening signatures

- Beam lateral decentre translates/asymmetrically illuminates the system without being represented as an input pointing phase.  The propagated Bessel line remains broadly parallel rather than reproducing the dedicated pointing-error diagonal.
- Beam-radius changes primarily alter the useful axial Bessel region and its intensity envelope.
- Beam ellipticity changes the input footprint; both x-z and y-z views are required to diagnose the downstream asymmetry.
- 4F iris offset/radius changes affect higher-order vortex cases more strongly than B0 because the selected-order spectrum is broader.
- Thin-lens decentre produces prism-like steering, distinct from iris offset and lens despace.
- Rounded-tip axicon cases generate strong axial modulation.  B0 is more strongly affected than V1/V3 in the first screening bundle, consistent with the intended literature benchmark.
- Axicon base-angle and refractive-index sweeps change the exact-Snell radial wavevector monotonically.

## Presentation/metric deficiencies found

The original suite wrote only a single-plane peak/power/centroid and 4F selected fraction to CSV even though dense x-z/y-z propagation was calculated.  That is insufficient to validate a physical error signature.

The revised suite therefore writes:

- x-y, x-z and y-z correlations relative to nominal;
- x-z and y-z fitted line-centre slope/intercept/span;
- line width and active axial length;
- normalized axial-peak residual RMS and correlation;
- exact axicon radial wavevector;
- output-power ratio relative to nominal;
- per-transform rotated-plane spectral-power ratios and interpolation model.

It also plots y-z in addition to x-z, uses wider propagation ROIs for translating/steering families, and shows input phase rather than input intensity for curvature-only errors.

## Rotated-plane numerical failure in first bundle

The first implementation resampled rotated angular spectra with bilinear interpolation.  In the uploaded N=1536 bundle this produced large artificial losses for rigid lens/axicon tilt even though no absorptive element was present.  Examples included roughly half the nominal propagated power for some +/-0.25 degree lens-tilt cases while the fixed Fourier iris still transmitted essentially all of the selected order.

This is treated as a numerical failure, not a physical prediction.

The backend has been replaced by cubic-spline interpolation of the real and imaginary angular spectrum with explicit spectral-power bookkeeping.  An independent smooth-Gaussian round-trip check at 0.5 degree gives approximately 0.99924 returned power and >0.999998 field overlap, instead of the large bilinear attenuation.  A regression test and a hard numerical audit gate now prevent rigid coordinate rotation from masquerading as optical absorption.

This change is consistent with the tilted-plane diffraction literature: Matsushima et al. formulate the method using Fourier-domain spectrum rotation and interpolation, while later NUFFT work explicitly identifies interpolation error as a major accuracy limitation at increasing rotation angle.

## Physical-fidelity boundary after numerical repair

Improving rotated-spectrum numerics does **not** make rigid axicon tilt quantitatively validated.  The scalar rotated-plane model still omits full two-surface vector Snell/Fresnel refraction through the real refractive axicon.  Published oblique-axicon work reports broadened focal segments and astigmatic/astroid caustics, so those behaviours remain an independent morphology benchmark before any report claim.

Similarly, SLM LUT/SPNU/fringing magnitude, lens OPD/clear apertures, axicon aperture/surface profile, camera response and laser jitter remain calibration/data-driven blockers rather than invented constants.

## Expanded sweep coverage

The revised registry adds missing x/y or component-specific counterparts, including:

- beam lateral decentre y;
- common x+y curvature and y-only curvature;
- SLM1 hologram offset y;
- SLM2 carrier-scale error;
- panel-specific phase-stroke/fringing sweeps and y-fringing sensitivity;
- 4F iris offset y;
- L1/L2 focal-length error;
- L1/L2 decentre y;
- L1/L2 rigid tilt x;
- axicon lateral decentre y;
- axicon rigid tilt x.

All remain screening sensitivity values unless an actual measured tolerance/calibration is supplied.
