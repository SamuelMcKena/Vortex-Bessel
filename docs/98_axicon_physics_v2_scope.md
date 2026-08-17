# Axicon physics v2: validation scope

This branch follows the carrier-aware rotated-plane numerical repair.  It does
not weaken or reinterpret the Phase 2E numerical power gates.

## Problems addressed

1. Fixed laboratory y-z/x-z slices can miss a translated beam and appear blank.
2. The old +/-0.5 degree rigid-tilt sweep is too weak to demonstrate the known
   oblique-axicon morphology and cannot be naively widened because the optical
   carrier exceeds the FFT-grid Nyquist frequency.
3. The old flat-tip sweep included 10 and 25 micrometre radii on a 10 mm / 1536
   grid (6.51 micrometre native spacing), which is not adequate spatial sampling
   for a local physical defect.
4. The old rounded-tip sweep exposed a vertical hyperboloid parameter rather
   than the physically interpretable radial hyperbolic curvature scale.

## New contracts

### Beam-following morphology

The publication-oriented diagnostic samples x-z and y-z fields directly from
the angular spectrum on coordinates following a tracked beam axis.  It does not
interpolate images.  Absolute steering remains in CSV metrics while morphology
is plotted relative to the tracked axis.

### Tip resolution

A non-zero local tip radius must span at least 12 native 2-D grid pixels in the
N=1536 production figures.  Current v2 sensitivity radii are 100, 200, 400 and
800 micrometres.  Smaller physical defects require a finer/multiscale solver and
must not be presented as resolved 2-D results.

Rounded-tip sensitivity is parameterised by the radial hyperbolic scale r_h in

    f(r) = tan(gamma) [sqrt(r^2 + r_h^2) - r_h].

The branch contains an independent high-resolution radial Fresnel integral and
requires agreement with the 2-D ASM in the shallow-angle regime.

### Large-angle rigid tilt

The optical carrier associated with 5-10 degree plane rotations is tracked
analytically while only the baseband envelope is sampled.  Absolute wavevectors,
not aliased sampled carriers, are used in the rotated-spectrum mapping.

The large-angle scalar thin-axicon response is checked against an independent
two-interface vector-Snell ray reference.  Passing this check establishes a
carrier-safe non-invariant scalar oblique response; it does **not** authorise an
absolute thick/refractive/vector claim.

## Remaining absolute-physics blockers

A full physical laboratory axicon prediction still requires:

- manufacturer angle convention / exact part geometry;
- clear aperture and centre thickness;
- wavelength-dependent refractive index;
- surface-by-surface vector Snell refraction and Fresnel transmission;
- measured tip/surface profile if tip-defect magnitudes are to be claimed;
- polarisation treatment where the physical cone angle makes scalar treatment
  insufficient.

`report_figures_authorised` remains false until those blockers are resolved.
