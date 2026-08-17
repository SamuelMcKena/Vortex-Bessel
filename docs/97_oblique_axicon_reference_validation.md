# Oblique axicon reference validation

## Status

This note defines the validation boundary for `axicon_rigid_tilt_x` and
`axicon_rigid_tilt_y` in the Phase 2E vortex/Bessel system-error study.

The carrier-aware rotated-angular-spectrum repair has removed the previous
numerical power-loss artefact. That numerical success is necessary but is not,
by itself, evidence that a rigidly tilted refractive axicon is quantitatively
modelled.

**Report policy:** oblique axicon tilt remains **report-blocked for absolute
refractive claims**. The current production route is retained as a relative
scalar/diffractive sensitivity model and must pass the independent reference
checks below.

## Why a separate reference is required

A real axicon under oblique illumination is not equivalent to applying an
axisymmetric radial phase on a parallel laboratory plane. The ray/eikonal
geometry contains two refractive interfaces:

1. refraction through the flat entrance face;
2. refraction through a conical exit face whose surface normal varies with
   azimuth.

At normal incidence the outgoing rays lie on an axisymmetric cone. Under
oblique incidence the outgoing cone becomes azimuth dependent, which is the
geometrical precursor of the astigmatic broadening/caustic behaviour reported
for obliquely illuminated axicons.

The independent reference implementation is:

- `vbb_study/digital_twin/vortex_axicon_oblique_reference.py`

It performs vector Snell refraction at the flat and conical interfaces and
measures the azimuthal outgoing-cone anisotropy. It is intentionally independent
of the production wave propagator.

The corresponding wave/reference gate is:

- `tools/check_axicon_oblique_benchmark.py`

and its regression tests are:

- `tests/test_vortex_axicon_oblique_reference.py`

## Current numerical result

GitHub Actions run `31440647142` evaluated B0 on a 1024 x 1024, 10 mm window
for 0, 1 and 2 degree rigid y tilt. The benchmark passed.

At 2 degrees:

- required laboratory carrier: `33915.9346 cycles/m`;
- minimum tilted-plane spectral-power diagnostic: `0.99902988`;
- x-z correlation to zero tilt: `0.99963755`;
- y-z correlation to zero tilt: `0.99999873`;
- x-z/y-z correlation: `0.99963503`;
- wave width anisotropy diagnostic: `0.00122069`;
- independent two-interface Snell cone anisotropy: `0.00137406`;
- independent second-harmonic cone-radius fraction: `0.00015296`.

The important conclusion is therefore not that the 2 degree field is strongly
astigmatic. It is that the current wave route is **not spuriously invariant**
once an oblique angle large enough to be resolved on the declared sampled grid
is tested. The very small response in the existing +/-0.5 degree sensitivity
sweep is therefore not, on its own, evidence of a broken solver.

## Sampling boundary

A tilted plane carries a laboratory transverse carrier approximately

`f_tilt = sin(theta) / lambda`.

At 1029 nm, large literature-scale incidence angles can exceed the Nyquist
frequency of the current 10 mm FFT grid. The benchmark therefore refuses angles
whose carrier reaches 80% of Nyquist rather than allowing an aliased result to
be interpreted as physical.

Consequently, the present grid is suitable for the 0--2 degree reference gate,
but it is **not** a valid route for simply inserting a 5--10 degree tilt and
restoring that carrier on the same sampled field. A high-angle wave study needs
an envelope/nonuniform/locally tilted representation or another rigorously
validated sampling strategy.

## Hardware-parameter blocker

The current integrated route obtains `axicon_base_angle_deg = 2.0` from the
Phase 2A canonical hardware manifest. That value is marked `assumed` and
`calibration_required` and originates from the Nathan source-parity
configuration. It is not yet a measured/verified geometry contract for the
physical axicon used in the laboratory.

Do not silently change that value from a remembered or nominal product angle.
Before an absolute tilted-axicon prediction is authorised, bind the actual
optic with an explicit angle convention and, where required by the selected
surface model, thickness/clear-aperture information.

## Validation contract

A future absolute refractive-tilt backend must satisfy all of the following:

1. zero-tilt limit reproduces the accepted exact-Snell radial `kr`;
2. x/y rigid tilts are rotationally equivalent for the ideal axisymmetric
   optic;
3. numerical tilted-plane/co-ordinate transforms conserve represented spectral
   power above the existing `0.985` gate;
4. oblique response is non-invariant at a resolvable benchmark angle;
5. azimuthal anisotropy grows consistently with the independent two-interface
   Snell reference in the small/moderate-angle regime;
6. sufficiently large, adequately sampled oblique incidence reproduces the
   qualitative astigmatic/caustic broadening established in the literature;
7. no absolute bench claim is made until the actual axicon angle convention,
   refractive index, clear aperture and any surface/tip metrology required by
   the model are bound to evidence.

## Literature basis

- Z. Bin and L. Zhu, *Diffraction property of an axicon in oblique
  illumination*, Applied Optics **37**, 2563--2568 (1998).
- A. Thaning, Z. Jaroszewicz and A. T. Friberg, *Diffractive axicons in
  oblique illumination: analysis and experiments and comparison with
  elliptical axicons*, Applied Optics **42**, 9--17 (2003).
- J. Dudutis et al., *Aberration-controlled Bessel beam processing of glass*,
  Optics Express **26**, 3627--3637 (2018).

These references establish the physical expectation used by the independent
reference gate; they do not calibrate the specific laboratory optic.
