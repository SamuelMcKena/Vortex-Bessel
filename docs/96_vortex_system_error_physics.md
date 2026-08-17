# Vortex/Bessel physical system-error model — Phase 2E research branch

## Route

The research route is now

`Gaussian beam -> SLM1 -> SLM2/carrier -> explicit 4F relay + fixed +1 iris -> axicon -> free space`.

The branch does not overwrite accepted Phase 2A/2B/2C artifacts.  Its purpose is
to determine the physical signature of individual bench errors before those
errors are admitted into the report atlas.

## Input beam

The input Gaussian is evaluated in a beam-fixed transverse basis.  This permits:

- lateral decentre without inventing a steering phase;
- finite pointing angle with the correct projected footprint;
- independent x/y 1/e field radii (ellipticity/astigmatic Gaussian state);
- finite x/y radii of curvature (collimation/divergence error).

Small transverse beam/axicon decentre has an external validation target: Dufour
et al., *Study of the Transversal Misalignment of an Axicon*, Frontiers in Optics
2009, report that small displacements preserve focal-line resolution and keep the
line approximately parallel to the optical axis.

## SLMs

Pattern translation/rotation/scale are applied to the commanded hologram
coordinates before pixelation.  Phase quantisation, finite stroke and phase bias
are explicit.  Optional measured arrays are accepted for:

- per-panel grey-level -> phase LUT;
- per-panel static phase/wavefront map.

Spatial phase-response nonuniformity and device flatness are therefore not
replaced by arbitrary Zernikes when measured maps become available.  Relevant
calibration literature includes Reichelt, Applied Optics 52, 2610-2618 (2013),
Lu et al., Applied Optics 55, 7796-7802 (2016), and Xu et al., Optics Letters 43,
2993-2996 (2018).

A direction-dependent convolution surrogate is provided for LC fringing-field
pixel crosstalk.  Its kernel width defaults to zero and is not a measurement of
the HOLOEYE panels.  It must be fitted to panel diffraction data before absolute
claims.  This follows the modelling strategy of Lingel, Haist & Osten, Applied
Optics 52, 6877-6883 (2013), who show that fringing blurs sharp pixel phase edges
and changes hologram diffraction efficiency.

## Explicit 4F relay

The old collapsed `FFT -> mask -> IFFT` model cannot represent physical lens
spacing errors.  The research backend therefore propagates through

`object -> L1 -> Fourier plane/iris -> L2 -> output`

using angular-spectrum propagation and thin-lens phases.

The fixed iris is centred on the nominal +1 carrier order, not the optical axis.
For the canonical 1029 nm, f=300 mm, G=6.25 lp/mm model this centre is about
1.929 mm from the zero order.  Iris offset is an error relative to this physical
centre.

Supported physical errors:

- iris x/y offset and opening;
- L1/L2 axial despace;
- L1/L2 focal-length scale;
- L1/L2 decentre;
- finite lens aperture;
- user-supplied lens OPD maps;
- rigid L1/L2 tilt via rotated angular-spectrum mapping onto the local lens plane.

The tilted-lens implementation remains a scalar/paraxial thin-lens model.  Large
thick-lens tilts require surface-by-surface refraction.  Real Fourier-transform
lens aberrations are known to disturb generated beam-shaping patterns; see Zhang
et al., Applied Optics 54, 8891-8898 (2015).

## Tilted optical planes

Rigid plane tilt uses spectral coordinate rotation following Matsushima,
Schimmel & Wyrowski, JOSA A 20, 1755-1762 (2003).  The destination regular
angular spectrum is obtained by rotating wavevectors and bilinearly resampling
the source spectrum.  The Jacobian |fz_source/fz_destination| is applied.

This backend is used for thin-lens and axicon plane tilt.  It is scalar and does
not by itself supply full vector Fresnel coefficients or thick-element surface
refraction.

## Axicon

The sharp ideal cone uses the exact normal-incidence Snell cone deflection rather
than the shallow `(n-1) tan(gamma)` slope.  Physical errors supported are:

- lateral translation of the axicon/apex relative to the beam;
- rigid plane tilt using the rotated angular-spectrum backend;
- clear aperture, when a measured radius is supplied;
- base-angle scale;
- refractive-index scale;
- hyperboloidal rounded tip;
- flat/blunt truncated tip;
- optional measured surface-height error map.

Rounded/flat defects are applied as the defect phase *relative to the sharp cone*,
so the far-from-apex field keeps the exact Snell cone angle.  For shallow cone
angles this is a suitable scalar manufacturing-defect model.  It is not promoted
to quantitative high-angle refractive prediction without a full surface model.

Brzobohaty, Cizmar & Zemanek, Optics Express 16, 12688-12700 (2008), show that a
rounded tip creates an additional refracted component which interferes with the
quasi-Bessel component and causes axial modulation; their calculation uses
Hankel-transform propagation.  Thaning, Jaroszewicz & Friberg, Applied Optics 42,
9-17 (2003), show that oblique axicon illumination broadens the focal line and
can produce astigmatic/astroid-caustic behaviour.

## Executable sweep families

`vortex_system_error_sweeps.py` contains controlled screening values for:

- beam lateral decentre;
- beam radius;
- beam ellipticity;
- beam curvature/collimation;
- SLM1 hologram offset;
- SLM2 carrier rotation;
- SLM phase stroke;
- SLM fringing surrogate;
- 4F iris offset/radius;
- L1/L2 despace;
- L1/L2 decentre;
- L1/L2 rigid tilt;
- axicon lateral decentre;
- axicon rigid tilt;
- rounded and flat apex defects;
- axicon base angle and refractive index.

These sweep values are sensitivity values, not measurements.

## Still data/calibration driven

The code deliberately refuses to invent values for:

- actual SLM phase LUT/stroke at 1029 nm and bench incidence/polarisation;
- static SLM SPNU/flatness maps;
- absolute fringing-field kernel parameters;
- lens OPD maps;
- axicon clear aperture;
- axicon surface metrology;
- exact meaning of the physical lab axicon's quoted angle;
- camera physical scale/response;
- pulse-energy and wavelength jitter distributions.

Objective/sample errors remain in the separate vector Debye/interface branch and
must not be folded into this centimetre-scale source propagation model.

## Report policy

System-error figures are not automatically report evidence because the script
runs.  A family becomes report-eligible only after its nominal route and expected
failure signature pass an independent physical/numerical validation gate.  Final
morphology figures must also use the converged high-resolution source sampling,
not the historical N=512 grid.
