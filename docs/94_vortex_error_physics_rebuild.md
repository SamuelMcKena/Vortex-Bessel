# Phase 2E vortex physical-error physics rebuild

## Status

This branch supersedes the first `phase2e-vortex-physical-errors` report figures.
Those figures may be retained as development history, but they are not report
evidence.

The reason for supersession is not that every trend was necessarily false.  In
particular, a +/-1 mrad input pointing error can legitimately be dominated by
beam-axis steering.  The problem was that the figure was presented as a complete
physical misalignment response without exposing the SLM grating-order motion,
fixed Fourier iris, selected-order efficiency, approximation regime, or an
independent oblique-axicon benchmark.

No physical-error figure is accepted because it merely looks plausible.

## 1. Input pointing before the SLMs

For a collimated input wave with pointing components theta_x, theta_y,

    E_in(x,y) = A(x,y) exp[i k (sin(theta_x) x + sin(theta_y) y)].

This is consistent with the thin-axicon oblique-illumination starting point used
by Zhao & Li (Applied Optics 37, 2563-2568, 1998): an oblique incident wave is
multiplied by the axicon transmittance and subsequently diffracted.

However, our laboratory route contains two SLMs and a fixed +1-order Fourier
filter before the axicon.  The programmed carrier is therefore part of the
physics.

For a carrier G along +x, tangential momentum conservation gives

    kx,m = kx,in + m 2 pi G,

or

    s_x,m = sin(theta_x) + m lambda G.

The expected +1-order spatial frequency is

    f_x,+1 = G + sin(theta_x)/lambda.

The physical Fourier iris does not follow the order when the input beam points
away from its nominal direction.  It remains centred on G.  Therefore every
input-pointing study must report:

- expected +1-order centre;
- measured numerical spectral centroid;
- displacement relative to the fixed iris;
- displacement in the Fourier plane;
- selected-order spectral fraction / route efficiency;
- propagated beam only after those upstream diagnostics.

For the current canonical values lambda=1029 nm, G=6.25 lp/mm, f=300 mm and
iris radius=2.5 lp/mm:

- nominal +1 position is approximately 1.9294 mm from zero order;
- +1 mrad pointing shifts the order by approximately +0.3000 mm;
- this equals approximately 0.3887 of the modeled iris radius in spatial-frequency space.

The dedicated renderer is

    tools/render_vortex_input_angle_research.py

and the old generic renderer now refuses `input_beam_angle_x_rad`.

### LCOS angle response

The scalar route treats the reflective SLM in an unfolded wave-optics geometry.
This is sufficient for studying relative pointing around a calibrated nominal
incidence only if the LC electro-optic phase response does not materially change
across that perturbation.  Zhao et al., Chinese Optics Letters 16, 090701
(2018), measured little change at small incidence changes but substantial phase
curve changes above roughly 10 degrees.  The actual bench incidence and an
angle-appropriate NIR-149 LUT therefore remain calibration requirements for
large-angle claims.

## 2. Oblique axicon illumination

Small-angle thin-axicon propagation and full off-axis axicon aberration are not
the same validation question.

Zhao & Li (1998) provide analytic/diffraction treatment of a thin axicon under
oblique illumination.  Thaning, Jaroszewicz & Friberg (Applied Optics 42, 9-17,
2003) show that oblique illumination broadens the focal line and produces
astigmatic/astroid-caustic behaviour, confirmed by direct diffraction simulation
and experiment.

Therefore input-angle acceptance requires two regimes:

1. a small-angle benchmark in the thin-axicon regime;
2. a moderate-angle standalone benchmark that visibly recovers the known
   off-axis broadening/astigmatism trend.

The +/-1 mrad lab-sensitivity plot is not expected to show dramatic astigmatism
by itself; it is too small an angle for that to be a useful visual validation.

## 3. Rigid axicon tilt

A rigidly tilted optic is a nonparallel-plane diffraction problem.  The inherited
`rotated_thin_element_opd_small_angle` implementation is now **blocked for report
use**.

The required production backend is a tilted-plane angular-spectrum method,
following the coordinate-rotation approach of Matsushima, Schimmel & Wyrowski,
JOSA A 20, 1755-1762 (2003), or an equivalent accurately sampled formulation.
It must be validated against an independent direct diffraction calculation on a
small grid before report figures are authorised.

The generic atlas CLI therefore refuses the inherited axicon-tilt sweep on this
branch.

## 4. Refractive axicon cone angle

For a ray that enters a flat axicon face normally and leaves the conical face,
Snell's law gives

    n_ax sin(gamma) = n_ext sin(gamma + beta),

so

    beta = asin[(n_ax/n_ext) sin(gamma)] - gamma.

The exact radial direction sine is sin(beta).  The usual shallow thin-element
approximation is

    sin(beta) ~= (n_ax/n_ext - 1) tan(gamma).

For n_ax=1.458 in air:

- gamma=2 deg -> beta about 0.91667 deg and shallow relative error about 0.028%;
- gamma=20 deg -> beta about 9.91174 deg and shallow relative error about 3.16%.

Consequently the existing 2-degree Nathan/canonical branch may keep the shallow
phase for its intended regime, but a high-angle laboratory axicon must use a
separate explicitly bound physical profile and exact-Snell/two-surface model.
The meaning of any quoted `20 deg` manufacturer specification must also be bound
explicitly (base/physical angle versus apex-angle convention) before use.

## 5. Rounded / blunt axicon apex

Brzobohaty, Cizmar & Zemanek, Optics Express 16, 12688-12700 (2008), model a
real rounded tip using a hyperboloidal surface.  The rounded central region
produces a low-spatial-frequency refracted component that interferes with the
quasi-Bessel cone.  Their predicted axial modulation period is

    Lambda_z = lambda / [1 - cos(alpha_0)],

where lambda is the wavelength in the propagation medium and alpha_0 is the
Bessel cone angle.

Sahin, Turkish Journal of Physics 42, 47-60 (2018), specifically studies Bessel
and vortex-Bessel beams from a blunt-tip axicon and finds zero-order Bessel beams
substantially more sensitive than higher-order vortex beams under matched
conditions.

Rounded-tip report figures therefore require:

- sharp-tip convergence;
- an independent axisymmetric Hankel/Fresnel B0 reference;
- axial modulation period consistent with the analytic reference where its
  assumptions apply;
- matched B0/V1/V3 comparison showing the expected order-dependent sensitivity
  trend rather than assuming it.

## 6. Generic Zernike aberrations

Defocus, astigmatism, coma and spherical aberration remain useful as controlled
wavefront-error studies.  They are not allowed to stand in for a specific
misaligned physical optic unless that optic model derives the same wavefront.

## 7. Fidelity matrix

| Error / perturbation | Current status | Required physics before report use |
|---|---|---|
| Input beam pointing | candidate after reference checks | SLM carrier/order translation, fixed iris efficiency, thin-axicon benchmark, moderate-angle oblique benchmark |
| Input beam decentre / radius | candidate after convergence | rebuild upstream Gaussian through SLM/4F, not post-hoc reweighting |
| Fourier iris offset | physical sensitivity case | fixed Fourier-plane mask and measured/assumed physical scale |
| Axicon decentre | candidate after convergence | translated physical sag and clear-aperture binding |
| Rigid axicon tilt | blocked | rotated-plane angular spectrum + independent direct-diffraction validation |
| Rounded/blunt tip | candidate after reference checks | measured/assumed sag, independent radial reference, beat-period/order-sensitivity validation |
| 2-degree canonical axicon | shallow model acceptable in current scope | source sampling convergence already required |
| High-angle refractive axicon | shallow model blocked quantitatively | exact Snell/two-surface model and explicit manufacturer angle convention |
| Generic Zernike errors | diagnostic | explicit plane, normalization and RMS convention |

## 8. Report policy

The first `phase2e-vortex-physical-errors` plots are development diagnostics only.
The physics-rebuild branch will authorise each error family separately after its
reference checks.  Test success is software QA and does not by itself constitute
physical validation.

## References

- Zhao Bin and Li Zhu, "Diffraction property of an axicon in oblique illumination," Applied Optics 37, 2563-2568 (1998), DOI 10.1364/AO.37.002563.
- A. Thaning, Z. Jaroszewicz, A. T. Friberg, "Diffractive axicons in oblique illumination: analysis and experiments and comparison with elliptical axicons," Applied Optics 42, 9-17 (2003), DOI 10.1364/AO.42.000009.
- K. Matsushima, H. Schimmel, F. Wyrowski, "Fast calculation method for optical diffraction on tilted planes by use of the angular spectrum of plane waves," JOSA A 20, 1755-1762 (2003), DOI 10.1364/JOSAA.20.001755.
- O. Brzobohaty, T. Cizmar, P. Zemanek, "High quality quasi-Bessel beam generated by round-tip axicon," Optics Express 16, 12688-12700 (2008), DOI 10.1364/OE.16.012688.
- R. Sahin, "Bessel and Bessel vortex beams generated by blunt-tip axicon," Turkish Journal of Physics 42, 47-60 (2018), DOI 10.3906/fiz-1707-8.
- Z. Zhao et al., "Characterizing a liquid crystal spatial light modulator at oblique incidence angles using the self-interference method," Chinese Optics Letters 16, 090701 (2018), DOI 10.1364/COL.16.090701.
