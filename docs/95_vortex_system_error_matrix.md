# Vortex-Bessel system error matrix

## Policy

Pointing error is only one error family. Every error in the bench must be introduced at the physical plane where it occurs and validated against either an independent solver, an analytic result, measured calibration data, or a published physical signature. Generic Zernike phase terms are retained only as labelled wavefront-sensitivity studies and must not be presented as the physical consequence of a named misaligned optic unless a separate model derives that aberration.

The source-scale bench is

`laser/input state -> SLM1 -> SLM2 -> common 4F/+1-order filter -> axicon -> free-space Bessel region`.

The objective/sample branch is separate and uses the Phase 2C vector/Debye/interface machinery.

## Error families to implement and validate

### 1. Input beam state

- **Input pointing**: oblique Gaussian/plane wave before SLM1. Must move the SLM diffraction order according to tangential momentum conservation, interact with the fixed 4F iris, then generate the off-axis axicon field. Small-angle steering is expected; larger obliquity requires the published axicon oblique-illumination benchmark.
- **Lateral beam decentre**: translate the Gaussian while leaving its direction nominal. This is not steering. For small relative beam/axicon offsets the literature reports a focal line that remains approximately parallel to the optical axis.
- **Beam radius**: change the Gaussian before the SLMs. Real-axicon literature shows beam size changes the axial structure strongly and can make a blunt axicon act more lens-like for small beams.
- **Wavefront curvature/collimation error**: use a Gaussian q-parameter or measured quadratic phase. Axicon literature shows incident divergence changes the axial intensity distribution.
- **Ellipticity/astigmatic Gaussian**: independent x/y beam parameters, not a post-hoc Zernike unless that is the measured wavefront.
- **Pointing/decentre jitter**: ensemble simulation from measured distributions, not a single deterministic image.

### 2. SLM and hologram

- **Grey-to-phase LUT / phase-stroke error** for each PLUTO panel.
- **Spatial phase-response nonuniformity and panel flatness**. LCoS literature shows both inherent wavefront distortion and spatially varying phase response can be several tenths of a wavelength or more before calibration.
- **Fringing-field pixel crosstalk**. Sharp phase transitions are blurred and diffraction efficiency changes; panel-specific measured or fitted fringing response is required for quantitative claims.
- **Pixel aperture, fill factor and phase quantisation**. These redistribute power between diffraction orders and impose the pixel sinc envelope.
- **SLM1/SLM2 registration error**: translation, rotation and magnification mismatch of the commanded phase coordinates.
- **Incidence-angle-dependent LCOS response**: published measurements show small incidence changes have little effect, while larger angles (roughly beyond 10 degrees in one study) change the phase response and reduce phase depth. This is calibration-bound for the actual panel geometry.

### 3. 4F and spatial filter

- **Iris/pinhole lateral offset**: translate the physical Fourier-plane aperture.
- **Iris radius/opening**: too large passes unwanted orders/zero order; too small clips the desired spatial spectrum.
- **Fourier-lens aberration**: real transform-lens aberrations distort beam-shaping patterns and must be represented by a measured/prescribed lens OPD, not by silently assuming an ideal FFT.
- **Lens despace/focal error**: explicit physical propagation is required because the iris ceases to sit in the exact Fourier plane.
- **Lens decentre/tilt**: explicit lens coordinates and phase/pupil are required; expected consequences include chief-ray shift, coma/astigmatism and asymmetric filtering.

The current collapsed Fourier-filter function is adequate for ideal-filter and iris-offset/radius studies but is **not** sufficient for 4F lens misalignment/despace. Those cases require an explicit lens-propagation route.

### 4. Axicon

- **Beam-apex lateral decentre**: translate the physical sag and clear aperture relative to the beam. Do not add a linear steering phase.
- **Rigid-body tilt / oblique illumination**: use rotated-plane angular-spectrum propagation and refractive/conical transmission. Published theory/experiment predicts broadened focal lines and astroid-like caustics with increasing obliquity. The previous rotated-thin-element approximation is not report evidence.
- **Rounded/hyperboloidal tip**: explicit sag profile. Real-axicon theory and experiment show strong axial oscillation from interference between the lens-like central contribution and the conical contribution.
- **Flat/blunt apex**: explicit flat central region. Published vortex-Bessel calculations predict B0 is much more sensitive to bluntness than higher-order vortex beams.
- **Base-angle manufacturing error**: use exact refractive geometry for quantitative work, not only the shallow-cone approximation.
- **Refractive index / dispersion**: actual material and n(lambda) are required for lab matching.
- **Finite clear aperture / damaged edge**: multiply the translated/tilted sag transmission by the real aperture at the same optic coordinates. Hard clipping can create axial diffraction ripple.
- **Surface figure / local slope error**: use a measured height map when available. Generic Zernikes are only a sensitivity surrogate.

### 5. Generic upstream wavefront error

Defocus, astigmatism, coma and spherical aberration remain useful as controlled RMS wavefront studies at a declared plane. They answer "what if this wavefront error reaches the axicon?" They do not answer "what does this named mechanical misalignment do?" unless a separate optical model derives the corresponding aberration.

### 6. Objective and sample

Keep these out of the centimetre-scale source/Bessel-zone route:

- objective decentre, tilt and despace;
- relay errors;
- sample/interface tilt;
- refractive-index mismatch and interface aberration;
- focal/sample-plane vector effects.

These belong to the Phase 2C-style vector Debye/interface branch. Mixing them into source-scale axicon propagation is forbidden.

### 7. Laser variability

Pulse-energy fluctuations, wavelength drift, beam pointing jitter and changing spatial mode are ensemble/calibration problems. Wavelength changes both the axicon cone geometry and the SLM/4F diffraction-order geometry. Absolute fluence effects remain calibration-blocked.

## Literature-backed validation targets

The implementation must reproduce these qualitative/quantitative behaviours before report authorisation:

1. Small transverse Gaussian/axicon misalignment: focal line remains approximately parallel to the optical axis while illumination becomes asymmetric (Dufour et al., Frontiers in Optics 2009).
2. Oblique axicon illumination: broadened focal segment and astigmatic/astroid caustics at sufficiently large angle; direct diffraction and experiment agree (Zhao & Li, Applied Optics 1998; Thaning et al., Applied Optics 2003).
3. Real/blunt axicon: strong on-axis oscillation and beam-size dependence (Optics Communications 281, 4240-4244, 2008).
4. Blunt-tip vortex behaviour: B0 more sensitive than higher-order Bessel-vortex beams (Sahin, Turkish Journal of Physics 42, 47-60, 2018).
5. LCOS spatial nonuniformity/flatness: compensation materially improves generated diffraction patterns; panel-specific calibration is necessary (Applied Optics 43, 6400-6406, 2004; Applied Optics 55, 7796-7802, 2016).
6. LCOS fringing field: phase edges blur and diffraction efficiency changes; measured subpixel/convolution models reproduce experiment (Applied Optics 52, 6877-6884, 2013; Applied Optics 54, 5903-5910, 2015).
7. Pixelation/fill factor/quantisation: diffraction-order intensities are controlled by phase-state quantisation plus pixel aperture sinc envelope (JOSA A 18, 205-215, 2001).
8. Real Fourier-transform lens aberration changes beam-shaping output and requires compensation/explicit modelling (Applied Optics 54, 8891-8898, 2015).

## Implementation order

1. Validate **lateral beam/axicon decentre** against the parallel-line benchmark.
2. Add **Gaussian curvature/divergence and ellipticity** at the input plane.
3. Complete **SLM phase-error layer**: LUT scale, static measured-map hook, fringing-model hook and order/zero-order metrics.
4. Build an **explicit 4F propagation route** so lens despace/decentre/tilt and real-lens OPD can be simulated physically.
5. Finish **axicon manufacturing** validation: rounded/flat tip, aperture, base angle, index and surface-map hooks.
6. Implement **rotated-plane oblique axicon solver** and validate against the published caustic behaviour.
7. Only then generate the full report atlas and combined-error Monte-Carlo cases.
8. Objective/sample errors remain a separate downstream validation branch.

No system-error figure is report-ready merely because it looks different or because unit tests pass.
