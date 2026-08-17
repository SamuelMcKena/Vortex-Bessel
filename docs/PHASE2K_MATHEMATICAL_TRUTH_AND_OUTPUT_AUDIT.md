# Phase 2K — Mathematical Truth and Complete Output Audit

## Status

**Active audit branch:** `phase2k-mathematical-physics-output-audit`

**Policy:** no historical PNG, PDF, CSV, JSON, notebook output, presentation asset, or GitHub Actions artifact is accepted as scientific evidence merely because it exists, looks plausible, or reproduces a historical regression baseline. Every output must trace back to a producer whose governing equations, numerical implementation, sampling and physical inputs have passed the truth hierarchy below.

The current checked-in `outputs/` tree is therefore quarantined by default. Scientific reuse is unlocked only output-family by output-family after regeneration from a validated producer.

## Truth hierarchy

An output can advance only through these gates, in order.

1. **Analytic identity / equation gate**
   - Compare the code with the governing optical equation, including definitions, sign conventions, units and coordinate planes.
   - Do not use a regression baseline as the reference when the regression baseline was produced by the same code.

2. **Independent numerical reference gate**
   - Where an analytic closed form is unavailable, compare against a separately implemented solver or an independent formulation.
   - Example: the finite Bessel–Gauss closed form is checked against an independently written FFT Fresnel transfer-function propagator.

3. **Sampling / convergence gate**
   - Demonstrate convergence under spatial-grid, propagation-step, window-size, pupil/quadrature and interpolation changes appropriate to the observable being claimed.
   - Plot smoothness is not convergence.

4. **Optical-component / coordinate gate**
   - Confirm that SLM, 4F/Fourier-plane, objective, interface and camera coordinates are related by the correct wavelength-bearing physical transforms.
   - Distinguish an intended optical loss/filter from numerical loss.

5. **Hardware-provenance gate**
   - A mathematically valid calculation is still not a calibrated bench prediction when SLM identity/LUT, beam radius, focal lengths, iris dimensions, objective fill, relay mapping, axicon geometry or camera scale are assumed.

6. **Experimental-closure gate**
   - A numerical prediction becomes experimentally validated only after comparison with measured data under a traceable geometry and uncertainty model.
   - Model-to-model inverse recovery is not experimental validation.

## Primary mathematical references

The audit is anchored to primary or authoritative optics literature rather than to the repository's own historical plots.

- J. Durnin, “Exact solutions for nondiffracting beams. I. The scalar theory,” *JOSA A* **4**, 651–654 (1987), DOI: `10.1364/JOSAA.4.000651`.
- F. Gori, G. Guattari and C. Padovani, “Bessel-Gauss beams,” *Optics Communications* **64**, 491–495 (1987), DOI: `10.1016/0030-4018(87)90276-8`.
- J. H. McLeod, “The axicon: a new type of optical element,” *JOSA* **44**, 592–597 (1954), DOI: `10.1364/JOSA.44.000592`.
- K. Matsushima and T. Shimobaba, “Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields,” *Optics Express* **17**, 19662–19673 (2009), DOI: `10.1364/OE.17.019662`.
- Q. Zhan, “Cylindrical vector beams: from mathematical concepts to applications,” *Advances in Optics and Photonics* **1**, 1–57 (2009), DOI: `10.1364/AOP.1.000001`.
- J. Arlt and K. Dholakia, “Generation of high-order Bessel beams by use of an axicon,” *Optics Communications* **177**, 297–301 (2000), DOI: `10.1016/S0030-4018(00)00572-1`.

Further component-specific references will be added as each producer family is audited (Debye/Richards–Wolf focusing, Fresnel/interface propagation, phase-only holography, axicon-surface diffraction, inverse phase retrieval and uncertainty/identifiability).

## Corrections already made in Phase 2K

### 1. Finite Bessel–Gauss propagation

The old `bessel_gauss_field` implementation froze the transverse Bessel/Gaussian profile with `z` and only multiplied by an axial phase. That is not propagation of a finite Bessel–Gauss beam.

The Phase 2K implementation uses the paraxial finite-energy Bessel–Gauss solution with

```text
q(z) = 1 + i z/z_R,
z_R  = k w0^2 / 2,

U_l(r,phi,z) = A/q * J_|l|(k_r r/q)
               * exp[-r^2/(w0^2 q)]
               * exp[-i k_r^2 z/(2 k q)]
               * exp(i l phi),
```

with the optional fast carrier `exp(i k z)` kept separate. A nonzero-z call now requires wavelength and refractive index. The implementation is independently tested against FFT Fresnel propagation of the same waist field.

### 2. Infinite-Bessel versus finite-BG ring radius

The root of `J'_l` gives the bright-ring location for the infinite/pure Bessel reference. It is not generally the exact maximum of

```text
I(r) = |J_l(k_r r)|^2 exp(-2 r^2/w0^2)
```

when Gaussian apodisation is appreciable. Phase 2K therefore retains the `J'_l` value as a named asymptotic reference and separately solves for the finite-BG intensity maximum.

A deliberately strong-apodisation truth check (`l=3`, `k_r=1.0e6 m^-1`, `w0=4 um`) shifts the finite-BG peak inward by about 21% relative to the pure-Bessel `J'_3` reference. This test is not a bench prediction; it demonstrates why the two definitions must not be conflated.

### 3. Exact `J0` first zero in inverse design

The historical constant `2.405` is only a rounded approximation to

```text
j_0,1 = 2.40482555769577...
```

Phase 2K uses the exact numerical Bessel root supplied by SciPy and propagates that definition consistently through target diameter ↔ `k_r` conversions. Historical regression tests that demanded the rounded value are being replaced by mathematical-reference tests rather than forcing the corrected code back to the old number.

### 4. Refractive-axicon reference

The thin phase-screen relation is now explicitly separated from exact geometrical refraction. For conical-surface normal tilt `gamma`, exit index `n_a` and external index `n_e`, the exact ray deflection satisfies

```text
n_a sin(gamma) = n_e sin(gamma + theta),

theta = asin[(n_a/n_e) sin(gamma)] - gamma,

k_r = k0 n_e sin(theta).
```

The thin optical-path phase-screen approximation uses the **vacuum** wavenumber

```text
k_r ~= k0 (n_a - n_e) tan(gamma),
```

not `k_medium (n_a-n_e) tan(gamma)`, which would double-count the external refractive index outside air.

The axicon angle convention must be explicit. Manufacturer **apex angle** and repository **base/surface-normal tilt** are not interchangeable; an unconfirmed physical optic angle is therefore a hardware-provenance blocker, not a parameter to guess.

## Current numerical evidence state

The complete current checked-in output tree is inventoried automatically. The first Phase 2K inventory contains **587 files** and quarantines every pre-existing output until its producer is traced and revalidated.

The existing repository itself already records major reasons not to treat the historical output tree as a single trusted evidence set. Among the tabular outputs, large groups contain explicit invalidity flags, excessive propagation-power drift, failed acceptance gates, or pre-repair coordinate/model semantics. In particular, old Phase 1/legacy propagation files with more than 5% plane-power drift remain diagnostics only.

Phase 2D calibration governance also states that there is currently no experimentally validated prediction and that absolute dimensions/fluence remain locked pending real calibration. Phase 2E finite-propagation evidence contains a separate z-step convergence failure, so those finite-propagation report replacements remain blocked until regenerated at a converged axial sampling.

## Hardware truth gate

The Phase 2K audit presently blocks an absolute fixed-bench quantitative claim because the following critical values are not all measured/verified in the repository manifest. The manifest's old `fixed_bench_prediction_ready=true` label has therefore been corrected: `nominal_fixed_parameter_simulation_ready=true`, while `fixed_bench_prediction_ready=false` until the calibration gate is closed.

- physical axicon angle convention/value and clear aperture,
- beam radius incident on the SLM,
- exact SLM model/panel, phase stroke and calibrated LUT,
- 4F focal length and Fourier-iris geometry,
- objective NA/effective focal length and pupil fill,
- relay effective focal length/magnification,
- camera pixel-to-object scale/rotation.

This does **not** prevent mathematical or controlled numerical studies. It prevents assumed values from being described as physically measured bench truth.

## Output-family disposition rule

Each family will receive one of these terminal dispositions after producer audit:

- `VALIDATED_ANALYTIC_CONTROL`
- `VALIDATED_NUMERICAL_MODEL_CALIBRATION_BLOCKED`
- `VALIDATED_FIXED_BENCH_MODEL`
- `EXPERIMENTALLY_VALIDATED`
- `DIAGNOSTIC_ONLY`
- `SUPERSEDED_REGENERATE`
- `REJECTED_PHYSICS_OR_NUMERICS`

Presentation derivatives never outrank the source arrays/data that generated them.

## Next producer audits

The remaining audit is intentionally producer-first rather than image-first:

1. finish exact-root cleanup in every active structured-beam producer;
2. validate scalar BL-ASM/SAS propagation against independent angular-spectrum references and convergence sweeps;
3. audit the 4F carrier/order geometry and finite iris with wavelength-bearing coordinates;
4. audit refractive-axicon exact/thin model regimes and tip/tilt/decentre error models;
5. audit objective focusing against independent Debye/Richards–Wolf controls;
6. audit interface Fresnel/vector propagation using lossless energy and transversality constraints;
7. audit vortex winding/topology metrics independently of intensity morphology;
8. audit polygonal/hexagonal/discrete families against their defining spectra, not only visual resemblance;
9. audit inverse-recovery identifiability with synthetic truth, noise/registration perturbations and then measured data;
10. regenerate only the families whose producers are green, then visually inspect and classify every regenerated output before any thesis/presentation figure selection.

A thesis figure shortlist will be produced **after** these gates, not before them.
