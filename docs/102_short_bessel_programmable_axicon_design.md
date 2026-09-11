# Short-Bessel programmable-axicon design

**Status:** inverse-design / feasibility study only.  This document does not claim a calibrated laboratory prediction for the current 4F + physical-axicon bench.

## Design question

Can the existing PHAROS + SLM architecture be steered toward a deliberately short Bessel/Bessel-like interaction region, for example

- **J0 central-core FWHM:** approximately **5 um**;
- **finite Bessel-Gauss reference length:** approximately **500 um**;

while retaining the downstream physical axicon and using an upstream SLM radial phase as a signed tuning term?

The new calculator in `vbb_study.short_bessel` treats this as two mostly separable first-order design knobs:

1. radial wavevector `k_r` controls the transverse J0 scale;
2. illuminated Gaussian radius controls the finite overlap/reference length.

For a J0 intensity profile

```text
I(r) = J0(k_r r)^2
```

let `x_half` satisfy `J0(x_half)^2 = 1/2`.  Then

```text
D_FWHM = 2 x_half / k_r
x_half = 1.126364239377483...
```

The repository historically uses the equivalent ell=0 **first-zero diameter** as its `target_core_diameter_m` convention, so the calculator also reports

```text
D_first_zero = 2 j_0,1 / k_r
j_0,1 = 2.404825557695772...
```

This avoids silently confusing a 5 um FWHM with a 5 um first-zero diameter.

## 5 um x 500 um baseline result

Using the current repository defaults where applicable:

- wavelength `lambda0 = 1029 nm`;
- fused silica planning index `n = 1.45`;
- objective `NA = 0.45`;
- objective effective focal length `4.0 mm`;
- effective relay focal length `495.597636 mm`;
- fixed-optics transverse demagnification `M = 0.00807106`;
- SLM pixel pitch `8 um`;
- SLM active height `1080 px`;
- conservative safe radial fraction `0.90` of the SLM half-height;
- current pre-objective Gaussian 1/e field radius `2.0 mm`.

The target produces approximately:

| quantity | value |
|---|---:|
| target J0 FWHM | 5.000 um |
| equivalent J0 first-zero diameter | 10.675 um |
| target sample `k_r` | 450,546 m^-1 |
| cone angle in `n=1.45` sample | 2.917 deg |
| air-equivalent cone angle | 4.231 deg |
| required objective NA | 0.0738 |
| required sample 1/e field radius for 500 um reference length | 25.44 um |
| required pre-objective 1/e field radius | 3.152 mm |
| conservative SLM-safe radius | 3.888 mm |
| aperture margin | about +0.736 mm |
| predicted reference length with current 2.0 mm pre-radius | about 317 um |
| required beam-radius scale relative to 2.0 mm | about 1.576x |
| target pre-plane `k_r` under current fixed mapping | 3,636 m^-1 |
| corresponding 2pi radial phase-wrap period | about 1.728 mm |
| phase-wrap period at 8 um pixels | about 216 pixels |

This is a useful result: the **5 um x 500 um target is not obviously excluded by the current scalar NA, SLM sampling or SLM-height budgets**.  It remains a feasibility result, not proof that the current bench will produce the target.

## Why the length should not be tuned with cone angle alone

For the geometrical finite Bessel-Gauss overlap reference used by the repository,

```text
L_ref = w0 * k / k_r
```

so increasing `k_r` simultaneously shrinks the J0 core and shortens the Bessel region.  That couples two quantities we would rather control separately.

For a fixed requested 5 um FWHM, the cleaner first-order strategy is therefore:

```text
k_r                 -> transverse spot scale
illuminated radius  -> finite Bessel length
```

The target calculator consequently solves `k_r` from FWHM first, then solves the required beam radius from the requested length.

## Combining the SLM radial phase with the physical axicon

The programmable phase convention is

```text
phi_slm(r) = -k_r,slm r
```

In a same-sign/conjugate-plane thin-phase approximation,

```text
k_r,total = k_r,physical + k_r,slm
```

which immediately gives

```text
k_r,slm = k_r,target - k_r,physical
```

The most robust way to obtain `k_r,physical` for the real bench is **not** to trust an unverified nominal axicon angle.  Measure the physical-axicon-only central J0 core at the relevant sample plane and infer

```text
k_r,physical = 2 x_half / D_FWHM,physical
```

The new tool accepts that measurement directly:

```bash
python tools/run_short_bessel_design.py \
  --fwhm-um 5 \
  --length-um 500 \
  --physical-fwhm-um <MEASURED_PHYSICAL_ONLY_FWHM_UM>
```

If the physical axicon alone makes a core **smaller than 5 um**, its `k_r` is too large and the generated SLM radial phase has the opposite sign to partially cancel it.  If the physical-only core is **larger than 5 um**, the SLM adds radial wavevector in the same sign.

When `--physical-fwhm-um` is omitted, the generated phase mask is explicitly labelled `target_only_no_physical_calibration`.  It is not presented as the correct command for the combined system.

## Outputs

Running

```bash
python tools/run_short_bessel_design.py --fwhm-um 5 --length-um 500
```

creates

```text
outputs/short_bessel_design/
  short_bessel_design_report.json
  short_bessel_design_sweep.csv
  short_bessel_design_summary.png
  slm_axicon_phase_target_only_no_physical_calibration.png
```

With a physical-only FWHM measurement supplied, the mask becomes

```text
slm_axicon_phase_physical_measured_residual.png
```

The summary figure contains:

1. a 3--8 um FWHM / 200--1000 um length feasibility map showing required upstream beam radius;
2. the exact target `J0^2` transverse profile with FWHM and first-zero markers;
3. the wrapped SLM radial phase command;
4. an audit panel with NA, aperture, phase-sampling and beam-radius requirements.

## Suggested lab experiment

A clean first experiment is deliberately simple and does not require claiming that the full 4F digital twin is calibrated.

1. **Characterise the physical-axicon-only beam.**  With no extra SLM axicon term, measure the central-core FWHM and the useful axial region from a z stack.
2. **Record the upstream 1/e field radius.**  The length calculation is only meaningful if the beam radius entering the mapped focusing train is known.
3. **Enter the measured physical-only FWHM into the calculator.**  This gives a signed residual SLM radial phase.
4. **Apply the residual radial phase before the 4F system**, together with the existing carrier/correction terms, and continue selecting the desired diffraction order.
5. **Sweep the residual around the calculated value** (for example 0.7x, 0.85x, 1.0x, 1.15x, 1.3x) rather than trusting one model point.
6. **At each command, acquire a z stack** and measure core FWHM, ring radius, peak-vs-z and useful Bessel-region length.
7. **Adjust beam radius independently** toward the predicted approximately 3.15 mm pre-objective 1/e field radius if the transverse target is correct but the axial region remains too short.

This experiment tests the central hypothesis directly: an upstream signed radial phase can add to or subtract from the effective conical wavefront delivered by the downstream physical axicon, while beam radius is used as the main first-order control of finite propagation length.

## What this still does not prove

The current repository explicitly does not have a calibrated component-owned model of the full physical 4F order-selection train.  Before treating the result as an absolute bench prediction, the following still need to be bound by measurement or a validated optical model:

- exact SLM pixel pitch / active area / phase response for the installed device;
- SLM-to-4F geometry and 4F magnification;
- +1 order filter centre, radius and clipping;
- physical axicon base/apex-angle convention and refractive index at 1029/1030 nm;
- physical axicon position and clear aperture;
- objective/relay mapping into the actual sample plane;
- interface refraction/aberration for the chosen write depth;
- nonlinear propagation and material response at processing intensity.

The short-Bessel calculator intentionally stops before those claims.  Its purpose is to turn the 5 um x 500 um idea into a **dimensionally explicit, experimentally testable design target**.
