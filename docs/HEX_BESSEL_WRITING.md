# HEX_BESSEL_WRITING - True hollow hexagonal writing beam status

## Requirement

A true hexagonal Bessel writing beam is not just a hexagon at one focal plane.
It must keep the hollow hexagon outline over a useful propagation depth in the
sample medium.

The current acceptance gate requires each z plane to keep:

- outline F1 >= 0.65
- core peak <= 0.08
- side-lobe peak <= 0.25
- one connected outline component

## Current result

`run_hex_bessel_like_checkpoint.py` tests the lab-realistic phase-only SLM path
over z in Cr:ZnSe.

| candidate | line FWHM | accepted depth | accepted planes | mean outline F1 | max core | max side lobe |
|---|---:|---:|---:|---:|---:|---:|
| wide_single_plane_seeded | 2.12 um | 5.0 um | 2 | 0.535 | 0.997 | 1.000 |
| wide_multiplane_seeded | 2.12 um | 0.0 um | 0 | 0.000 | 0.062 | 1.000 |
| balanced_multiplane_seeded | 1.53 um | 0.0 um | 0 | 0.000 | 0.069 | 1.000 |

## Zernike phase-mask test

`run_zernike_hex_bessel_checkpoint.py` tests a lab-realistic holographic
axicon/vortex Bessel cone with a sixfold Zernike phase perturbation. The sweep
uses ell = 4, 6, 8; Zernike terms (6,6), (8,6), (10,6); and amplitudes from
-3 to +3 rad.

Every swept Zernike case has accepted depth 0.0 um under the hollow-hex
z-stack gate.

| selector | candidate | accepted depth | mean outline F1 | focus outline F1 | focus core | focus side lobe |
|---|---|---:|---:|---:|---:|---:|
| best z-stack score | ell4_Z6_6_amp+0.00 | 0.0 um | 0.366 | 0.000 | 1.000 | 0.168 |
| best focal outline F1 | ell8_Z6_6_amp+0.50 | 0.0 um | 0.095 | 0.763 | 0.192 | 0.981 |
| best focal side control | ell8_Z8_6_amp+1.00 | 0.0 um | 0.094 | 0.746 | 0.187 | 0.720 |

Interpretation: a sixfold Zernike term can bias the annular Bessel beam, but
in this lab-realistic phase-only encoding it does not create a stable hollow
hexagonal Bessel writing beam. At best it gives a round ring with weak sixfold
brightness modulation at one plane; the side lobes remain too high and the
shape does not survive over z.

## Interpretation

The current SLM focal-plane hologram is a hollow hexagon outline, but it is not
a true hexagonal Bessel writing beam. It survives only about 5 um by the current
depth gate, then fills the core and breaks into lobes.

The repo's conical/angular-spectrum polygonal route is closer to true Bessel
physics: the ideal continuous-ring case remains stable over z. However, the
current lab-realistic phase-only encoding rounds it toward an annulus and does
not yet preserve the desired hollow hexagonal outline.

## Outputs

- Figure: `outputs/figures/hex_outline/16_hex_bessel_like_multiplane_survival.png`
- Summary CSV: `outputs/csv/hex_outline/16_hex_bessel_like_summary.csv`
- Z-profile CSV: `outputs/csv/hex_outline/16_hex_bessel_like_z_profile.csv`
- Run manifest: `outputs/json/hex_outline/16_hex_bessel_like_run_manifest.json`
- Zernike figure: `outputs/figures/hex_outline/17_zernike_hex_bessel_sweep.png`
- Zernike CSV: `outputs/csv/hex_outline/17_zernike_hex_bessel_sweep.csv`
- Zernike run manifest: `outputs/json/hex_outline/17_zernike_hex_bessel_run_manifest.json`

## Next route

The next design should be conical from the start:

1. constrain the spectrum to a Bessel cone, not a focal-plane CGH;
2. optimize the angular spectrum for a hollow hexagonal outline;
3. use Zernike/similar low-order terms only as correction or initialization
   terms inside that cone optimization, not as the whole beam design;
4. encode that cone with lab-realistic complex-amplitude control, two-SLM
   encoding, or a fabricated/vector polygonal optic;
5. only accept candidates that pass the z-stack gate above.

In plain terms: the correct physical family is a conical angular-spectrum
hexagonal Bessel beam. A single phase-only focal hologram can make the picture,
but not the writing volume.
