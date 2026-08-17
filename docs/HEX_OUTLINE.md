# HEX_OUTLINE - Hollow regular-hexagon outline checkpoint

## Target

The fabrication target is a regular hexagon outline with a hollow core:

- flat-to-centre radius: 7.0 um
- no filled interior
- no six-spoke caustic accepted as a substitute
- no round annulus accepted as a substitute
- side lobes scored separately from core leakage

## Current best candidate

The best lab-corrected phase-only candidate is now the transient-seeded hybrid from
`run_hex_outline_hybrid_checkpoint.py`. It uses the screenshot-like transient
sixfold caustic as the focal-plane phase seed, but still constrains the final
field to the hollow regular-hexagon outline target before passing the phase map
through the quantized/interface-corrected lab model.

| candidate | line FWHM | outline F1 | core peak | side-lobe peak | outline energy | edge uniformity |
|---|---:|---:|---:|---:|---:|---:|
| wide_transient_phase_seed | 2.12 um | 0.825 | 0.00734 | 0.110 | 0.968 | 0.988 |

The direct random-seeded wide outline remains the baseline:

| candidate | line FWHM | outline F1 | core peak | side-lobe peak | outline energy | edge uniformity |
|---|---:|---:|---:|---:|---:|---:|
| wide_random | 2.12 um | 0.820 | 0.0109 | 0.118 | 0.963 | 0.980 |

The narrower balanced outline gives a sharper visual line, but with higher side lobes:

| candidate | line FWHM | outline F1 | core peak | side-lobe peak | outline energy | edge uniformity |
|---|---:|---:|---:|---:|---:|---:|
| balanced_random | 1.53 um | 0.858 | 0.0154 | 0.165 | 0.941 | 0.966 |

The very thin 1.06 um FWHM outline fails: the phase-only system pushes the brightest features into side structure rather than the perimeter.

## Outputs

- Figure: `outputs/figures/hex_outline/13_hollow_hex_outline_checkpoint.png`
- Metrics: `outputs/csv/hex_outline/13_hollow_hex_outline_checkpoint.csv`
- Hybrid figure: `outputs/figures/hex_outline/15_hybrid_transient_seed_lab_gate.png`
- Hybrid metrics: `outputs/csv/hex_outline/15_hybrid_transient_seed_lab_gate.csv`
- Best hologram: `outputs/holograms/hex_outline/hybrid_wide_transient_phase_seed_hollow_hex_outline_phase.png`
- Run manifest: `outputs/json/hex_outline/13_hollow_hex_outline_run_manifest.json`
- Hybrid run manifest: `outputs/json/hex_outline/15_hybrid_transient_seed_run_manifest.json`

## Interpretation

The direct outline route now produces the intended object family: a hollow regular-hexagon perimeter. The transient-seeded hybrid improves the robust 2.12 um FWHM outline by lowering core leakage and improving edge uniformity while staying inside the lab-realistic phase-only path. The 1.53 um FWHM outline is still the sharper candidate if more side-lobe risk is acceptable.
