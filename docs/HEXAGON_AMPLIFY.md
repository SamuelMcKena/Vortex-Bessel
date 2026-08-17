# HEXAGON_AMPLIFY — Air sixfold-harmonic study

## Summary

Vector-axicon route does NOT produce an accepted hexagon: best order6 = 0.0997, but hexagon acceptance is False. A dedicated polarization/polygon-shaping optic is required for a visually hexagonal intensity pattern.

## Parameter sweep results

Best order6/order0 achieved: **0.0997**
Accepted hexagon found: **no**

| sweep knob | best value | best order6 |
|---|---|---|
| n_axicon | 1.70 | 0.0734 |
| angle_deg | 20.00 | 0.0997 |
| segment_count | 2.00 | 0.0744 |

## Design recipe (strongest single knob per parameter)

- n_axicon: **1.70**
- apex angle: **20°**
- segment count: **4** total sectors

## Zone lengths

- Air sixfold-harmonic zone: **114 µm** (z-range where order6 ≥ 50 % of peak)
- In-medium sixfold-harmonic survival: **220 µm** in Cr:ZnSe

These zone lengths are harmonic diagnostics only. They do not mean the intensity image passes the visual hexagon-acceptance test.

## Verdict

The vector-axicon Fresnel route does **not** produce a visually accepted hexagon at this wavelength: best order6 = 0.0997, and no sweep point passes the full intensity acceptance test. The Fresnel s/p amplitude contrast at 1029 nm is a few percent even for high-index axicons. A dedicated polarization-shaping optic (polygon beam shaper, spiral phase plate array, or polarization-converting element) is required for strong hexagonal contrast.

The vector-axicon route should be considered a weak/nonlocal sixfold-harmonic bias, not a primary hexagon generator.
