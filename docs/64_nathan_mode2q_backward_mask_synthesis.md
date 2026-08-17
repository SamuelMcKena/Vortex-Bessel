# Nathan MODE 2Q - Backward / Adjoint Source-Scale Mask Synthesis

**Status:** MODE 2Q source-scale inverse mask design only. The validated V0 **complex vector
field** at `z_ref = 60 mm` is the target; the masks are whatever the inverse optical system says
they must be, then everything is verified by full forward propagation. No inherited
objective/sample microfabrication geometry, no MODE 1C/M1E sample-plane constraints, no panel
realism, and six bright lobes are never called a hexagon unless the strict C3-vs-C6 classifier
gate passes.

## Correction Of Aim

The displayed SLM/HWP masks are not the target - the propagated beam is. MODE 2Q therefore
implements:

```text
V0 complex target at z = 60 mm
  -> inverse angular-spectrum propagation   E~(0) = E~(z) exp(-i kz z), propagating modes only
  -> inverse thin-axicon Jones              reciprocal p/s diagonal, conjugate conical phase
  -> inverse (adjoint) QWP                  J^-1 = J^dagger, unitarity checked
  -> required H/V complex fields            phi/amp extraction
  -> phase-only / constrained projection    EH = A/sqrt(2) exp(i angle(EH_required))
  -> full forward verification              bench operator to z = 60 mm and the z-stack
```

The 4F iris discards spatial frequencies and has **no exact inverse**; it is handled as an adjoint
projection with the unreachable spectral energy reported explicitly.

## Target Availability

The V0 propagation machinery retains complex vector fields (`return_fields=True`), so the **full
complex vector target was available** - no intensity-only phase-retrieval fallback was needed. The
V0 target plane passes the strict hexagon gate (`visual_hexagonal_field`, dark core `0.0009`,
`c120 - c60 = -0.042`).

## Backward-Pass Diagnostics (confirmation run, grid 1024)

The vital consistency check - inverting the V0 target back through the propagation and axicon must
recover Nathan's known raw input - passes:

- backward-recovered pre-axicon field vs raw Nathan input: complex vector overlap
  `0.99999997`, Stokes RMS `5.8e-4`, alpha-angle RMS `1.1e-4 rad`, power ratio `0.9997`;
- evanescent clipped energy in backpropagation: `0.0`;
- inverse-axicon Jones condition number: `1.000128` (no near-singular transmission);
- residual `Ez` power in the recovered pre-axicon field: `1.3e-4` (paraxial, as expected).

Inverse QWP (adjoint of the M2P-selected `-pi/4` QWP) then yields the required H/V channels:

- required amplitudes match the phase-only supply `A(r)/sqrt(2)` to RMS `2.0e-4` on both
  channels - i.e. the inverse-designed masks are **phase-only realizable**, and the recovered
  phase masks reproduce the M2P convention (`H: +alpha`, `V: -alpha + pi/2`) as the inverse
  solution rather than as an assumption;
- 4F adjoint bookkeeping: `5.06%` of the required-field spectral energy lies outside the
  2.5 lp/mm first-order passband (`exactly_realizable_through_this_4f = False`); this is the
  sector-discontinuity tail loss, and it matches the 5.1% first-order efficiency loss seen in the
  forward 4F chain.

## Forward Verification (confirmation run, grid 1024)

| candidate | pre-axicon overlap (required / raw) | z60 corr | complex overlap at z60 | strict class | pass |
|---|---|---:|---:|---|---|
| phase_only_direct | 1.000000 / 1.000000 | 1.000000 | 1.000000 | visual_hexagonal_field | pass |
| phase_only_4f | 0.949289 / 0.949279 | 0.993609 | 0.949284 | visual_hexagonal_field | pass |

Both candidates keep the dark core (`~0.001`) and C6 dominance (`c120 - c60 ~ -0.04...-0.05`), and
the true-hexagon gate passes on the majority of sampled z-planes (0.89 direct, 0.67 through 4F;
the early forming planes near z ~ 0 are not yet hexagonal, honestly reported).

**Optimisation:** the bounded low-dimensional optimiser (six per-sector pistons, sector rotation,
sector duty scale, global V piston; Nelder-Mead) is implemented but was **not run**: in `auto` mode
it only triggers when the analytic backward initialisation fails, and here the phase-only
projection of the inverse solution already passes. Pixel-level optimisation remains out of scope
until the low-dimensional model demonstrably fails.

## Outputs

Generated in `outputs/figures/digital_twin/nathan_mode2q_backward_mask_synthesis/`:

- `mode2q_v0_target_z60.png`
- `mode2q_backward_required_pre_axicon.png`
- `mode2q_backward_vs_raw_nathan_input.png`
- `mode2q_required_hv_amplitude_phase.png`
- `mode2q_phase_only_projected_masks.png`
- `mode2q_forward_phase_only_z60.png`
- `mode2q_zstack_summary.png`
- `mode2q_backward_diagnostics.json`
- `mode2q_mask_candidates.csv/json`
- `mode2q_outcome_report.json`
- `simulation_scope_manifest.json`

(`mode2q_forward_optimised_z60.png` is only produced when the optimiser runs.)

## Outcome

**M2Q-A.** Backward inversion recovers Nathan's raw pre-axicon field from the V0 complex target
(inverse/forward operators are consistent), the required H/V fields are phase-only realizable, and
the physically constrained SLM/QWP masks reproduce the V0 hexagonal Bessel output after full
forward propagation - both directly and through the carrier/4F chain (0.9936 correlation, strict
hexagon gate passed). Source-scale IRL replication via inverse-designed masks is plausible.

The forward-displayed masks (MODE 2N) and the inverse-designed masks (MODE 2Q) coincide for this
ideal bench because the component chain is unitary up to the 4F clipping; the inverse framework's
value is that it now *proves* that coincidence and provides the machinery (adjoint 4F bookkeeping,
amplitude-mismatch reporting, low-dimensional precompensation, strict gate) for the non-ideal
cases where they will diverge - SLM quantisation/fill-factor, waveplate errors, misaligned iris.

## Next Action

If this branch continues, apply the MODE 2Q inverse/precompensation machinery to a *degraded*
bench (pixelated quantised SLM, retardance/axis errors, iris misalignment) where the analytic
masks stop being optimal and the low-dimensional optimiser earns its keep. This remains
source-scale only; it makes no microfabrication/sample-plane claim.
