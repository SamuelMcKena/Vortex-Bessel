# 02 Pulse Energy and Optical Transmission

## Purpose

Measure incident pulse energy and each governed optical throughput once, preserving the Phase 2A
rule that selected first-order efficiency is represented exactly once.

## Required Equipment

A wavelength-appropriate calibrated energy or power meter, repetition-rate record, approved sampling
optics/attenuation, Stage 9A logs, and authorised laboratory safety controls.

## Beam-Safe Setup

Use the formal laser safety procedure, interlocks, enclosures, rated components, and approved meter
limits. Only authorised personnel may change beam-path hardware. This document is not an alignment
procedure.

## Measurement Sequence

Record incident energy and paired input/output energy at SLM1, SLM2, common 4F, objective, and
interface planes using a documented plane contract. Take at least ten readings per plane, record dark
offsets and meter range, and retain timestamps and calibration certificates. Do not include selected
first-order filtering inside the measured `four_f` factor unless the bundle explicitly replaces the
accepted factor and the ledger contract is revised under a separate audit.

## Equation And Units

```text
T_stage = E_out / E_in
u(T)^2 = (u(E_out)/E_in)^2 + (E_out*u(E_in)/E_in^2)^2
E_j = E_0 product(T_i)
```

Energy is in joules; transmission is dimensionless and must satisfy `0 < T <= 1`.

## Repeats And Uncertainty

Use at least ten independent readings. Combine meter calibration, repeatability, dark correction,
range resolution, and timing/repetition-rate uncertainty where applicable.

## Acceptance Criteria

No meter saturation; stable readings with documented outlier handling; closure of paired measurements
within their combined uncertainty; exactly one selected-order factor in the generated ledger.

## Output Format

Populate `calibration/templates/energy_transmission_template.csv`, then map accepted values to
`laser.pulse_energy_J` and `transmissions.*` in the JSON bundle.

## Code Mapping And Claims

These fields feed the Phase 2D staged energy and peak-fluence propagation. Absolute fluence also
requires calibrated physical scale; energy data alone does not unlock it.
