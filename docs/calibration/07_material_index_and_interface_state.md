# 07 Material Index and Interface State

## Purpose

Bind the vector Fresnel model to the actual sample refractive index, wavelength, coating state and
surface orientation instead of the current placeholder interface.

## Required Equipment

Sample identity/certificate, coating records, approved index metrology or authoritative dispersion
data, surface-orientation record, inspection tools, and authorised laboratory safety controls.

## Beam-Safe Setup

Prefer documentary and non-laser inspection. Any optical measurement must follow the formal safety
procedure, approved risk assessment and engineering controls. This document is not an alignment or
high-power exposure instruction.

## Measurement Sequence

Record sample serial, composition, temperature, wavelength, index source, coating stack/state and
surface normal convention. If index is measured, acquire at least five independent readings or use
the metrology procedure's required repeat count. Preserve certificates and raw evidence hashes.

## Equation And Units

For an uncoated lossless reference, the normal-incidence check is

```text
R = ((n1 - n2)/(n1 + n2))^2
T = 1 - R
```

The canonical component calculation uses spectral s/p Fresnel coefficients. Index and transmission
are dimensionless; wavelength is metres; orientation is defined against the project coordinate frame.

## Repeats And Uncertainty

Report standard uncertainty from certificate, fit or repeated metrology and account for wavelength
and temperature. Do not assign an uncertainty where the source supplies none.

## Acceptance Criteria

Sample identity, wavelength applicability, coating state and surface orientation are explicit;
refractive index is physical (`n >= 1`) with traceable source; no unknown coating is treated as
uncoated.

## Output Format

Populate `calibration/templates/material_interface_template.csv`; map accepted values to
`material.refractive_index`, `material.coating_state`, and `material.surface_orientation_verified`.

## Code Mapping And Claims

These fields unlock calibrated interface power and component-resolved Fresnel predictions. They do
not unlock absolute fluence without energy and scale calibration.
