# Actual Lab Vector Case 1

This note records the current vector-beam hardware boundary used by the
publication study. It is documentation only; it does not change the vector
model.

## Purpose

Vector and polarization-structured Bessel beams are first-class branches of the
structured-beam atlas. The repeated comparison is the same as the scalar branch:
an ideal target field is compared with a lab-realistic implementation route and
its known limitations.

## Case-1 Interpretation

The Case-1 lab vector route is treated as a modeled hardware route only where
the available optical train can credibly generate the required state of
polarization. When a radial, azimuthal, or otherwise spatially varying
polarization element is not part of the modeled hardware, the study should not
label that target as already lab-realizable.

Recommended taxonomy labels for current Case-1 vector runs:

- `beam_family`: `vector_bessel`
- `model_level`: `vector_jones` or `lab_encoded`
- `generation_method`: `segmented_vector_element` when a spatial polarization
  element is modeled; otherwise inherited from the scalar route
- `hardware_status`: `case1_existing_hardware` only for routes represented by
  the current lab model; otherwise `proposed_hardware`
- `material_model_status`: `not_applicable` unless a materials proxy is layered
  on top

## Reporting Rule

Reports should separate three statements:

- the ideal vector target exists mathematically;
- the scalar/holographic/physical route can target a related intensity field;
- a lab vector implementation is claimed only when the required polarization
  hardware is explicitly modeled.

This keeps vector beams in the main atlas while preventing the hardware status
from being overstated.
