# Sensorless correction

The sensorless framework optimises **correction commands**, not inferred physical
aberration coefficients. Supported parameters are tip x/y in mrad, astigmatism,
coma, defocus and spherical commands in waves, plus retrieved-map gain.

## Safety sequence

1. Snapshot the accepted baseline state.
2. Create distinct signed nonzero deltas.
3. Shuffle candidates deterministically from a recorded seed.
4. Insert start and end baseline controls.
5. Apply each provisional command and associate its score with a formal capture id.
6. Reject the sweep if control drift exceeds the configured fraction.
7. Recommend the measured candidate with the better score direction.
8. Require a new verification trial and a different capture id.
9. Accept only if fresh improvement exceeds the configured threshold.
10. Otherwise restore the original value, enable-state and accepted-correction id.

No recommendation mutates `accepted_correction_id`. Only
`verify_and_decide()` can do that after a passing fresh capture. Trial scores are
immutable; recapture requires a new trial rather than overwriting evidence.

## Terms and units

- Tip/tilt is a dedicated physical steering layer in mrad and remains separate
  from the fixed order-selection carrier.
- Coma is never a steering substitute.
- Low-order values are commanded waves over the configured reference pupil.
- A retrieved-map value is a dimensionless gain of an existing radians map.
- A successful value is an **accepted correction command** unless an independent
  calibrated response establishes a physical aberration interpretation.

## Resumption

`OptimisationRun` serialises the baseline state, shuffled order, captures, scores,
recommendation, verification and terminal decision. Use `SensorlessOptimiser.save`
and `.load` to checkpoint. A resumed provisional run should either continue its
prescribed trial or explicitly roll back before other optical changes.
