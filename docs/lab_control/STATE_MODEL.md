# Authoritative experiment state

## Rule

`ExperimentStore` is the only mutable authority. `store.snapshot()` returns a
defensive copy, so a widget, recipe or script cannot silently mutate shared state.
All writes pass through `store.update(mutator, source=..., reason=...)`.

Each real change increments a revision and emits `StateChangeEvent` with:

- event id and UTC timestamp;
- source and human-readable reason;
- exact changed paths;
- before and after values;
- a defensive post-change snapshot.

For example, changing `slm1.phase.vortex_charge` from 0 to 20 makes the phase
preview, analysis family, overlay selection, recipe state and provenance observe
the same revision. The advanced GUI subscribes directly. A compact GUI opened
against the same controller also refreshes from the event.

## Main state sections

| Section | Examples |
|---|---|
| `application` | backend, HEDS SDK version, transfer mode, output and preset roots |
| `system` | experiment label, wavelength, physical axicon state, physical configuration, data kind |
| `slm1`, `slm2` | complete phase configuration, connection, generated hash, last successful cast hash/time, accepted correction id |
| `camera` | provider/status, acquisition, exposure, gain, pixel size, dimensions, z, z reference, last frame/error |
| `geometry` | camera-axis calibration flag/id and coordinate interpretation |
| `session` | session, recipe, run, step, trial, accepted trial and golden reference ids |
| `calibration_statuses` | published view of the separate evidence-rich calibration registry |

## Distinct records

- A **preset** expresses intended phase configuration.
- **Current state** expresses what is configured now.
- **Generated state** has `complete_phase_sha256` for the composed radians array.
- **Cast state** has `last_cast_sha256` only after the provider reports success.
- A **measurement** is a fresh numeric camera observation sealed by the capture service.
- A **calibration** is evidence with validity and dependencies.
- An **accepted correction** is a command that passed a fresh verification capture.
- A **golden reference** is a trusted measurement/state identifier.

These ids and hashes are related, never collapsed.

## Vortex semantics

`effective_vortex_charge()` sums enabled, nonzero contributions on both panels.
The analysis family is `central_beam` when that sum is zero and `vortex_bessel`
otherwise. This is intentionally not restricted to q=20.

## Serialisation

`ExperimentState.to_dict()` produces JSON-safe enums, lists and mappings.
`ExperimentState.from_dict()` restores typed nested objects and validates:

- schema version;
- positive system wavelength;
- locked panel identities/geometries;
- one wavelength across the installed dual-SLM system;
- valid camera scale, range and dimensions.

Use `ExperimentStore.save()` and `ExperimentStore.load()` for state checkpoints.
