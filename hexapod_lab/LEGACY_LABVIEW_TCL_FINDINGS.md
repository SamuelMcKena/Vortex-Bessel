# Legacy LabVIEW + TCL findings

Source set reviewed: the uploaded **TCL Scripts** folder and **Labview VIS** folder.

This document records what the old lab code actually supports, what has now been
implemented in the standalone Python controller, and what still needs physical
confirmation before real processing.

## High-confidence findings from the TCL scripts

### Pockels / process-beam digital output

Three complete TCL scripts use the same pattern:

```tcl
GPIODigitalSet $socketID GPIO4.DO 1 1
# writing move
GPIODigitalSet $socketID GPIO4.DO 1 0
```

The scripts label the first state as the writing/laser-on interval and the second
as laser-off.

That is strong evidence that the historical processing scripts used:

- digital channel: `GPIO4.DO`
- mask: `1`
- writing state: value `1`
- non-writing state: value `0`

**This is not yet treated as electrically verified PHAROS LX13 wiring.** It tells
us what the old HXP software commanded, not the physical connector pinout or
whether the present cable/wiring is unchanged.

### Writing motion

The writing scripts consistently use:

```tcl
HexapodMoveIncrementalControlWithTargetVelocity
    $socketID HEXAPOD Work Line dX dY dZ velocity
```

Examples include:

```tcl
... HEXAPOD Work Line -7 0 0 1
... HEXAPOD Work Line -7 0 0 $p
... HEXAPOD Work Line 7 0.02 0 0
```

So the legacy processing workflow is not merely a normal incremental move followed
by a host-side delay. It uses the HXP's explicit **Line + target velocity** motion
command in the **Work** coordinate system.

The Python controller now implements that exact HXP command and exposes it as a
script-builder block.

### Typical legacy writing pattern

The common pattern is:

1. Set analogue power/attenuation command when required.
2. Set digital output to writing state.
3. Execute a target-velocity line move, commonly `dX=-7 mm`.
4. Set digital output to non-writing state.
5. Return `dX=+7 mm` while stepping `Y` by `0.02 mm`.

The 2D movement map and sample-local process trace are well suited to displaying
this exact workflow.

## Analogue power / attenuation evidence

Two TCL scripts use:

```tcl
GPIOAnalogSet $socketID GPIO2.DAC1 <value>
```

with values from **1 to 5**.

Those files are named:

- `powerino.tcl`
- `Power_Speed_Sweep.tcl`

This is strong evidence that an HXP analogue output historically controlled some
quantity treated by those scripts as **power**.

However, `Pulse_Sweep.tcl` instead contains:

```tcl
GPIOAnalogSet $socketID GPIO4.AO 5
```

Therefore the standalone controller must **not** yet assume that the present
attenuator is always `GPIO2.DAC1`, nor convert 1–5 directly into transmission
percent without a calibration.

The low-level Python HXP client now supports `GPIOAnalogSet`, but the real
attenuator provider remains unbound until the active channel and calibration are
confirmed.

## Sweep structures found in the legacy TCL

### Power / speed sweep

`Power_Speed_Sweep.tcl`:

- analogue values: 5, 4, 3, 2, 1
- writing speeds: 2.0 down to 1.5 mm/s in 0.1 mm/s increments
- write line: `dX=-7 mm`
- return/row step: `dX=+7 mm, dY=+0.02 mm`

This is a useful template for a future high-level **parameter sweep** block rather
than requiring the user to manually create every recipe row.

### Velocity / pulse-density style sweep

`Pulse_Sweep.tcl`:

- 5 series
- 20 lines per series
- velocity from 0.2 to 2.1 mm/s in 0.1 mm/s increments
- writing line `dX=-7 mm`
- return `dX=+7 mm, dY=+0.02 mm`
- `dY=+0.1 mm` between series

The exact physical interpretation of the file name "Pulse_Sweep" still depends on
the laser repetition-rate/pulse-picker configuration; the script itself directly
varies **stage velocity**.

## TCL quality / completeness findings

Not every uploaded TCL file should be treated as production-ready.

- `Power_Speed_Sweep.tcl`, `density.tcl`, and `powerino.tcl` execute with
  stubbed HXP commands and are syntactically complete.
- `Pulse_Sweep.tcl` references `$socketID` without opening/defining it in the
  uploaded file, so it appears to be a fragment intended for an existing session
  or copied out of a larger script.
- `loop over pulse density.tcl` has Tcl syntax errors in the uploaded copy
  (including malformed procedure/loop syntax) and should be treated as a draft,
  not as a known-working reference.
- `density.tcl` manually unrolls many nearly identical blocks; the new recipe
  engine should express this as a sweep/loop rather than reproduce that structure.

## LabVIEW project findings

The uploaded LabVIEW project references:

```text
Newport.HXP.CommandInterface
version 1.0.0.0
```

The project contains dedicated VIs for:

- initialize/home
- absolute motion
- incremental line motion
- incremental motion with target velocity
- move-start and move-end events
- XY position
- position recording
- GPIO monitoring
- GPIO set-to-zero
- GPIO set-to-four
- a front-panel controller

The front-panel project was built against LabVIEW 2024-era files; a newer
`HEXAPOD_FRONT_PANEL_v3.vi` also contains **VISA Configure Serial Port**, so at
least one revision appears to communicate with an additional serial device.

The VI binaries confirm the Newport command-interface dependency, but the uploaded
folder does not include the full `Newport.HXP.CommandInterface` dependency itself.

### Important unresolved GPIO discrepancy

The TCL scripts use `GPIO4.DO mask=1 value=1/0` for writing on/off.

The LabVIEW folder contains VIs named:

- `GPIO set to 0.vi`
- `GPIO set to 4.vi`

The VI binary format does not expose enough block-diagram information through the
current inspection tooling to prove whether "4" is the actual digital output value,
a displayed/read-back state, a mask, or an unrelated test value.

**Do not replace the verified-config requirement with a hard-coded 1/0 mapping
until this is reconciled on the lab system.**

## Network / project metadata

The LabVIEW project aliases file contains:

```text
My Computer = "137.195.22.224"
```

This is a LabVIEW project alias for "My Computer". It should not be assumed to be
the HXP controller IP address.

## Changes already made in the Python controller after this review

- Added direct HXP `GPIOAnalogSet` support.
- Added direct HXP
  `HexapodMoveIncrementalControlWithTargetVelocity(..., Line, ...)` support.
- Added real and virtual provider support for target-velocity line moves.
- Added a **Line move at target velocity** block to the Script Builder.
- Added recipe validation/tests for the new line block.
- Kept the physical attenuator unbound pending channel/calibration confirmation.
- Kept real LX13/Pockels mapping behind explicit verification despite the strong
  historical TCL evidence.

## Remaining evidence requested before first real processing run

Most of the software behavior is now clear. The remaining high-value items are:

1. A screenshot/export of the LabVIEW block diagrams for:
   - `GPIO set to 0.vi`
   - `GPIO set to 4.vi`
   - `GPIO Monitoring.vi`
   - `basic labvie inremental with target velocity.vi`
   - `Labview startup sequence.vi`
2. The active HXP `system.ini` and referenced geometry file from the physical
   controller.
3. The exact PHAROS LX13 electrical/pinout documentation, or a wiring diagram for
   the cable presently plugged into LX13.
4. Identification of the serial device referenced by
   `HEXAPOD_FRONT_PANEL_v3.vi` if it is part of the current processing setup.
5. Identification/calibration of the analogue power/attenuation path:
   - current HXP analogue channel
   - commanded voltage/range
   - measured optical transmission or pulse energy versus command.
