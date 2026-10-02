# Legacy LabVIEW + TCL deep analysis

Source set: the uploaded **TCL Scripts.zip** and **Labview VIS.zip**.

This document records source-derived behavior from the old lab controller. The
LabVIEW VIs were not treated as opaque binaries: their RSRC `FPHb` and `BDHb`
heaps were decompressed and inspected, which exposes front-panel labels, .NET HXP
method nodes, constant strings and many numeric constants.

## The important correction

A shallow strings-only inspection understated how much information is present in
the VIs. The compressed LabVIEW heaps recover most of the old controller's HXP
addressing, GPIO channels, masks/states, motion APIs and operator controls.

There are, however, **multiple historical GPIO mappings** in the supplied files,
so the correct conclusion is not to hard-code the first mapping found. The files
show a TCL-era mapping and a later LabVIEW-front-panel mapping.

## HXP connection parameters recovered from the LabVIEW VIs

The HXP-related VIs repeatedly contain the same connection constants:

```text
HXP address : 192.168.0.254
Port        : 5001
Timeout     : 10000 ms
Group       : HEXAPOD
Move frame  : Work
```

The same address/port/timeout combination is present throughout the uploaded
motion, GPIO, homing, position and front-panel VIs, including
`HEXAPOD_FRONT_PANEL_v3.vi`.

This is substantially stronger evidence for the historical lab configuration
than the generic address previously used in the Python example configuration.

## Motion behavior recovered from LabVIEW

The LabVIEW controller explicitly uses the Newport
`Newport.HXP.CommandInterface` assembly, version `1.0.0.0`.

Recovered methods include:

- `OpenInstrument`
- `CloseInstrument`
- `GroupInitialize`
- `GroupHomeSearch`
- `GroupStatusGet`
- `GroupPositionCurrentGet`
- `GroupPositionSetpointGet`
- `HexapodMoveAbsolute`
- `HexapodMoveIncrementalControlWithTargetVelocity`
- `GPIODigitalGet`
- `GPIODigitalSet`
- `GPIOAnalogGet`

The line-motion VIs and front panels explicitly contain:

```text
GroupName             = HEXAPOD
CoordinateSystem      = Work
HexapodTrajectoryType = Line
Velocity              = operator value
```

The old front panel exposes three movement modes:

- **ABS Move**
- **LINE Move**
- **LINE Move_While Write**

It also exposes X/Y/Z increments and a Velocity control. This supports carrying
the same quick manual line-writing workflow into the Python Control tab instead
of making every line operation a script.

## TCL-era process-beam behavior

Three complete TCL scripts use the same writing sequence:

```tcl
GPIODigitalSet $socketID GPIO4.DO 1 1
HexapodMoveIncrementalControlWithTargetVelocity \
    $socketID HEXAPOD Work Line -7 0 0 <velocity>
GPIODigitalSet $socketID GPIO4.DO 1 0
```

Therefore the TCL source directly supports:

```text
GPIO       GPIO4.DO
Mask       1
Write      1
Non-write  0
```

This is strong historical software evidence. It still does not by itself prove
which physical HXP terminal is presently wired to the populated PHAROS LX13
connector.

## Later LabVIEW front-panel Pockels logic

The later front panels reveal a different and more detailed arrangement.

`HEXAPOD_FRONT_PANEL_v3.vi` contains the explicit block-diagram comment:

> Pockels Cell control and Control During Writing motion

In that section, repeated `GPIODigitalSet` nodes use:

```text
GPIOName            = GPIO3.DO
Mask                = 1
DigitalOutputValue  = 0 or 1
```

The earlier `HEXAPOD_FRONT_PANEL.vi` contains the same GPIO3.DO / mask-1 /
0-or-1 pattern.

The recovered heap proves the channel, mask and two commanded states. It does
**not** by itself establish which of 0/1 physically corresponds to the current
PHAROS Pockels OPEN state, because the supplied software spans different hardware
revisions and the current LX13 cable routing is not encoded in the VI.

For that reason the Python controller should present this as a
**Legacy LabVIEW Pockels candidate** rather than silently arming it.

## GPIO1.DO is a separate gate/writing-state marker

The uploaded LabVIEW files strongly distinguish `GPIO1.DO` from the GPIO3
Pockels-control section.

### Dedicated state VIs

Deep heap comparison of:

- `GPIO set to 0.vi`
- `GPIO set to 4.vi`

recovers exactly:

```text
GPIOName = GPIO1.DO
Mask     = 4
```

and the differing numeric constants are exactly:

```text
GPIO set to 0.vi -> DigitalOutputValue = 0
GPIO set to 4.vi -> DigitalOutputValue = 4
```

### Monitoring

`GPIO Monitoring.vi` calls `GPIODigitalGet` on:

```text
GPIO1.DO
```

and its front-panel indicator is explicitly labelled:

```text
Gate Open?
```

`XYposonlywriting.vi` also reads `GPIO1.DO` and contains the comment:

> recording graph data only whne gate is open

### Motion events

`MoveStartEvent.vi` configures:

```text
ExtendedEventName  = HEXAPOD.1.SGamma.MotionStart
ExtendedActionName = GPIO1.DO.DOToggle
```

`MoveEndEvent.vi` configures:

```text
ExtendedEventName  = HEXAPOD.1.SGamma.MotionEnd
ExtendedActionName = GPIO1.DO.DOToggle
```

Taken together, this strongly indicates that GPIO1.DO bit 2 (mask/value 4) was
used as a **motion/writing gate-state marker** that could be toggled by native HXP
motion-start/end events and read by plotting/recording code.

That is more specific than the earlier assumption that every GPIO reference was
the direct physical Pockels command.

## GPIO4.DO also remains present in v3

The v3 front panel contains a later `GPIODigitalSet` on `GPIO4.DO` near the
writing-line logic, and the TCL scripts use GPIO4.DO as their writing gate.

The v3 heap records one observed GPIO4.DO digital-set constant pair as mask 4 /
value 1. Its exact physical role cannot be assigned confidently from that
observation alone.

This is why the recovered profiles retain GPIO4.DO as evidence rather than
merging it with the GPIO3 or GPIO1 roles.

## Power / attenuation findings

### TCL scripts

`powerino.tcl` and `Power_Speed_Sweep.tcl` use:

```tcl
GPIOAnalogSet $socketID GPIO2.DAC1 <value>
```

with values from 1 through 5.

So the TCL-era processing setup directly supports a historical analogue
power/attenuation path through `GPIO2.DAC1`.

`Pulse_Sweep.tcl` instead contains:

```tcl
GPIOAnalogSet $socketID GPIO4.AO 5
```

which is another indication that the uploaded source spans more than one
hardware/configuration revision.

### LabVIEW v3

The v3 front panel exposes:

- **Power Setpoint**
- **Power Control**
- **Beam Block**

and its block diagram contains:

```text
GPIOAnalogGet
GPIO2.DAC1
```

It also contains two **VISA Configure Serial Port** nodes with:

```text
VISA resource name = COM7
```

This strongly suggests that the later power-control implementation included a
serial device while still monitoring an HXP analogue channel. The exact COM7
device command protocol is not established by the recoverable strings alone, so
the Python real-attenuator driver should remain unbound until that protocol is
known or directly tested.

## Legacy writing/sweep structures

### Power_Speed_Sweep.tcl

- analogue setpoints: 5, 4, 3, 2, 1
- writing speeds: 2.0 to 1.5 mm/s in -0.1 mm/s steps
- write line: dX = -7 mm
- row return: dX = +7 mm, dY = +0.02 mm

### powerino.tcl

- analogue setpoints: 5 to 1
- writing line: dX = -7 mm
- writing speed: 1 mm/s
- return/row step: dX = +7 mm, dY = +0.02 mm

### density.tcl

- manually unrolled writing velocities covering the density sweep
- write line: dX = -7 mm
- row pitch: +0.02 mm
- digital writing control: GPIO4.DO mask 1 values 1/0

### Pulse_Sweep.tcl

- five series
- 20 lines per series
- velocity 0.2 to 2.1 mm/s
- write dX = -7 mm
- return dX = +7 mm, dY = +0.02 mm
- +0.1 mm Y spacing between series
- references an existing `$socketID`, so the uploaded file is not standalone

### loop over pulse density.tcl

The supplied version contains Tcl syntax errors and should be treated as a draft,
not a known-working production script.

## Changes made to the Python controller from this analysis

The standalone branch now includes:

- direct HXP target-velocity Line motion;
- a Script Builder **Line move at target velocity** block;
- `GPIOAnalogSet`;
- `GPIOAnalogGet`;
- virtual target-velocity-line simulation and tests;
- a machine-readable
  `assets/legacy_hardware_evidence.json` containing the recovered profiles;
- the existing Pockels/attenuator provider boundary remains fail-closed rather
  than choosing between conflicting historical mappings without commissioning.

## What the uploaded files now answer

They answer most of the software-side questions:

- historical HXP IP: **192.168.0.254**
- HXP port: **5001**
- historical HXP timeout: **10000 ms**
- group: **HEXAPOD**
- processing frame: **Work**
- writing trajectory: native HXP **Line + target velocity**
- TCL writing gate: **GPIO4.DO / mask 1 / 1 then 0**
- later LabVIEW Pockels candidate: **GPIO3.DO / mask 1 / 0 and 1**
- LabVIEW writing-state marker: **GPIO1.DO / mask 4 / 0 and 4**
- native motion start/end action: **GPIO1.DO.DOToggle**
- historical TCL analogue power path: **GPIO2.DAC1**
- later LabVIEW analogue monitor: **GPIO2.DAC1**
- later LabVIEW serial resource: **COM7**
- exact front-panel workflows and operator terminology.

## What is genuinely not contained in these files

A few physical facts cannot be proven from software artifacts alone:

1. Which HXP physical terminal/cable is **currently** routed to the populated
   PHAROS LX13 connector.
2. The electrical characteristics/pinout of LX13 itself.
3. Whether the present lab wiring corresponds to the TCL-era GPIO4 mapping or the
   later LabVIEW GPIO3 arrangement.
4. The identity/protocol of the COM7 device.
5. The calibrated relation between analogue/serial power setpoint and actual
   optical transmission or pulse energy.
6. The active controller `system.ini` / geometry file, which is not present in
   the uploaded LabVIEW/TCL archives.

Those should be treated as commissioning checks, not reasons to leave the rest of
the controller unspecified.
