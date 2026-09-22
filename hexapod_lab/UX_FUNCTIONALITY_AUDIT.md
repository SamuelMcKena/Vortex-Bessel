# Hexapod Lab — usability and functionality audit

Date: 2026-09-22

This audit reviews the standalone `hexapod_lab/` controller as an operator-facing
lab tool rather than only as a code prototype.

## Audit outcome

The first v0.1 layout was functional but not operator-friendly enough for repeated
lab use. The largest problem was that manual control, advanced wiring settings and
script construction were all visible at once. It also used the generic word
"gate" for the PHAROS process-beam command, which obscured the operator's actual
intent: **open or close the Pockels cell**.

The GUI has therefore been reorganized around three jobs:

1. **CONTROL** — normal manual operation.
2. **SCRIPT BUILDER** — experiment sequence construction and execution.
3. **SETUP + DIAGNOSTICS** — infrequent hardware/network/wiring configuration.

The global header remains visible on every tab and always shows stage state, beam
state and attenuator state, with permanent **CLOSE BEAM** and
**STOP MOTION + CLOSE BEAM** actions.

## Findings and changes

### 1. Script controls were permanently in the operator's way

**Finding:** The recipe builder occupied the bottom of the main control screen even
when the operator only wanted to jog the stage, set attenuation or inspect the
digital twin.

**Resolution:** Script construction now has its own tab. The Control tab contains
only normal manual operation, the 3D digital twin, the 2D path map, live pose,
Pockels controls and attenuation controls.

### 2. "Laser gate ON/OFF" was ambiguous

**Finding:** The old wording could be read as switching the PHAROS source itself.

**Resolution:** Operator-facing controls now say:

- **LASER ON — OPEN POCKELS CELL**
- **LASER OFF — CLOSE POCKELS CELL**

The UI explicitly states that this is the process-beam/Pockels command and does
not power-cycle the PHAROS laser source. The existing provider field
`gate_enabled` is retained internally for compatibility but exposes a
`pockels_open` property.

### 3. Beam closure was not globally prominent enough

**Finding:** Beam-off should not require navigating to a particular panel.

**Resolution:** **CLOSE BEAM** is permanently visible in the header. A separate
**STOP MOTION + CLOSE BEAM** software-abort action is also always visible. The GUI
does not call this an emergency stop because the hardware E-stop/interlock remains
the authoritative safety system.

### 4. Real/virtual provider switching could be confusing

**Finding:** Earlier code could display "Real HXP" while the retained active object
was still the virtual provider if no real connection had been established. Similar
confusion was possible for the laser provider.

**Resolution:** Commands now resolve the selected provider at command time and
raise a clear "real provider not connected" error rather than silently falling
back to simulation. Manual controls are disabled when the selected stage is not
connected.

### 5. Switching away from a real beam provider could hide an open command

**Finding:** A mode switch must never make a previously selected real Pockels
provider disappear from view while it remains commanded open.

**Resolution:** Changing Pockels provider mode first requests closure on every
Pockels provider that may have been touched. Reconnecting/replacing the HXP also
tears down the old real Pockels provider so it cannot retain a stale HXP client.

### 6. Real beam enable needed a deliberate manual arm

**Finding:** A single click should not open the real Pockels cell merely because a
real provider is selected.

**Resolution:** Manual real-beam opening additionally requires
**ARM MANUAL REAL-BEAM CONTROL**. Closing the beam never requires that arm state.
Scripts use a separate **ARM REAL SCRIPT EXECUTION** control.

### 7. Attenuator control was absent

**Resolution:** A normalized attenuator subsystem is now present in both manual
control and scripts.

Manual control includes:

- transmission setpoint from 0–100 %;
- slider and numeric entry;
- 0/25/50/75/100 % quick presets;
- explicit **SET ATTENUATOR** action;
- live header/status indication.

Script steps can set attenuator transmission. Preflight validates the 0–100 %
range and warns if attenuation is changed while the Pockels cell is open.

The GUI intentionally uses a normalized **transmission percent** abstraction.
A future device-specific provider can map that value through the measured
calibration to motor angle, analogue voltage or another native quantity.

The physical attenuator make/model/protocol is not yet known, so real attenuator
control is deliberately represented by a provider that refuses hardware commands.
No serial protocol, voltage range, motor angle or calibration has been guessed.

### 8. Advanced wiring fields cluttered normal operation

**Finding:** GPIO name, masks and active values are setup information, not everyday
controls.

**Resolution:** HXP network settings, LX13/HXP GPIO mapping, wiring verification,
attenuator hardware binding, CAD loading and diagnostic tools now live on
**SETUP + DIAGNOSTICS**.

### 9. Recipe safety needed stronger preflight

**Resolution:** Preflight now checks:

- invalid motion payloads;
- Pockels cell left open at the end of a recipe;
- duplicate Pockels state commands;
- attenuator range;
- attenuation changes while Pockels is open;
- selected real provider not connected/armed;
- real attenuator selected without a bound driver;
- mixed real/virtual provider execution.

Every script starts by requesting the Pockels cell closed and every normal
completion/failure/stop requests closure again.

### 10. Manual interference during a running script

**Finding:** Jogging, opening the beam, changing attenuation or switching providers
during a script could invalidate the sequence.

**Resolution:** Manual motion, beam-open, attenuation and provider switching are
disabled while a script is active. **LASER OFF / CLOSE POCKELS** and the global
stop/close actions remain available.

### 11. Motion-mode switching during movement

**Resolution:** Stage provider switching is disabled while the selected stage is
moving. This prevents losing the active motion controls by changing from real to
virtual mid-trajectory.

### 12. Path visualization

The Control tab retains:

- exact STEP articulation;
- fixed lab beam terminating at the sample top surface;
- sample-local written path;
- 2D gridded map;
- selectable laser-on-sample path vs HXP XY carriage path;
- adjustable map span.

## Current operator workflow

### Ordinary manual use

1. Open **CONTROL**.
2. Select Virtual or Real HXP in the header.
3. Connect stage if required.
4. Jog or move the target pose.
5. Set attenuator transmission.
6. Use **LASER ON / OPEN POCKELS CELL** and
   **LASER OFF / CLOSE POCKELS CELL** as required.
7. Watch live XYZUVW, actuator geometry, exact CAD and 2D path.

### Scripted experiment

1. Open **SCRIPT BUILDER**.
2. Add motion, Pockels, attenuator and wait steps.
3. Drag to reorder.
4. Review the live preflight panel and selected execution providers.
5. Arm real script execution only if real hardware is intentionally selected.
6. Run; stop remains available at all times.

### Hardware setup

1. Open **SETUP + DIAGNOSTICS**.
2. Configure/connect HXP.
3. Configure and verify PHAROS LX13/HXP GPIO mapping.
4. Bind the attenuator once its actual controller/model and calibration are known.
5. Load exact STEP if desired.
6. Use firmware/pose diagnostics and save the hardware config.

## Remaining hardware-bound work

These are intentionally not claimed complete:

- laboratory validation of the direct HXP TCP command path;
- confirmed PHAROS LX13 pinout/active level/electrical interface;
- physical Pockels readback, if the installed interface exposes one;
- device-specific real attenuator driver and measured transmission calibration;
- controller-native high-rate synchronized trajectory/event output;
- measured lab-to-HXP/sample registration for a physically calibrated beam axis.

Those items require the actual lab hardware/documentation. The GUI now has defined
provider boundaries for them rather than embedding guesses in the operator UI.
