# Optical Cockpit — Energy Accounting

**Stage:** 8B  
**Last updated:** 2026-06-18  
**Model status:** `energy_accounting_prediction` / `exposure_bookkeeping`  
**Final export allowed:** False

---

## 1. Purpose

The optical cockpit is the first executable layer of the beam-to-write digital twin.
It computes, in sequence:

1. Energy flow from laser to sample through the full optical chain.
2. Average power at each point in the chain and at the sample.
3. Approximate fluence and peak intensity at the focus.
4. Geometric exposure quantities: pulse spacing, pulses per spot, dose proxy, line-scan summary.

The cockpit does **not** compute material response, beam propagation, or nonlinear effects.
It is a bookkeeping layer — a numerical sanity check for the laser/optics parameter space
before any sample is exposed.

---

## 2. Relation to Stage 8A and Stage 8A.1

Stage 8A defined the three-engine digital twin architecture.  Engine 2 (Exposure / Writing / Dose)
is the engine this stage implements at its most basic level: energy accounting and geometric
exposure bookkeeping.

Stage 8A.1 defined the literature anchors and model status hierarchy.  The energy accounting
equations used here are all in the "no calibration required" tier (analytical formulas derived
from first principles), so the model status `energy_accounting_prediction` applies.

Stage 8C will promote the cockpit by wiring it to real optical-field arrays from Engine 1
(`bessel_twin_core`), enabling `fluence_prediction` from the actual field rather than from
an assumed effective beam area.

---

## 3. What the Energy Cockpit Calculates

### 3.1 Sequential energy flow

Each optical component reduces the pulse energy by a multiplicative factor:

```
E_out,i = E_in,i × T_i × η_i
```

where T_i is the power transmission fraction and η_i is the efficiency fraction (e.g., SLM
diffraction efficiency into all orders, then first-order selection fraction).

The energy at sample is the product of all component factors applied to the input energy:

```
E_sample = E_laser × T_pre × η_SLM × η_+1 × T_relay × T_obj × T_interface × T_extra
```

### 3.2 Average power

```
P_avg = E_p [J] × f_rep [Hz]   [W]
```

with E_p converted from µJ: `E_p [J] = E_p [µJ] × 10⁻⁶`.

### 3.3 Fresnel normal-incidence transmission (optional)

For the air–sample interface when no AR coating is specified:

```
T = 1 - R,   R = ((n₁ - n₂) / (n₁ + n₂))²
```

For air (n₁=1) to ZnSe (n₂=2.4): T ≈ 0.83 (17% Fresnel loss per surface).

### 3.4 Fluence from effective area (Mode A)

Quick cockpit estimate before real optical fields are available:

```
F [J/cm²] = E_sample [J] / A_eff [cm²]
```

where A_eff is the assumed effective beam area in µm², converted to cm²: `A_eff [cm²] = A_eff [µm²] × 10⁻⁸`.

### 3.5 Fluence from normalised intensity array (Mode B — Stage 8C+)

For an intensity array normalised so that `∫|U|² dA = 1 [m⁻²]`:

```
F(x,y) [J/cm²] = E_sample [J] × |U(x,y)|² [m⁻²] / (∫|U|² dA [m²]) × 10⁻⁴
```

Numerically: `F = E_sample × I / (sum(I) × dA_cm²)`.

This conserves energy: `∫F(x,y) dA_cm² = E_sample`.

### 3.6 Peak intensity estimate (approximate)

```
I_peak ≈ F_peak / τ
```

where τ is the pulse duration in seconds (converted from fs: `τ [s] = τ [fs] × 10⁻¹⁵`).

For a Gaussian temporal pulse, a more accurate expression is:

```
I_peak_Gauss = F_peak × √(4 ln2 / π) / τ_FWHM ≈ 0.94 × F_peak / τ_FWHM
```

The default is flat-top (conservative). The result is explicitly labelled approximate.

### 3.7 Exposure bookkeeping

**Pulse spacing (centre-to-centre):**

```
Δs [µm] = v [µm/s] / f_rep [Hz]
```

where v in mm/s is converted: `v [µm/s] = v [mm/s] × 1000`.

**Effective pulses per spot:**

```
N_eff ≈ d_eff [µm] × f_rep [Hz] / v [µm/s]
```

N_eff < 1 means the scan is in the dotted/discontinuous regime.

**Dose-per-unit-length proxy:**

```
D_L = E_p [J] × f_rep [Hz] / v [m/s]   [J/m]
```

This is NOT a calibrated material dose.

**Line scan:**

```
t_line = L / v                              [s]
N_total = int(t_line × f_rep)              [pulses on line]
E_total = E_p × N_total                    [µJ on line]
```

---

## 4. What the Cockpit Does Not Calculate

- **Beam propagation**: the 3D field distribution is not computed at Stage 8B.
  Stage 8C wires the cockpit to `bessel_twin_core` optical fields.
- **Material response**: no threshold crossing, no modification, no damage, no waveguide.
- **Nonlinear effects**: no Kerr self-focusing, no plasma, no filamentation.
- **Heat accumulation**: no shot-to-shot thermal buildup between pulses.
- **Spatial fluence distribution**: Mode A uses an assumed effective area.
  Mode B (from real field) is introduced at Stage 8C.
- **Calibrated predictions**: all outputs are analytical bookkeeping,
  not calibrated against any experiment.

---

## 5. Equations Summary

| Quantity | Formula | Units |
|---|---|---|
| E_sample | E_laser × ∏(T_i × η_i) | µJ |
| P_avg | E_p [J] × f_rep | W |
| T_Fresnel | 1 − ((n₁−n₂)/(n₁+n₂))² | dimensionless |
| F (Mode A) | E_sample / A_eff | J/cm² |
| F(x,y) (Mode B) | E_sample × I / (ΣI × dA_cm²) | J/cm² |
| I_peak (approx) | F_peak / τ | W/cm² |
| Δs | v [µm/s] / f_rep | µm |
| N_eff | d_eff × f_rep / v [µm/s] | pulses |
| D_L | E_p [J] × f_rep / v [m/s] | J/m |
| N_total | int(L [m] × f_rep / v [m/s]) | pulses |

---

## 6. Units

All functions use SI internally and convert at the boundary:

| Input unit | Conversion | Internal unit |
|---|---|---|
| µJ | × 10⁻⁶ | J |
| mm/s | × 10⁻³ | m/s |
| mm/s | × 10³ | µm/s |
| µm² | × 10⁻⁸ | cm² |
| fs | × 10⁻¹⁵ | s |

All transmission and efficiency values must be in [0, 1].  A `ValueError` is raised otherwise.

---

## 7. Warnings and Failure Modes

The energy ledger emits warnings for:

| Condition | Warning |
|---|---|
| Average power exceeds user limit | `"exceeds limit of X W"` |
| Component throughput < 5% | `"Unusually low throughput"` |
| Component throughput = 0% | `"Zero throughput"` |
| Total throughput < 1% | `"Total throughput X% is very low"` |
| No energy reaches sample | `"No energy reaches the sample"` |

The exposure bookkeeping emits warnings for:

| Condition | Warning |
|---|---|
| N_eff < 1 | `"dotted/discontinuous write regime"` |
| N_eff > 1000 | `"very high ... heat accumulation"` |
| Δs > d_eff | `"gaps between spots — track will not be continuous"` |
| N_total = 0 | `"No pulses on line"` |

Invalid inputs (zero or negative scan speed, rep rate, energy) raise `ValueError`.

---

## 8. Notebook Controls

The notebook `notebooks/digital_twin/01_optical_cockpit_energy_accounting.ipynb`
exposes the following user controls as Python variables near the top:

| Variable | Description | Default |
|---|---|---|
| `planning_mode` | Mark outputs as planning estimates | `True` |
| `save_outputs` | Save CSV and figure (blocked if show_caveats=False) | `False` |
| `figure_dpi` | Figure resolution | `180` |
| `show_caveats` | Show caveat panel (must be True to save) | `True` |
| `show_diagnostic_panels` | Render diagnostic figure | `True` |
| `wavelength_nm` | Laser wavelength | `1030.0` |
| `pulse_duration_fs` | Pulse duration | `260.0` |
| `repetition_rate_Hz` | Rep rate | `25000.0` |
| `pulse_energy_before_optics_uJ` | Input pulse energy | `200.0` |
| `average_power_limit_W` | Power limit warning threshold | `10.0` |
| `beam_radius_mm` | 1/e² beam radius at SLM | `2.0` |
| `effective_area_um2` | Assumed focus effective area | `100.0` |
| `slm_diffraction_efficiency` | SLM into all orders | `0.75` |
| `selected_first_order_fraction` | +1 order fraction | `0.73` |
| `relay_transmission` | 4f relay | `0.90` |
| `objective_transmission` | Objective at laser λ | `0.85` |
| `sample_interface_transmission` | AR/Fresnel at surface | `0.95` |
| `scan_speed_mm_s` | Linear scan speed | `1.0` |
| `line_length_um` | Line length | `500.0` |
| `effective_diameter_um` | Beam effective diameter at focus | `3.0` |

Attempting to save outputs with `show_caveats=False` raises `ValueError`.

---

## 9. How Stage 8C Will Connect Real Optical Fields

At Stage 8C the cockpit will be wired to actual SurfaceField outputs from `bessel_twin_core`:

1. `bt.run_case(cfg, preset, path)` returns `result['surface_field']`.
2. The field `U(x,y)` is normalised: `∫|U|² dA = 1 [m⁻²]`.
3. Mode B fluence scaling will replace the Mode A effective-area estimate:
   ```python
   F = scale_intensity_to_fluence_j_cm2(
       np.abs(surface_field.Ex)**2,
       dx_um=surface_field.grid.dx * 1e6,
       dy_um=surface_field.grid.dx * 1e6,
       pulse_energy_uJ=ledger.energy_at_sample_uJ,
   )
   ```
4. `peak_fluence_j_cm2(F)` and `peak_intensity_w_cm2(...)` will use the real field.
5. The effective_area_um2 control will be deprecated in the cockpit after Stage 8C.

---

## 10. Claim Boundaries

| Claim | Allowed at Stage 8B |
|---|---|
| "Energy at sample is X µJ" | Yes — energy accounting prediction |
| "Average power is Y mW" | Yes — energy accounting prediction |
| "Peak fluence is approximately Z J/cm²" | Yes — fluence prediction (approximate, Mode A) |
| "Peak intensity is approximately I W/cm²" | Yes — approximate, labelled as such |
| "Pulse spacing is Δs µm" | Yes — exposure bookkeeping |
| "N_eff pulses overlap at the spot" | Yes — exposure bookkeeping |
| "Material will be modified" | **No** — material response not implemented |
| "Fluence exceeds threshold" | **No** — threshold comparison requires Stage 8C+ |
| "Track will form" | **No** — requires calibrated material response |
| "Waveguide will be written" | **No** — requires calibrated material response |
| "Peak intensity causes ionisation" | **No** — requires nonlinear model + calibration |

**Energy at sample and fluence estimates do not imply material modification.**  
**Peak intensity estimates assume no nonlinear propagation, no plasma defocusing, and no heat accumulation.**  
**Exposure bookkeeping is not a dose-calibrated material-response model.**
