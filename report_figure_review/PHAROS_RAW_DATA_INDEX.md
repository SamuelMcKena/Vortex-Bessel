# Recovered 11 November 2026 scope experiment — evidence index

The original user-supplied `Laser_Lab_Results_2026-11-11.zip` contains actual DSO3062A CH1/CH2 raw_waveforms.csv records and jitter_delays.csv. In this GitHub branch the raw binary archive itself has **not** yet been uploaded; it is provided in the separate downloadable Claude repair handoff ZIP.

## Original experiment series
M1: electrical-only control; M2c: count=0 continuous train; M5a: count=1 first-window detector event; M5b: separately captured +1 s window; P1: paired 3 ms gate requests; J1/J2: 30 raw timing values each at nominal 1 kHz and 1 Hz.

## Recalculation from 60 original individual jitter-delay CSV rows

| Nominal picker output | n | mean μs | median μs | sample SD μs | min–max μs | KS D vs U(0,40 μs) | p |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 kHz | 30 | 21.9541 | 24.2072 | 10.6121 | 4.5428–39.3082 | 0.15591 | 0.41681 |
| 1 Hz | 30 | 21.7273 | 21.0088 | 10.3214 | 1.5060–39.3029 | 0.16097 | 0.37797 |

Timing definition: LX8 falling 2.5 V threshold to CH2 50%-rise on exported sample-grid interpolation. This is detector timing, not calibrated photon-emission time. A distribution within a 40 μs period is compatible with, but does not prove, a 25 kHz internal-clock wait model. The KS non-rejection is NOT evidence of causation.

The old F6.1 contained a schematic CH1/CH2 trace rather than raw data and must be replaced. Sample counts/readback settings, sensor attenuation and detector linearity need provenance. Cannot infer 100-pulse rising envelope, energy stability, or a no-later-pulses guarantee from short acquisition windows. The separate 16 November raw archive remains unavailable to this GitHub branch.