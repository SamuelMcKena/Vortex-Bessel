"""Research-grounded system-error taxonomy for the vortex-Bessel bench.

This is governance/validation infrastructure.  ``implemented_*`` means a model
exists, not that the resulting figure is physically validated or report-ready.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal[
    "implemented_screening",
    "implemented_needs_validation",
    "calibration_required",
    "solver_required",
    "separate_downstream_branch",
]


@dataclass(frozen=True)
class SystemErrorSpec:
    error_id: str
    family: str
    physical_plane: str
    physical_parameter: str
    required_model: str
    validation_target: str
    calibration_blocker: str
    status: Status
    report_ready: bool = False


SYSTEM_ERRORS: tuple[SystemErrorSpec, ...] = (
    SystemErrorSpec(
        "IN-POINT", "input_beam", "before_SLM1", "input pointing angle",
        "beam-fixed oblique Gaussian + SLM carrier + fixed 4F iris",
        "grating-order momentum law and small-angle oblique-axicon benchmark",
        "measured pointing and SLM incidence geometry", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-DEC", "input_beam", "before_SLM1", "lateral beam decentre",
        "translated Gaussian amplitude with no artificial steering phase",
        "Dufour et al.: small transverse misalignment keeps focal line approximately parallel",
        "measured beam/apex centres", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-RAD", "input_beam", "before_SLM1", "beam radius",
        "Gaussian radius changed before SLM1 and full route rebuilt",
        "Gaussian-axicon zone/envelope scaling and zero-error recovery",
        "measured 1/e field radius", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-CURV", "input_beam", "before_SLM1", "wavefront curvature/collimation",
        "independent x/y Gaussian curvature radii as quadratic input phase",
        "ABCD/Fresnel reference plus Shack-Hartmann curvature when measured",
        "measured curvature", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-ELLIP", "input_beam", "before_SLM1", "ellipticity/astigmatic Gaussian",
        "beam-fixed elliptical Gaussian with independent wx/wy",
        "circular limit and independent propagation reference",
        "measured wx/wy/orientation", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "IN-JITTER", "input_beam", "before_SLM1", "shot-to-shot pointing/decentre jitter",
        "ensemble propagation over measured distributions",
        "Monte-Carlo convergence against measured camera time series",
        "measured jitter distribution", "calibration_required",
    ),
    SystemErrorSpec(
        "SLM-LUT", "slm", "SLM1_and_SLM2", "grey-to-phase LUT / phase stroke",
        "pixelwise measured LUT hook plus controlled stroke sensitivity",
        "measured blazed/binary grating diffraction efficiency",
        "measured LUT per panel at 1029 nm and actual incidence/polarisation", "calibration_required",
    ),
    SystemErrorSpec(
        "SLM-SPNU", "slm", "SLM1_and_SLM2", "static flatness / spatial phase nonuniformity",
        "user-supplied per-panel phase map hook",
        "interferometric or Shack-Hartmann residual after compensation",
        "measured panel maps", "calibration_required",
    ),
    SystemErrorSpec(
        "SLM-FRINGE", "slm", "SLM1_and_SLM2", "fringing-field pixel crosstalk",
        "direction-dependent complex-phasor convolution surrogate",
        "fit diffraction efficiencies of measured binary/blazed gratings",
        "panel-specific kernel/subpixel model", "implemented_screening",
    ),
    SystemErrorSpec(
        "SLM-PIX", "slm", "SLM1_and_SLM2", "pixelation/fill factor/quantisation",
        "existing pixelation/quantisation and resolved aperture when grid permits",
        "analytic/order-efficiency checks",
        "confirmed panel geometry/fill factor", "implemented_screening",
    ),
    SystemErrorSpec(
        "SLM-REG", "slm", "SLM1_relative_to_SLM2", "pattern translation/rotation/scale mismatch",
        "transform commanded hologram coordinates before pixelation",
        "identity recovery and imposed transform/order-centre checks",
        "panel registration calibration", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "SLM-INC", "slm", "SLM reflection geometry", "incidence/polarisation-dependent phase response",
        "angle-dependent measured LCOS LUT/Jones response",
        "measured phase response versus incidence/polarisation",
        "panel angular calibration", "calibration_required",
    ),
    SystemErrorSpec(
        "4F-IRIS-X", "fourf", "Fourier_plane", "iris lateral offset",
        "physical iris translated relative to nominal +1 order centre",
        "spectrum/aperture overlap and zero-offset recovery",
        "measured iris position", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "4F-IRIS-R", "fourf", "Fourier_plane", "iris radius/opening",
        "physical circular aperture at the propagated Fourier plane",
        "passed-order purity/bandwidth against measured Fourier-plane image",
        "measured iris diameter/scale", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "4F-LENS-AB", "fourf", "L1_or_L2", "declared/measured lens OPD",
        "explicit lens-plane OPD-map hook inside propagated 4F route",
        "zero-map recovery and measured/manufacturer wavefront replay",
        "lens OPD map for absolute claims", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "4F-DESPACE", "fourf", "L1_or_L2", "longitudinal lens displacement",
        "explicit four propagation distances with fixed object/iris/output planes",
        "ABCD/Fresnel reference and nominal explicit-vs-collapsed 4F parity",
        "measured separations/focal lengths", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "4F-DEC", "fourf", "L1_or_L2", "lens lateral decentre",
        "quadratic thin-lens phase centred on displaced local optical axis",
        "paraxial chief-ray prediction plus diffraction parity",
        "measured lens centres", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "4F-TILT", "fourf", "L1_or_L2", "rigid lens tilt",
        "rotated angular spectrum to local tilted lens plane -> lens -> rotate back",
        "zero-tilt identity, roundtrip spectral checks and paraxial ray benchmark",
        "measured tilt; thick-lens prescription for large-angle absolute claims", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-DEC", "axicon", "axicon", "beam-apex lateral decentre",
        "translate axicon sag/apex relative to beam; no steering phase",
        "Dufour transverse-misalignment focal-line benchmark",
        "measured apex/beam centres and clear aperture", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-TILT", "axicon", "axicon", "rigid axicon tilt",
        "rotated angular spectrum to local axicon plane, conical transmission, rotate back",
        "Thaning et al. oblique broadening/astigmatic-caustic benchmark",
        "measured tilt/angle convention; vector surface refraction for absolute large-angle claims", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-TIP-ROUND", "axicon", "axicon", "rounded/hyperboloidal apex",
        "hyperboloidal defect phase relative to exact-Snell sharp cone",
        "independent Hankel/Fresnel reference and Brzobohaty axial beat scale",
        "measured apex profile", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-TIP-FLAT", "axicon", "axicon", "flat/blunt apex",
        "central flat/truncated sag relative to exact-Snell sharp cone",
        "B0 versus vortex blunt-tip sensitivity benchmark",
        "measured flat-tip radius/profile", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-ANGLE", "axicon", "axicon", "base-angle error",
        "exact normal-incidence refractive Snell cone angle",
        "exact-vs-shallow geometry and core/ring scaling",
        "actual angle and manufacturer convention", "implemented_needs_validation",
    ),
    SystemErrorSpec(
        "AX-N", "axicon", "axicon", "refractive-index uncertainty",
        "index-scale sensitivity inside exact Snell geometry",
        "Sellmeier/manufacturer index reference",
        "actual axicon glass/index at 1029 nm", "implemented_screening",
    ),
    SystemErrorSpec(
        "AX-AP", "axicon", "axicon", "finite clear aperture",
        "co-located aperture on translated local axicon coordinates",
        "finite-zone/hard-edge diffraction and measured aperture replay",
        "actual clear aperture", "calibration_required",
    ),
    SystemErrorSpec(
        "AX-FIG", "axicon", "axicon", "surface-height error",
        "user-supplied surface-height map converted to OPD",
        "metrology-map replay and zero-map recovery",
        "surface profilometry/interferometry", "calibration_required",
    ),
    SystemErrorSpec(
        "WF-ZERN", "wavefront", "declared_input_or_lens_plane", "defocus/astigmatism/coma/spherical",
        "unit-RMS Zernike OPD map in waves at an explicitly declared plane",
        "RMS checks and measured Shack-Hartmann/lens-wavefront comparison",
        "measured wavefront for lab attribution", "implemented_screening",
    ),
    SystemErrorSpec(
        "OBJ-ALIGN", "objective_sample", "objective", "objective alignment/aberration",
        "vector Debye/objective-pupil branch",
        "Phase2C vector controls",
        "objective/relay calibration", "separate_downstream_branch",
    ),
    SystemErrorSpec(
        "SAMPLE-TILT", "objective_sample", "sample_interface", "sample/interface tilt",
        "vector tilted dielectric-interface propagation",
        "vector transversality/energy and tilted-interface references",
        "sample angle/material index", "separate_downstream_branch",
    ),
    SystemErrorSpec(
        "LASER-NOISE", "laser", "source", "pulse-energy/wavelength/time noise",
        "ensemble propagation driven by measured time series/distributions",
        "Monte-Carlo convergence against measured laser statistics",
        "PHAROS measurements", "calibration_required",
    ),
)


def system_error_rows() -> list[dict[str, object]]:
    return [asdict(spec) for spec in SYSTEM_ERRORS]


def report_ready_errors() -> tuple[str, ...]:
    return tuple(spec.error_id for spec in SYSTEM_ERRORS if spec.report_ready)


def pending_errors() -> tuple[str, ...]:
    return tuple(spec.error_id for spec in SYSTEM_ERRORS if not spec.report_ready)
