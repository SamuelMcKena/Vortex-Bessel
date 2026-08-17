from __future__ import annotations

from dataclasses import replace

import bessel_twin_core as bt
import pytest
from vbb_study import vbb_regime
from vbb_study.viz_fields import _phase_winding


MEASURED_WINDINGS: dict[tuple[str, str], float] = {}


def _surface_winding(
    regime: str,
    conjugate_mode: str | None = None,
    *,
    allow_vortex_removal: bool = False,
    ell: int | None = None,
) -> tuple[float, int]:
    base = bt.default_config("fast")
    physical = base.physical_axicon
    if conjugate_mode is not None:
        physical = replace(
            physical,
            slm2_conjugate_mode=conjugate_mode,
            allow_vortex_removal=allow_vortex_removal,
        )
    target = base.target if ell is None else replace(base.target, ell=ell)
    cfg = replace(base, generation_method="physical", physical_axicon=physical, target=target)
    cfg = vbb_regime.config_for_regime(cfg, regime)

    result = bt.run_case(
        cfg,
        preset="fast",
        path="realistic",
        case_id=f"{regime}_physical_{conjugate_mode or 'default'}",
    )
    surface = result["surface_field"]
    design = result["design"]
    winding = _phase_winding(
        surface.Ex,
        surface.grid,
        float(design.vortex_main_ring_radius_m),
        n_phi=256,
    )
    MEASURED_WINDINGS[(regime, conjugate_mode or "default")] = winding
    return winding, int(design.ell)


def test_physical_slm2_preserve_vortex_keeps_real_engine_winding() -> None:
    measured: dict[tuple[str, str], float] = {}
    for regime in ("general", "limits"):
        winding, ell = _surface_winding(regime, "preserve_vortex")
        measured[(regime, "preserve_vortex")] = winding
        assert abs(winding - ell) < 0.1

    for regime in ("general", "limits"):
        winding, _ell = _surface_winding(regime, "full", allow_vortex_removal=True)
        measured[(regime, "full")] = winding
        assert abs(winding) < 0.1

    print(
        "SLM2 windings: "
        + ", ".join(
            f"{regime}/{mode}={value:.12g}"
            for (regime, mode), value in sorted(measured.items())
        )
    )


def test_nonzero_vortex_defaults_to_preserve_vortex() -> None:
    winding, ell = _surface_winding("general")
    assert abs(winding - ell) < 0.1


def test_nonzero_vortex_full_requires_explicit_acknowledgement() -> None:
    with pytest.raises(ValueError, match="removes the intended topological charge"):
        _surface_winding("general", "full")


def test_ell_zero_full_conjugation_remains_allowed() -> None:
    winding, ell = _surface_winding("general", "full", ell=0)
    assert ell == 0
    assert abs(winding) < 0.1
