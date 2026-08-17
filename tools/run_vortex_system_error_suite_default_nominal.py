from __future__ import annotations

"""Run the system-error suite with nominal selection tied to the no-error config.

The original generic selector treats any sweep containing 1.0 as ratio-like. That
is wrong for fringing sweeps, whose physical nominal is zero sigma. This wrapper
keeps the existing runner unchanged while selecting the unique sweep value whose
builder reproduces ``SystemErrorConfig()`` exactly. It is used by the remote
validation rerun and should later be folded into the main runner.
"""

import importlib.util
from pathlib import Path
from typing import Any, Callable

from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig


_RUNNER = Path(__file__).with_name("run_vortex_system_error_suite.py")
_SPEC = importlib.util.spec_from_file_location("_vortex_system_error_suite_impl", _RUNNER)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_RUNNER}")
_suite = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_suite)

_current_builder: Callable[[Any], SystemErrorConfig] | None = None
_original_run_family = _suite.run_family


def _nominal_value_from_default(values: tuple[Any, ...]) -> Any:
    if _current_builder is None:
        raise RuntimeError("nominal selector used outside run_family")
    default = SystemErrorConfig()
    matches = [value for value in values if _current_builder(value) == default]
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one sweep value to reproduce the default no-error "
            f"configuration, found {len(matches)} among {values!r}"
        )
    return matches[0]


def _run_family_with_default_nominal(family: str, *args: Any, **kwargs: Any):
    global _current_builder
    registry = _suite.system_sweep_registry()
    previous = _current_builder
    _current_builder = registry[family]["builder"]
    try:
        return _original_run_family(family, *args, **kwargs)
    finally:
        _current_builder = previous


_suite._nominal_value = _nominal_value_from_default
_suite.run_family = _run_family_with_default_nominal


if __name__ == "__main__":
    _suite.main()
