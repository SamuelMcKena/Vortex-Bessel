"""Lazy access to the legacy public facade.

Use this only for behaviour that still belongs to ``bessel_twin_core`` during
the staged decomposition. Leaf modules should prefer direct imports from
``vbb_study.config``, ``vbb_study.design``, and ``vbb_study.equations``.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType


@lru_cache(maxsize=1)
def core() -> ModuleType:
    """Return the legacy facade module on first runtime use."""

    return import_module("bessel_twin_core")


__all__ = ["core"]
