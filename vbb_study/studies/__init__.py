"""Study entry points and case registries for the structured-beam atlas.

This subpackage owns high-level study runners that tie the physics engine to
canonical output schemas.  Each module corresponds to one study branch.

Submodules
----------
scalar_cases
    Shortlist presets and helpers for scalar Bessel-Gauss sweep studies.
    These functions are the recommended entry points for regenerating scalar
    CSVs from the canonical schema in ``vbb_study.publication.tables``.
polygonal_cases
    Stage 8 hexagonal/polygonal case definitions.
discrete_nfold_cases
    Stage 8 discrete N-fold case definitions.
"""

from __future__ import annotations

from . import discrete_nfold_cases, polygonal_cases, scalar_cases

__all__ = ["discrete_nfold_cases", "polygonal_cases", "scalar_cases"]
