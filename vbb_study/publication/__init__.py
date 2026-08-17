"""Publication output helpers for the structured-beam atlas.

This subpackage owns the canonical column schema for scalar CSV outputs,
propagation power QA utilities, and helpers for assembling paper-ready
summary tables.

Submodules
----------
tables
    Canonical scalar CSV schema (column names, types, descriptions),
    power-QA label function, and row-annotation helpers.
lab_realism
    Phase 8 lab-realism CSV metadata and route/plane terminology helpers.
vector
    Phase 9 vector-beam CSV metadata and hardware-feasibility helpers.
materials
    Stage 6 materials-proxy CSV metadata and calibration-status helpers.
capsule
    Stage 7 capsule/application-geometry CSV metadata and proxy labels.
advanced
    Stage 8 hexagonal, polygonal, and discrete N-fold CSV metadata and
    hardware/progagation honesty labels.
captions
    Standard caption/caveat policies for governed exports.
figure_registry
    Stage 8.6 output-family allow/quarantine registry and export gate.
visuals
    Stage 8.7 reusable diagnostic visualisation helpers.
quicklook
    Stage 8.7 quick-look simulator configuration, runner, and writers.
"""

from __future__ import annotations

from . import advanced, capsule, captions, figure_registry, lab_realism, materials, tables, vector, visuals, quicklook

__all__ = [
    "advanced",
    "capsule",
    "captions",
    "figure_registry",
    "lab_realism",
    "materials",
    "quicklook",
    "tables",
    "vector",
    "visuals",
]
