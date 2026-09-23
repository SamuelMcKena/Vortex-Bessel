from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .recipe import Recipe, RecipeStep


def numeric_range(start: float, stop: float, step: float) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = float(step)
    if step == 0:
        raise ValueError("sweep step cannot be zero")
    if (stop - start) * step < 0:
        raise ValueError("sweep step points away from stop")

    values: list[float] = []
    value = start
    eps = abs(step) * 1e-9 + 1e-12
    if step > 0:
        while value <= stop + eps:
            values.append(round(value, 12))
            value += step
    else:
        while value >= stop - eps:
            values.append(round(value, 12))
            value += step
    return values


@dataclass(frozen=True, slots=True)
class RasterSweepSpec:
    name: str = "Writing sweep"
    write_dx_mm: float = -7.0
    row_pitch_mm: float = 0.02
    series_spacing_mm: float = 0.10
    velocity_start_mm_s: float = 0.2
    velocity_stop_mm_s: float = 2.1
    velocity_step_mm_s: float = 0.1
    attenuation_start_percent: float = 25.0
    attenuation_stop_percent: float = 25.0
    attenuation_step_percent: float = 1.0
    include_attenuator_steps: bool = False
    return_velocity_mm_s: float = 10.0

    def velocities(self) -> list[float]:
        values = numeric_range(
            self.velocity_start_mm_s,
            self.velocity_stop_mm_s,
            self.velocity_step_mm_s,
        )
        if any(v <= 0 for v in values):
            raise ValueError("all writing velocities must be > 0")
        return values

    def attenuations(self) -> list[float]:
        if not self.include_attenuator_steps:
            return [float(self.attenuation_start_percent)]
        values = numeric_range(
            self.attenuation_start_percent,
            self.attenuation_stop_percent,
            self.attenuation_step_percent,
        )
        if any(not 0.0 <= v <= 100.0 for v in values):
            raise ValueError("attenuator sweep must remain within 0–100 %")
        return values

    @property
    def return_dx_mm(self) -> float:
        return -float(self.write_dx_mm)


@dataclass(frozen=True, slots=True)
class RasterSweepSummary:
    series_count: int
    lines_per_series: int
    total_write_lines: int
    total_steps: int
    approximate_motion_time_s: float


def build_raster_sweep(spec: RasterSweepSpec) -> tuple[Recipe, RasterSweepSummary]:
    velocities = spec.velocities()
    attenuations = spec.attenuations()
    if not velocities:
        raise ValueError("velocity sweep produced no values")
    if not attenuations:
        raise ValueError("attenuator sweep produced no values")
    if spec.return_velocity_mm_s <= 0:
        raise ValueError("return velocity must be > 0")

    steps: list[RecipeStep] = []
    motion_time = 0.0

    for series_index, attenuation in enumerate(attenuations):
        if spec.include_attenuator_steps:
            steps.append(RecipeStep.pockels_cell(False))
            steps.append(RecipeStep.attenuator_set(attenuation))

        for line_index, velocity in enumerate(velocities):
            steps.append(RecipeStep.pockels_cell(True))
            steps.append(
                RecipeStep.move_line_velocity(
                    spec.write_dx_mm,
                    0.0,
                    0.0,
                    velocity,
                )
            )
            steps.append(RecipeStep.pockels_cell(False))

            # Return to the starting X while stepping to the next row.
            dy = spec.row_pitch_mm
            if (
                line_index == len(velocities) - 1
                and series_index < len(attenuations) - 1
            ):
                dy += spec.series_spacing_mm
            steps.append(
                RecipeStep.move_line_velocity(
                    spec.return_dx_mm,
                    dy,
                    0.0,
                    spec.return_velocity_mm_s,
                )
            )

            motion_time += abs(spec.write_dx_mm) / velocity
            motion_time += (
                (spec.return_dx_mm**2 + dy**2) ** 0.5
                / spec.return_velocity_mm_s
            )

    # A known-safe final command is explicit even though every row already closes.
    steps.append(RecipeStep.pockels_cell(False))
    recipe = Recipe(name=spec.name, steps=steps)
    summary = RasterSweepSummary(
        series_count=len(attenuations),
        lines_per_series=len(velocities),
        total_write_lines=len(attenuations) * len(velocities),
        total_steps=len(steps),
        approximate_motion_time_s=motion_time,
    )
    return recipe, summary
