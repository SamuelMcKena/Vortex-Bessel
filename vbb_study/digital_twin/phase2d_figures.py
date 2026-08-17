"""Publication-style 2D governance figures for Phase 2D."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2b_figures import _mpl, _save
from vbb_study.solver_policy import BEAM_CASES, CLAIM_TYPES


def plot_solver_policy_matrix(rows: Sequence[Mapping[str, Any]], root: Path) -> dict[str, str]:
    plt, colors, _ = _mpl()
    automatic = {
        (str(row["beam_case"]), str(row["claim_type"])): row
        for row in rows
        if row["requested_mode"] == "automatic_by_claim"
    }
    matrix = np.zeros((len(BEAM_CASES), len(CLAIM_TYPES)), dtype=float)
    for iy, beam in enumerate(BEAM_CASES):
        for ix, claim in enumerate(CLAIM_TYPES):
            matrix[iy, ix] = 1.0 if automatic[(beam, claim)]["selected_objective_solver"] == "vector_debye" else 0.0
    cmap = colors.ListedColormap(["#4C78A8", "#E45756"])
    fig, ax = plt.subplots(figsize=(14.8, 4.7), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=1.5, aspect="auto")
    ax.set_xticks(np.arange(len(CLAIM_TYPES)), [claim.replace("_", "\n") for claim in CLAIM_TYPES])
    ax.set_yticks(np.arange(len(BEAM_CASES)), BEAM_CASES)
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            ax.text(ix, iy, "V" if matrix[iy, ix] else "S", ha="center", va="center", color="white", weight="bold")
    ax.set_title("Phase 2D automatic objective-solver policy | S scalar FFT, V vector Debye")
    ax.set_xlabel("claim")
    ax.set_ylabel("canonical case")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1], shrink=0.8)
    colorbar.ax.set_yticklabels(["scalar screening", "vector reference"])
    paths = _save(fig, root / "solver_policy_matrix", dpi=360)
    plt.close(fig)
    return {"figure_id": "solver_policy_matrix", "png_path": str(paths[0]), "pdf_path": str(paths[1])}


def plot_calibration_readiness(rows: Sequence[Mapping[str, Any]], root: Path) -> dict[str, str]:
    plt, _, _ = _mpl()
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        measurement = str(row["required_measurement"])
        if measurement != "none":
            unique.setdefault(measurement, row)
    measurements = list(unique)
    available = np.asarray([1.0 if unique[name]["current_status"] == "available" else 0.0 for name in measurements])
    severity = np.asarray([1.0 if bool(unique[name]["blocks_claim"]) else 0.35 for name in measurements])
    order = np.argsort(-severity)
    measurements = [measurements[index] for index in order]
    available = available[order]
    severity = severity[order]
    y = np.arange(len(measurements))
    colours = ["#009E73" if value else "#D55E00" for value in available]
    fig, ax = plt.subplots(figsize=(12.5, max(6.0, 0.34 * len(measurements))), constrained_layout=True)
    ax.barh(y, severity, color=colours, alpha=0.92)
    ax.set_yticks(y, measurements)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.12)
    ax.set_xticks([0.35, 1.0], ["review", "blocking"])
    for index, name in enumerate(measurements):
        row = unique[name]
        label = f"{row['current_status']} | {row['current_source']}"
        ax.text(0.02, index, label, va="center", ha="left", color="white" if severity[index] > 0.7 else "black", fontsize=7.8)
    ax.set_title("Phase 2D laboratory calibration readiness | orange values block at least one calibrated claim")
    ax.set_xlabel("blocking severity")
    paths = _save(fig, root / "calibration_readiness", dpi=360)
    plt.close(fig)
    return {"figure_id": "calibration_readiness", "png_path": str(paths[0]), "pdf_path": str(paths[1])}


def plot_uncertainty_summary(summary: Mapping[str, Mapping[str, Any]], root: Path) -> dict[str, str]:
    plt, _, _ = _mpl()
    available = [
        (name, row) for name, row in summary.items()
        if row.get("uncertainty_status") == "available_from_supplied_uncertainty"
        and row.get("nominal") not in (None, 0.0)
    ]
    labels = [name.replace("_", " ") for name, _ in available]
    relative = np.asarray([
        100.0 * float(row["standard_uncertainty"]) / abs(float(row["nominal"]))
        for _, row in available
    ])
    y = np.arange(len(labels))
    colours = ["#4C78A8", "#F2CF5B", "#54A24B", "#E45756", "#72B7B2"]
    fig, ax = plt.subplots(figsize=(11.5, max(5.2, 0.42 * len(labels))), constrained_layout=True)
    ax.barh(y, relative, color=[colours[index % len(colours)] for index in range(len(labels))])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("relative standard uncertainty (%)")
    ax.set_title("Synthetic software-validation propagation | not experimental calibration")
    ax.grid(axis="x", alpha=0.22)
    paths = _save(fig, root / "uncertainty_summary" / "synthetic_full_uncertainty", dpi=360)
    plt.close(fig)
    return {"figure_id": "synthetic_uncertainty_summary", "png_path": str(paths[0]), "pdf_path": str(paths[1])}


def plot_calibration_state_comparison(rows: Sequence[Mapping[str, Any]], root: Path) -> dict[str, str]:
    plt, _, _ = _mpl()
    h1 = [row for row in rows if row["beam_case"] == "H1"]
    labels = [str(row["calibration_scenario"]).replace("_", "\n") for row in h1]
    dimensional = np.asarray([1.0 if row["dimensional_readiness"] == "ready" else 0.0 for row in h1])
    fluence = np.asarray([1.0 if row["fluence_readiness"] == "ready" else 0.0 for row in h1])
    x = np.arange(len(h1))
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.bar(x - 0.18, dimensional, width=0.36, color="#4C78A8", label="dimension inputs complete")
    ax.bar(x + 0.18, fluence, width=0.36, color="#E45756", label="fluence inputs complete")
    ax.set_xticks(x, labels)
    ax.set_yticks([0, 1], ["blocked", "software inputs complete"])
    ax.set_ylim(0.0, 1.2)
    ax.set_title("H1 calibration bridge states | synthetic rows are not calibrated laboratory results")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    paths = _save(fig, root / "calibrated_vs_uncalibrated" / "h1_calibration_state_comparison", dpi=360)
    plt.close(fig)
    return {"figure_id": "h1_calibration_state_comparison", "png_path": str(paths[0]), "pdf_path": str(paths[1])}
