"""Figure style and export helpers for the vortex-Bessel-beam study.

This module is the single place where I choose plotting defaults for the study.
The goal is not cosmetic uniformity for its own sake: every figure should make
the measured quantity, display transform, and units explicit enough that I can
reuse it in a manuscript without reverse-engineering the notebook cell.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

INLINE_DPI = 120
SAVE_DPI = 220

INTENSITY_CMAP = "inferno"
SIGNED_CMAP = "RdBu_r"
PHASE_CMAP = "twilight"

CATEGORICAL_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

FIGURE_NAME_PATTERN = "NN_topic_subject.png"
CSV_NAME_PATTERN = "NN_topic_subject.csv"
CAPTIONS_MANIFEST_NAME = "captions_manifest.jsonl"


def _slug(value: object) -> str:
    """Return a conservative snake-case token for filenames.

    I keep the naming rule boring on purpose: generated artefacts should sort
    predictably and survive being copied between Windows, Linux, and manuscript
    folders without surprises.
    """

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "untitled"


def artifact_name(index: int, topic: str, subject: str, suffix: str) -> str:
    """Build the shared ``NN_topic_subject.ext`` artefact filename.

    Parameters
    ----------
    index:
        Notebook or figure index. It is zero padded to two digits.
    topic, subject:
        Human-readable labels that are converted to snake case.
    suffix:
        File extension with or without the leading dot.
    """

    ext = str(suffix).strip()
    if not ext:
        raise ValueError("suffix must not be empty.")
    if not ext.startswith("."):
        ext = "." + ext
    return f"{int(index):02d}_{_slug(topic)}_{_slug(subject)}{ext}"


def figure_name(index: int, topic: str, subject: str) -> str:
    """Return a publication figure filename using the Stage 0 convention."""

    return artifact_name(index, topic, subject, ".png")


def csv_name(index: int, topic: str, subject: str) -> str:
    """Return a publication CSV filename using the Stage 0 convention."""

    return artifact_name(index, topic, subject, ".csv")


def apply_style(*, inline_dpi: int = INLINE_DPI, save_dpi: int = SAVE_DPI) -> None:
    """Apply the study-wide Matplotlib theme.

    I keep the rcParams restrained because these plots need to survive both
    notebook inspection and manuscript export. Axes are readable, top/right
    spines are removed, grids are light, and saved figures use 220 dpi.

    Parameters
    ----------
    inline_dpi:
        Figure resolution used by interactive notebook backends, in dots/inch.
    save_dpi:
        Default resolution used by ``savefig``, in dots/inch.
    """

    mpl.rcParams.update(
        {
            "figure.dpi": int(inline_dpi),
            "savefig.dpi": int(save_dpi),
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "image.cmap": INTENSITY_CMAP,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL_PALETTE),
            "figure.constrained_layout.use": True,
        }
    )


def display_scale(
    values: np.ndarray,
    *,
    gamma: float = 0.45,
    normalise: bool = True,
    clip_negative: bool = True,
) -> np.ndarray:
    """Return an explicit gamma-scaled display array.

    I use this for intensity-like images only. The physical data remain
    unchanged; the returned array is just the documented display transform
    ``I_display = (I / I_max)**gamma`` when ``normalise=True``.

    Parameters
    ----------
    values:
        Scalar image or volume slice. Use SI-derived physical arrays upstream.
    gamma:
        Dimensionless display exponent. Must be positive.
    normalise:
        If true, divide by the finite maximum before applying gamma.
    clip_negative:
        If true, set negative values to zero before the exponent. This matches
        intensity displays, where negative values indicate a bug or noise floor.

    Returns
    -------
    numpy.ndarray
        Dimensionless display array.
    """

    if gamma <= 0:
        raise ValueError("display gamma must be positive.")
    arr = np.asarray(values, dtype=float)
    out = np.array(arr, dtype=float, copy=True)
    finite = np.isfinite(out)
    out[~finite] = 0.0
    if clip_negative:
        out = np.maximum(out, 0.0)
    if normalise:
        peak = float(np.max(out)) if out.size else 0.0
        if peak > 0.0:
            out = out / peak
    return out**float(gamma)


def symmetric_limits(values: np.ndarray, *, percentile: float = 99.5) -> tuple[float, float]:
    """Return symmetric color limits for a signed quantity.

    I center Stokes/deviation/sensitivity maps on zero so the color scale does
    not imply a false offset. The percentile guard prevents one bad pixel from
    hiding the meaningful structure.
    """

    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (-1.0, 1.0)
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    if limit <= 0.0:
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    if limit <= 0.0:
        limit = 1.0
    return (-limit, limit)


def intensity_image_kwargs(*, gamma: float = 0.45, colorbar_label: str = "display intensity [a.u.]") -> dict[str, Any]:
    """Return standard image options for non-negative intensity displays."""

    return {
        "cmap": INTENSITY_CMAP,
        "gamma": float(gamma),
        "colorbar_label": colorbar_label,
    }


def signed_image_kwargs(values: np.ndarray, *, percentile: float = 99.5, colorbar_label: str = "signed value [a.u.]") -> dict[str, Any]:
    """Return standard image options for signed maps centered on zero."""

    vmin, vmax = symmetric_limits(values, percentile=percentile)
    return {
        "cmap": SIGNED_CMAP,
        "vmin": vmin,
        "vmax": vmax,
        "colorbar_label": colorbar_label,
    }


def caption_path_for(figure_path: str | Path) -> Path:
    """Return the sidecar caption path for a saved figure."""

    path = Path(figure_path)
    return path.with_suffix(path.suffix + ".txt")


def captions_manifest_path(figure_path: str | Path) -> Path:
    """Return the default captions manifest beside a figure."""

    return Path(figure_path).parent / CAPTIONS_MANIFEST_NAME


def _write_caption_record(
    path: Path,
    caption: str,
    *,
    manifest_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write the shared caption sidecar and JSONL manifest entry."""

    cap_path = caption_path_for(path)
    cap_path.write_text(str(caption).strip() + "\n", encoding="utf-8")

    if manifest_path is None:
        manifest_path = captions_manifest_path(path)

    if manifest_path is not None:
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "artifact": str(path),
            "figure": str(path),
            "caption": str(cap_path),
            "caption_text": str(caption).strip(),
            "figure_name_pattern": FIGURE_NAME_PATTERN,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata or {}),
        }
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return cap_path


def save_figure(
    fig: plt.Figure,
    figure_path: str | Path,
    caption: str,
    *,
    dpi: int = SAVE_DPI,
    manifest_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a figure, its caption sidecar, and an optional JSONL manifest row.

    Parameters
    ----------
    fig:
        Matplotlib figure to save.
    figure_path:
        Output path. The repository convention is ``NN_topic_subject.png``.
    caption:
        One to three manuscript-ready sentences describing what is shown, the
        display scaling, and the units. I do not invent this text here because
        the figure-producing function knows the science.
    dpi:
        Saved figure resolution in dots/inch.
    manifest_path:
        Optional JSONL file that accumulates figure/caption records. If omitted,
        I write ``captions_manifest.jsonl`` next to the figure.
    metadata:
        Optional computed run metadata, such as config hashes or metric names.

    Returns
    -------
    pathlib.Path
        The saved figure path.
    """

    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=int(dpi))

    _write_caption_record(path, caption, manifest_path=manifest_path, metadata=metadata)

    return path


def save_plotly_html(
    fig: Any,
    html_path: str | Path,
    caption: str,
    *,
    include_plotlyjs: str | bool = "inline",
    manifest_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a self-contained Plotly HTML artifact with the study caption flow."""

    path = Path(html_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=include_plotlyjs, full_html=True)
    _write_caption_record(path, caption, manifest_path=manifest_path, metadata=metadata)
    return path


def save_figure_with_caption(
    fig: plt.Figure,
    figure_path: str | Path,
    caption: str,
    *,
    dpi: int = SAVE_DPI,
    manifest_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Backward-compatible wrapper around :func:`save_figure`."""

    return save_figure(
        fig,
        figure_path,
        caption,
        dpi=dpi,
        manifest_path=manifest_path,
        metadata=metadata,
    )


__all__ = [
    "CATEGORICAL_PALETTE",
    "CAPTIONS_MANIFEST_NAME",
    "CSV_NAME_PATTERN",
    "FIGURE_NAME_PATTERN",
    "INLINE_DPI",
    "INTENSITY_CMAP",
    "PHASE_CMAP",
    "SAVE_DPI",
    "SIGNED_CMAP",
    "apply_style",
    "artifact_name",
    "caption_path_for",
    "captions_manifest_path",
    "csv_name",
    "display_scale",
    "figure_name",
    "intensity_image_kwargs",
    "save_figure",
    "save_figure_with_caption",
    "save_plotly_html",
    "signed_image_kwargs",
    "symmetric_limits",
]
