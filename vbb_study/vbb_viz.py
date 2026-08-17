"""Reusable visualization builders for publication figures.

Stage 4 moves the publication-facing views here so the notebooks can call one
consistent API for figure style, captions, and exported interactive artefacts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import vbb_metrics, vbb_style
from vbb_study.config import EPS as BT_EPS, PathKind, mm as BT_MM, um as BT_UM
from vbb_study.facade import core as _bt

try:
	import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled at call time if Plotly is absent.
	go = None


def _bundle_design(bundle: Mapping[str, Any]) -> Any:
	return bundle["result"]["design"]


def _bundle_case_id(bundle: Mapping[str, Any]) -> str:
	return str(bundle.get("case_id", bundle["result"].get("case_id", "publication_case")))


def _bundle_stage4_metrics(bundle: Mapping[str, Any], *, center_mode: str = "centroid") -> dict[str, Any]:
	design = _bundle_design(bundle)
	volume = bundle["result"]["volume"]
	per_z = vbb_metrics.per_z_metrics_from_volume(
		volume,
		ell=int(design.ell),
		kr_m_inv=float(design.kr_sample_m_inv),
		center_mode=center_mode,
	)
	total_power = np.asarray(per_z["total_power"], dtype=float)
	ring_power = np.asarray(per_z["ring_power"], dtype=float)
	per_z["ring_power_fraction"] = np.divide(
		ring_power,
		total_power + BT_EPS,
		out=np.zeros_like(ring_power),
		where=total_power > 0.0,
	)
	return per_z


def _zone_limits_um(bundle: Mapping[str, Any]) -> tuple[float, float]:
	metrics = bundle["result"]["metrics"]
	return (float(metrics.get("zone_start_um", np.nan)), float(metrics.get("zone_end_um", np.nan)))


def _sample_indices(z_um: np.ndarray, start_um: float, end_um: float, sample_count: int) -> np.ndarray:
	if np.isfinite(start_um) and np.isfinite(end_um) and end_um > start_um:
		candidate = np.flatnonzero((z_um >= start_um) & (z_um <= end_um))
	else:
		candidate = np.arange(len(z_um), dtype=int)
	if candidate.size == 0:
		candidate = np.arange(len(z_um), dtype=int)
	count = max(3, min(int(sample_count), int(candidate.size)))
	positions = np.linspace(0, candidate.size - 1, count)
	return np.unique(candidate[np.round(positions).astype(int)])


def _extent_xy_um(grid: Mapping[str, Any]) -> list[float]:
	x = np.asarray(grid["x"], dtype=float) / BT_UM
	y = np.asarray(grid.get("y", grid["x"]), dtype=float) / BT_UM
	return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _display_plane(plane: np.ndarray, *, gamma: float = 0.45) -> np.ndarray:
	return vbb_style.display_scale(np.asarray(plane, dtype=float), gamma=gamma, normalise=True)


def _metric_value(bundle: Mapping[str, Any], name: str) -> float:
	metrics = bundle["result"]["metrics"]
	if name in metrics:
		return float(metrics[name])
	if name == "ring_power_fraction_peak":
		stage4 = bundle.get("stage4_metrics")
		if stage4 is None:
			stage4 = _bundle_stage4_metrics(bundle)
		peak_index = int(bundle["result"]["volume"].get("peak_index", np.nanargmax(stage4["peak_in_plane"])))
		values = np.asarray(stage4["ring_power_fraction"], dtype=float)
		return float(values[peak_index])
	raise KeyError(f"Unsupported metric: {name}")


def _format_parameter_value(value: Any) -> str:
	if isinstance(value, bool):
		return "on" if value else "off"
	if isinstance(value, (int, np.integer)):
		return str(int(value))
	if isinstance(value, (float, np.floating)):
		return f"{float(value):.2f}".rstrip("0").rstrip(".")
	return str(value)


def _parameter_label(parameter: str) -> str:
	labels = {
		"ell": "topological charge l",
		"beam_radius_on_slm_mm": "input waist on SLM [mm, SLM plane]",
		"include_active_aperture": "active aperture",
	}
	return labels.get(str(parameter), str(parameter))


def _parameter_symbol(parameter: str) -> str:
	labels = {
		"ell": "l",
		"beam_radius_on_slm_mm": "w_in",
		"include_active_aperture": "aperture",
	}
	return labels.get(str(parameter), str(parameter))


def _sweep_config(base_config: Any, parameter: str, value: Any) -> Any:
	key = str(parameter).strip()
	if key == "ell":
		return replace(base_config, target=replace(base_config.target, ell=int(value)))
	if key == "beam_radius_on_slm_mm":
		return replace(base_config, laser=replace(base_config.laser, beam_radius_on_slm_m=float(value) * BT_MM))
	if key == "include_active_aperture":
		return replace(base_config, include_active_aperture=bool(value))
	raise ValueError(
		f"Unsupported sweep parameter: {parameter}. "
		"Stage 4 currently supports ell, beam_radius_on_slm_mm, and include_active_aperture."
	)


def _run_case_bundle(
	config: Any,
	*,
	preset: str,
	path: PathKind,
	case_id: str,
	sweep_parameter: str | None = None,
	sweep_value: Any = None,
) -> dict[str, Any]:
	result = _bt().run_case(config, preset=preset, path=path, case_id=case_id)
	bundle = {
		"config": config,
		"preset": preset,
		"case_id": case_id,
		"path": path,
		"result": result,
	}
	if sweep_parameter is not None:
		bundle["sweep_parameter"] = str(sweep_parameter)
		bundle["sweep_value"] = sweep_value
		bundle["sweep_label"] = _format_parameter_value(sweep_value)
	return bundle


def build_parameter_morph_bundles(
	base_config: Any,
	parameter: str,
	values: Sequence[Any],
	*,
	preset: str = "fast",
	path: PathKind = "ideal",
) -> list[dict[str, Any]]:
	"""Run a compact parameter sweep suitable for the morph-panel layout."""

	bundles: list[dict[str, Any]] = []
	for value in values:
		cfg = _sweep_config(base_config, parameter, value)
		label = f"{parameter}_{_format_parameter_value(value)}"
		bundle = _run_case_bundle(
			cfg,
			preset=preset,
			path=path,
			case_id=label,
			sweep_parameter=parameter,
			sweep_value=value,
		)
		bundle["stage4_metrics"] = _bundle_stage4_metrics(bundle)
		bundles.append(bundle)
	return bundles


def plot_radial_profile_waterfall(
	bundle: Mapping[str, Any],
	*,
	sample_count: int = 13,
	center_mode: str = "centroid",
	title: str | None = None,
) -> plt.Figure:
	"""Plot the ridgeline waterfall and the profile-vs-z image for one beam."""

	vbb_style.apply_style()
	stage4 = _bundle_stage4_metrics(bundle, center_mode=center_mode)
	z_um = np.asarray(stage4["z_m"], dtype=float) / BT_UM
	r_um = np.asarray(stage4["r_profile_m"], dtype=float) / BT_UM
	profiles = np.nan_to_num(np.asarray(stage4["radial_profile_smooth"], dtype=float), nan=0.0)
	ring_radius_um = np.asarray(stage4["ring_radius_m"], dtype=float) / BT_UM
	start_um, end_um = _zone_limits_um(bundle)
	selected = _sample_indices(z_um, start_um, end_um, sample_count)

	fig = plt.figure(figsize=(13.2, 6.1), constrained_layout=True)
	grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35])
	ax_ridge = fig.add_subplot(grid[0, 0])
	ax_img = fig.add_subplot(grid[0, 1], sharey=ax_ridge)

	cmap = mpl.colormaps[vbb_style.INTENSITY_CMAP]
	norm = mpl.colors.Normalize(vmin=float(z_um[0]), vmax=float(z_um[-1]))
	dz = np.diff(z_um[selected])
	amplitude = 0.75 * float(np.min(dz)) if dz.size else max(2.0, 0.12 * float(z_um[-1] - z_um[0] + BT_EPS))
	for idx in selected:
		base = float(z_um[idx])
		profile = profiles[idx]
		color = cmap(norm(base))
		ax_ridge.fill_between(r_um, base, base + amplitude * profile, color=color, alpha=0.18, linewidth=0.0)
		ax_ridge.plot(r_um, base + amplitude * profile, color=color, linewidth=1.4)

	if np.isfinite(start_um) and np.isfinite(end_um) and end_um > start_um:
		for axis in (ax_ridge, ax_img):
			axis.axhspan(start_um, end_um, color="#f2c14e", alpha=0.12, zorder=0)
			axis.axhline(start_um, color="#a56600", linestyle="--", linewidth=1.0)
			axis.axhline(end_um, color="#a56600", linestyle="--", linewidth=1.0)

	ax_ridge.plot(ring_radius_um, z_um, color=vbb_style.CATEGORICAL_PALETTE[5], linewidth=2.0, label="ring-radius track")
	ax_ridge.scatter(ring_radius_um[selected], z_um[selected], s=18, color=vbb_style.CATEGORICAL_PALETTE[5], zorder=3)
	ax_ridge.set_title("Ridgeline radial profiles")
	ax_ridge.set_xlabel("radius r [um, sample plane]")
	ax_ridge.set_ylabel("z [um, sample plane]")
	ax_ridge.legend(frameon=False, loc="upper right")
	fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_ridge, pad=0.02, label="z [um, sample plane]")

	image = ax_img.imshow(
		profiles,
		origin="lower",
		aspect="auto",
		extent=[float(r_um[0]), float(r_um[-1]), float(z_um[0]), float(z_um[-1])],
		cmap=vbb_style.INTENSITY_CMAP,
		vmin=0.0,
		vmax=1.0,
	)
	ax_img.plot(ring_radius_um, z_um, color=vbb_style.CATEGORICAL_PALETTE[5], linewidth=2.1)
	ax_img.set_title("Azimuthal-mean profile vs z")
	ax_img.set_xlabel("radius r [um, sample plane]")
	ax_img.set_ylabel("z [um, sample plane]")
	cbar = fig.colorbar(image, ax=ax_img, pad=0.02)
	cbar.set_label("normalised azimuthal mean intensity [a.u.]")

	fig.suptitle(
		title or f"{_bundle_case_id(bundle)} | radial-profile waterfall across the Bessel zone",
		fontsize=12,
	)
	return fig


def export_radial_profile_waterfall(
	bundle: Mapping[str, Any],
	figure_path: str | Path,
	*,
	sample_count: int = 13,
	center_mode: str = "centroid",
	manifest_path: str | Path | None = None,
) -> Path:
	fig = plot_radial_profile_waterfall(bundle, sample_count=sample_count, center_mode=center_mode)
	caption = (
		"Azimuthally averaged radial intensity profiles sampled across the propagation range for one vortex-Bessel case. "
		"Left: ridgeline waterfall with profiles coloured by z and the extracted ring-radius track overlaid; right: the same profiles as a radius-versus-z image. "
		"The dashed lines and shaded band mark the 50% Bessel-zone boundaries, and the colour scale reports locally normalised azimuthal mean intensity [a.u.]."
	)
	metadata = {
		"case_id": _bundle_case_id(bundle),
		"view": "radial_profile_waterfall",
		"sample_count": int(sample_count),
	}
	try:
		return vbb_style.save_figure(fig, figure_path, caption, manifest_path=manifest_path, metadata=metadata)
	finally:
		plt.close(fig)


def plot_annotated_bessel_region(
	bundle: Mapping[str, Any],
	*,
	center_mode: str = "centroid",
	title: str | None = None,
) -> plt.Figure:
	"""Plot a clean annotated x-z section with the axial side trace."""

	vbb_style.apply_style()
	result = bundle["result"]
	volume = result["volume"]
	stage4 = _bundle_stage4_metrics(bundle, center_mode=center_mode)
	z_um = np.asarray(stage4["z_m"], dtype=float) / BT_UM
	x_um = np.asarray(volume["crop_grid"]["x"], dtype=float) / BT_UM
	ring_radius_um = np.asarray(stage4["ring_radius_m"], dtype=float) / BT_UM
	peak_trace = np.asarray(volume["peak"], dtype=float)
	peak_trace = peak_trace / (float(np.max(peak_trace)) + BT_EPS)
	start_um, end_um = _zone_limits_um(bundle)
	z_max_um = float(z_um[0]) + float(_bundle_design(bundle).target_bessel_length_m / BT_UM)

	fig = plt.figure(figsize=(12.0, 5.3), constrained_layout=True)
	grid = fig.add_gridspec(1, 2, width_ratios=[1.85, 0.95])
	ax_img = fig.add_subplot(grid[0, 0])
	ax_side = fig.add_subplot(grid[0, 1])

	image = ax_img.imshow(
		_display_plane(np.asarray(volume["xz"], dtype=float)),
		origin="lower",
		aspect="auto",
		extent=[float(z_um[0]), float(z_um[-1]), float(x_um[0]), float(x_um[-1])],
		cmap=vbb_style.INTENSITY_CMAP,
		vmin=0.0,
		vmax=1.0,
	)
	if np.isfinite(start_um) and np.isfinite(end_um) and end_um > start_um:
		ax_img.axvspan(start_um, end_um, color="#f2c14e", alpha=0.12)
		ax_side.axhspan(start_um, end_um, color="#f2c14e", alpha=0.12)
		ax_img.axvline(start_um, color="#a56600", linestyle="--", linewidth=1.0)
		ax_img.axvline(end_um, color="#a56600", linestyle="--", linewidth=1.0)
	ax_img.axvline(z_max_um, color=vbb_style.CATEGORICAL_PALETTE[1], linestyle=":", linewidth=1.6)
	ax_img.text(z_max_um + 2.0, float(x_um[-1]) - 1.0, "z_max", color=vbb_style.CATEGORICAL_PALETTE[1], fontsize=9, va="top")
	ax_img.plot(z_um, ring_radius_um, color=vbb_style.CATEGORICAL_PALETTE[5], linewidth=2.0)
	ax_img.plot(z_um, -ring_radius_um, color=vbb_style.CATEGORICAL_PALETTE[5], linewidth=2.0)
	ax_img.set_title("Annotated x-z intensity section")
	ax_img.set_xlabel("z [um, sample plane]")
	ax_img.set_ylabel("x [um, sample plane]")
	cbar = fig.colorbar(image, ax=ax_img, pad=0.02)
	cbar.set_label("display intensity [a.u.]")

	ax_side.plot(peak_trace, z_um, color=vbb_style.CATEGORICAL_PALETTE[0], linewidth=2.0, label="peak in plane")
	ax_side.axvline(0.5, color="0.4", linewidth=1.0, linestyle="--")
	ax_side.set_ylim(float(z_um[0]), float(z_um[-1]))
	ax_side.set_title("Axial peak trace")
	ax_side.set_xlabel("normalised intensity [a.u.]")
	ax_side.set_ylabel("z [um, sample plane]")
	ax_side.legend(frameon=False, loc="lower right")

	fig.suptitle(title or f"{_bundle_case_id(bundle)} | shaded Bessel region and ring-radius track", fontsize=12)
	return fig


def export_annotated_bessel_region(
	bundle: Mapping[str, Any],
	figure_path: str | Path,
	*,
	center_mode: str = "centroid",
	manifest_path: str | Path | None = None,
) -> Path:
	fig = plot_annotated_bessel_region(bundle, center_mode=center_mode)
	caption = (
		"Sample x-z intensity section for one vortex-Bessel case with the non-diffracting zone highlighted. "
		"The dashed boundaries and shaded band mark the 50% Bessel zone, the dotted line marks the target z_max, and the cyan overlays trace the extracted ring radius in the positive and negative x cuts. "
		"The side panel shows the normalised peak-in-plane axial trace, and the image colourbar reports gamma-scaled display intensity [a.u.]."
	)
	metadata = {
		"case_id": _bundle_case_id(bundle),
		"view": "annotated_bessel_region",
	}
	try:
		return vbb_style.save_figure(fig, figure_path, caption, manifest_path=manifest_path, metadata=metadata)
	finally:
		plt.close(fig)


def plot_parameter_morph_panel(
	bundles: Sequence[Mapping[str, Any]],
	*,
	parameter: str,
	metric_specs: Sequence[tuple[str, str]] | None = None,
	title: str | None = None,
) -> plt.Figure:
	"""Show transverse beam morphing beside the corresponding metric curves."""

	if not bundles:
		raise ValueError("At least one bundle is required for a morph panel.")

	vbb_style.apply_style()
	specs = list(
		metric_specs
		or (
			("ring_radius_um", "ring radius [um, sample plane]"),
			("ring_width_um", "ring width [um, sample plane]"),
			("ring_power_fraction_peak", "peak-plane ring power fraction [-]"),
			("canonical_zone_um", "canonical FWHM zone [um, sample plane]"),
		)
	)
	values = [bundle.get("sweep_value") for bundle in bundles]
	x_positions = np.arange(len(bundles), dtype=float)
	xticklabels = [_format_parameter_value(v) for v in values]
	colors = [vbb_style.CATEGORICAL_PALETTE[i % len(vbb_style.CATEGORICAL_PALETTE)] for i in range(len(bundles))]

	fig = plt.figure(figsize=(15.4, 6.15), constrained_layout=True)
	outer = fig.add_gridspec(1, 2, width_ratios=[3.55, 1.45], wspace=0.06)
	left = outer[0, 0].subgridspec(2, len(bundles), height_ratios=[1.0, 0.065], hspace=0.05, wspace=0.05)
	right = outer[0, 1].subgridspec(len(specs), 1, hspace=0.16)

	image_axes = [fig.add_subplot(left[0, col]) for col in range(len(bundles))]
	colorbar_ax = fig.add_subplot(left[1, :])
	metric_axes = [fig.add_subplot(right[row, 0]) for row in range(len(specs))]

	image = None
	for col, (axis, bundle, color) in enumerate(zip(image_axes, bundles, colors)):
		plane = np.asarray(bundle["result"]["volume"]["planes"]["peak"], dtype=float)
		image = axis.imshow(
			_display_plane(plane),
			origin="lower",
			extent=_extent_xy_um(bundle["result"]["volume"]["crop_grid"]),
			cmap=vbb_style.INTENSITY_CMAP,
			vmin=0.0,
			vmax=1.0,
		)
		for spine in axis.spines.values():
			spine.set_visible(True)
			spine.set_linewidth(1.6)
			spine.set_edgecolor(color)
		axis.set_title(f"{_parameter_symbol(parameter)}={bundle.get('sweep_label', _format_parameter_value(values[col]))}", color=color)
		axis.set_xlabel("x [um, sample plane]")
		if col == 0:
			axis.set_ylabel("y [um, sample plane]")
		else:
			axis.set_yticklabels([])
		axis.grid(False)
	if image is not None:
		cbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
		cbar.set_label("display intensity [a.u.]")

	for axis, (metric_name, metric_label) in zip(metric_axes, specs):
		y = np.asarray([_metric_value(bundle, metric_name) for bundle in bundles], dtype=float)
		axis.plot(x_positions, y, color="0.25", linewidth=1.4)
		axis.scatter(x_positions, y, c=colors, s=58, zorder=3)
		axis.set_ylabel(metric_label)
		axis.set_xticks(x_positions, xticklabels)
		axis.grid(alpha=0.28)
	metric_axes[-1].set_xlabel(_parameter_label(parameter))
	for axis in metric_axes[:-1]:
		axis.tick_params(labelbottom=False)

	fig.suptitle(title or f"Parameter morph panel | {_parameter_label(parameter)} sweep", fontsize=12)
	return fig


def export_parameter_morph_panel(
	bundles: Sequence[Mapping[str, Any]],
	figure_path: str | Path,
	*,
	parameter: str,
	metric_specs: Sequence[tuple[str, str]] | None = None,
	manifest_path: str | Path | None = None,
) -> Path:
	fig = plot_parameter_morph_panel(bundles, parameter=parameter, metric_specs=metric_specs)
	caption = (
		f"Parameter-morph panel for the {_parameter_label(parameter)} sweep. "
		"The left montage shows gamma-scaled peak-plane transverse intensity for each sweep value using one consistent inferno colormap, while the right column reports the matched morphology metrics: ring radius, ring width, peak-plane ring-power fraction, and Bessel-zone length. "
		"This pairs the visual beam morph with the quantitative trend lines in one captioned figure."
	)
	metadata = {
		"parameter": str(parameter),
		"view": "parameter_morph_panel",
		"cases": [str(bundle.get("case_id", "case")) for bundle in bundles],
	}
	try:
		return vbb_style.save_figure(fig, figure_path, caption, manifest_path=manifest_path, metadata=metadata)
	finally:
		plt.close(fig)


def plot_interactive_radial_surface(
	bundle: Mapping[str, Any],
	*,
	center_mode: str = "centroid",
	z_margin_fraction: float = 0.12,
) -> Any:
	"""Build a Plotly surface over radius and z with a ring-power guide trace."""

	if go is None:
		raise RuntimeError("Plotly is not available in the current environment.")

	stage4 = _bundle_stage4_metrics(bundle, center_mode=center_mode)
	z_um = np.asarray(stage4["z_m"], dtype=float) / BT_UM
	r_um = np.asarray(stage4["r_profile_m"], dtype=float) / BT_UM
	profiles = np.nan_to_num(np.asarray(stage4["radial_profile_smooth"], dtype=float), nan=0.0)
	ring_radius_um = np.asarray(stage4["ring_radius_m"], dtype=float) / BT_UM
	ring_power_fraction = np.asarray(stage4["ring_power_fraction"], dtype=float)
	ring_power_norm = ring_power_fraction / (float(np.max(ring_power_fraction)) + BT_EPS)
	start_um, end_um = _zone_limits_um(bundle)

	if np.isfinite(start_um) and np.isfinite(end_um) and end_um > start_um:
		pad = z_margin_fraction * (end_um - start_um)
		mask = (z_um >= start_um - pad) & (z_um <= end_um + pad)
	else:
		mask = np.ones_like(z_um, dtype=bool)

	z_sel = z_um[mask]
	profiles_sel = profiles[mask]
	ring_sel = ring_radius_um[mask]
	ring_power_sel = ring_power_norm[mask]
	zz, rr = np.meshgrid(z_sel, r_um, indexing="ij")

	fig = go.Figure()
	fig.add_trace(
		go.Surface(
			x=zz,
			y=rr,
			z=profiles_sel,
			surfacecolor=profiles_sel,
			colorscale="Inferno",
			cmin=0.0,
			cmax=1.0,
			showscale=True,
			colorbar={"title": "normalised azimuthal mean intensity [a.u.]"},
			name="profile surface",
			hovertemplate="z=%{x:.1f} um<br>r=%{y:.2f} um<br>I=%{z:.3f}<extra></extra>",
		)
	)
	fig.add_trace(
		go.Scatter3d(
			x=z_sel,
			y=ring_sel,
			z=ring_power_sel,
			mode="lines+markers",
			line={"color": "#56B4E9", "width": 6},
			marker={"size": 3.5, "color": ring_power_sel, "colorscale": "Inferno", "cmin": 0.0, "cmax": 1.0},
			name="ring-power ribbon",
			hovertemplate="z=%{x:.1f} um<br>ring radius=%{y:.2f} um<br>ring power=%{z:.3f}<extra></extra>",
		)
	)
	fig.update_layout(
		template="plotly_white",
		title=f"{_bundle_case_id(bundle)} | radial-profile surface through the Bessel zone",
		scene={
			"xaxis_title": "z [um, sample plane]",
			"yaxis_title": "radius r [um, sample plane]",
			"zaxis_title": "normalised intensity / ring-power height [a.u.]",
			"camera": {"eye": {"x": 1.55, "y": 1.25, "z": 0.9}},
		},
		margin={"l": 0, "r": 0, "t": 60, "b": 0},
	)
	return fig


def export_interactive_radial_surface(
	bundle: Mapping[str, Any],
	html_path: str | Path,
	*,
	center_mode: str = "centroid",
	manifest_path: str | Path | None = None,
) -> Path:
	fig = plot_interactive_radial_surface(bundle, center_mode=center_mode)
	caption = (
		"Interactive 3D surface of the azimuthally averaged radial intensity profile over radius and propagation distance. "
		"The inferno surface shows the locally normalised azimuthal mean intensity, while the overlaid ribbon traces the extracted ring radius with height set by the normalised ring-power fraction through the Bessel zone. "
		"Open the self-contained HTML in any browser to rotate, zoom, and inspect the beam evolution."
	)
	metadata = {
		"case_id": _bundle_case_id(bundle),
		"view": "interactive_radial_surface",
		"html_open_hint": "Open the HTML file directly in a browser; Plotly JS is embedded inline.",
	}
	return vbb_style.save_plotly_html(fig, html_path, caption, manifest_path=manifest_path, metadata=metadata)


__all__ = [
	"build_parameter_morph_bundles",
	"export_annotated_bessel_region",
	"export_interactive_radial_surface",
	"export_parameter_morph_panel",
	"export_radial_profile_waterfall",
	"plot_annotated_bessel_region",
	"plot_interactive_radial_surface",
	"plot_parameter_morph_panel",
	"plot_radial_profile_waterfall",
]
