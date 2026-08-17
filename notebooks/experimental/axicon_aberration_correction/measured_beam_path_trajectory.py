"""High-resolution sensor-coordinate trajectory for the measured z scan.

The modal morphology stacks are deliberately recentered plane-by-plane.  This
report uses the stored full-sensor centre estimates instead, preserving measured
beam pointing.  It does not invent a corrected absolute trajectory because the
current correction model contains no calibrated camera-to-SLM steering map.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_measured_beam_path(metrics_csv, output_dir, *, pixel_pitch_um=5.5):
    metrics_csv, output_dir = Path(metrics_csv), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(metrics_csv).sort_values("z_rel_mm")
    z = data.z_rel_mm.to_numpy(float)
    x_sensor_um = data.cx_px_sensor.to_numpy(float) * pixel_pitch_um
    y_sensor_um = data.cy_px_sensor.to_numpy(float) * pixel_pitch_um
    x = x_sensor_um - x_sensor_um[0]
    y = y_sensor_um - y_sensor_um[0]
    displacement = np.hypot(x, y)
    fit_x = np.polyfit(z, x, 1)
    fit_y = np.polyfit(z, y, 1)
    x_line = np.polyval(fit_x, z)
    y_line = np.polyval(fit_y, z)
    residual_x = x-x_line
    residual_y = y-y_line

    path = pd.DataFrame({
        "z_relative_mm": z, "x_sensor_um": x_sensor_um,
        "y_sensor_um": y_sensor_um, "x_relative_um": x,
        "y_relative_um": y, "transverse_displacement_um": displacement,
        "linear_x_fit_um": x_line, "linear_y_fit_um": y_line,
        "x_fit_residual_um": residual_x, "y_fit_residual_um": residual_y,
    })
    path.to_csv(output_dir/"measured_beam_axis_trajectory.csv", index=False)

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    ax_x = fig.add_subplot(2, 2, 1)
    ax_y = fig.add_subplot(2, 2, 2)
    ax_xy = fig.add_subplot(2, 2, 3)
    ax_3d = fig.add_subplot(2, 2, 4, projection="3d")
    ax_x.plot(z, x, "o-", label="measured sensor centre")
    ax_x.plot(z, x_line, "--", label=f"linear slope {fit_x[0]:.2f} um/mm")
    ax_y.plot(z, y, "o-", label="measured sensor centre")
    ax_y.plot(z, y_line, "--", label=f"linear slope {fit_y[0]:.2f} um/mm")
    for ax, label in ((ax_x, "x displacement (um)"),
                      (ax_y, "y displacement (um)")):
        ax.set(xlabel="relative z (mm)", ylabel=label)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    points = ax_xy.scatter(x, y, c=z, cmap="viridis", s=55, zorder=3)
    ax_xy.plot(x, y, color="0.45", lw=1)
    ax_xy.scatter([0], [0], marker="*", s=170, color="red", label="z=-17 mm reference")
    ax_xy.set(xlabel="relative x (um)", ylabel="relative y (um)",
              title="Measured transverse sensor path")
    ax_xy.set_aspect("equal", adjustable="datalim"); ax_xy.grid(alpha=.25)
    ax_xy.legend(fontsize=8)
    fig.colorbar(points, ax=ax_xy, label="relative z (mm)")

    ax_3d.plot(x, y, z, "o-", color="#0072B2", lw=2, label="measured sensor centre")
    ax_3d.plot(np.zeros_like(z), np.zeros_like(z), z, "--", color="#D55E00",
               lw=1.5, label="straight reference axis (not corrected prediction)")
    ax_3d.set(xlabel="relative x (um)", ylabel="relative y (um)",
              zlabel="relative z (mm)", title="Measured 3D beam-axis trajectory")
    ax_3d.set_box_aspect((1.25, 1, 1.8)); ax_3d.view_init(elev=23, azim=-55)
    ax_3d.legend(fontsize=8)
    fig.suptitle("Measured beam path from full-sensor centre coordinates\n"
                 "separate from recentered modal-morphology volumes", fontsize=15)
    fig.savefig(output_dir/"measured_beam_axis_trajectory.png", dpi=400,
                bbox_inches="tight")
    fig.savefig(output_dir/"measured_beam_axis_trajectory.pdf", dpi=400,
                bbox_inches="tight")
    plt.close(fig)

    result = {
        "z_range_mm": [float(z[0]), float(z[-1])],
        "net_x_displacement_um": float(x[-1]-x[0]),
        "net_y_displacement_um": float(y[-1]-y[0]),
        "net_transverse_displacement_um": float(np.hypot(x[-1]-x[0], y[-1]-y[0])),
        "linear_x_slope_um_per_mm": float(fit_x[0]),
        "linear_y_slope_um_per_mm": float(fit_y[0]),
        "linear_axis_tilt_mrad": float(np.hypot(fit_x[0], fit_y[0])),
        "x_linear_fit_rms_residual_um": float(np.sqrt(np.mean(residual_x**2))),
        "y_linear_fit_rms_residual_um": float(np.sqrt(np.mean(residual_y**2))),
        "scope": "Sensor-centre estimate may include morphology-driven centroid motion; corrected absolute path is unavailable without camera-to-SLM steering calibration.",
    }
    (output_dir/"measured_beam_axis_trajectory_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return path, result


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = (here/"outputs"/"slm_closed_loop_alignment"/"modal_q20"/
           "comprehensive_error_validation")
    _, summary = write_measured_beam_path(
        here/"bessel_zscan_metrics.csv", out)
    print(json.dumps(summary, indent=2))
