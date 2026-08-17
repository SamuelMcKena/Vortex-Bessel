"""Make the focused measured / ideal / inverse-mask comparison requested by the user."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_triptych(stack_path, output_path, z_relative_mm=-10.0):
    stack_path = Path(stack_path)
    output_path = Path(output_path)
    data = np.load(stack_path)
    z = data["z_relative_mm"]
    index = int(np.argmin(np.abs(z - float(z_relative_mm))))
    selected_z = float(z[index])
    axis = data["x_um"]
    arrays = (data["measured"][index], data["ideal"][index],
              data["inverse_correction"][index])
    titles = ("LAB MEASURED", "IDEAL FORWARD MODEL",
              "IDEAL + INVERSE SAVED MASK")
    extent = [axis[0], axis[-1], axis[0], axis[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True)
    for ax, image, title in zip(axes, arrays, titles):
        shown = ax.imshow(image, origin="lower", extent=extent, cmap="inferno",
                          vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(0, color="white", lw=.4, alpha=.5)
        ax.axvline(0, color="white", lw=.4, alpha=.5)
        ax.set(title=title, xlabel="x (um)", ylabel="y (um)")
        ax.set_aspect("equal")
    fig.colorbar(shown, ax=axes, label="plane-normalized intensity", shrink=.84)
    fig.suptitle(f"Single-z back-check at relative z = {selected_z:g} mm\n"
                 "inverse case: ideal input x exp(-i x saved correction phase)",
                 fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return selected_z


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = (here / "outputs" / "slm_closed_loop_alignment" / "modal_q20" /
           "single_mask_inverse_forward_test")
    selected = make_triptych(out / "single_mask_forward_stacks.npz",
                             out / "single_z_minus10_measured_ideal_inverse.png",
                             z_relative_mm=-10.0)
    print(f"Wrote focused triptych for z={selected:g} mm")
