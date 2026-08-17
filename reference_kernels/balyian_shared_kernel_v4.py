"""Shared scalar Bessel / vortex-Bessel tools split out from the design-sweep notebook.

This module keeps the original propagation engine and metric conventions, then adds
small workflow helpers for the split-notebook research package.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special as sp
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable, Union

# -----------------------------------------------------------------------------
# Research references used in the audit-driven fixes
# -----------------------------------------------------------------------------
# 1) Matsushima, K. & Shimobaba, T., Optics Express 17, 19662-19673 (2009),
#    doi:10.1364/OE.17.019662.
# 2) HOLOEYE PLUTO-2.1 / NIR-family product pages, used as the realism-device
#    baseline: 8 µm pitch, 93% fill factor, 15.36 × 8.64 mm active area,
#    and 8-bit addressing.
# 3) NumPy fftfreq / fftshift conventions for frequency-grid comments.
# -----------------------------------------------------------------------------


# Units
m = 1.0
cm = 1e-2 * m
mm = 1e-3 * m
um = 1e-6 * m
nm = 1e-9 * m


@dataclass
class GridSpec:
    N: int = 4096
    L: float = 5.0 * mm

    @property
    def dx(self) -> float:
        return 2 * self.L / self.N


@dataclass
class BeamSpec:
    wavelength: float = 532 * nm
    n_medium: float = 1.0
    w0: float = 2.0 * mm
    ell: int = 3


@dataclass
class AxiconSpec:
    n_axicon: float = 1.5
    gamma_deg: float = 1.0
    kr_mode: str = "tan"  # "tan" | "small_angle"


@dataclass
class MaskSpec:
    signum_pi_flip: bool = False


@dataclass
class SLMRealism:
    quantize_bits: Optional[int] = None
    pixel_pitch: Optional[float] = None
    carrier_lpmm: float = 0.0
    tilt_mrad: Tuple[float, float] = (0.0, 0.0)
    active_size: Optional[Tuple[float, float]] = None
    fill_factor: Optional[float] = None


@dataclass
class SimSpec:
    include_evanescent: bool = True
    alpha_np_per_m: float = 0.0


@dataclass
class PathScanSpec:
    s_i: float = 0.0
    s_f: float = 320 * mm
    ns: int = 180
    x_crop_mm: float = 1.0


@dataclass
class CameraProbe:
    name: str
    z: float


@dataclass
class BenchGeom:
    laser: Tuple[float, float] = (-180.0, -20.0)
    be: Tuple[float, float] = (-135.0, -8.0)
    hwp: Tuple[float, float] = (-95.0, 4.0)
    bs: Tuple[float, float] = (-25.0, 18.0)
    slm: Tuple[float, float] = (-110.0, 95.0)
    cam: Tuple[float, float] = (60.0, -30.0)


@dataclass
class Grid:
    N: int
    L: float
    dx: float
    x: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    R: np.ndarray
    PHI: np.ndarray
    FX: np.ndarray
    FY: np.ndarray

    @property
    def extent_mm(self) -> List[float]:
        return [
            self.x[0] / mm,
            (self.x[-1] + self.dx) / mm,
            self.x[0] / mm,
            (self.x[-1] + self.dx) / mm,
        ]


@dataclass
class ScalarField:
    U: np.ndarray
    wavelength: float
    n_medium: float
    grid: Grid

    def copy(self) -> "ScalarField":
        return ScalarField(self.U.copy(), self.wavelength, self.n_medium, self.grid)


def make_grid(gs: GridSpec) -> Grid:
    dx = 2 * gs.L / gs.N
    x = -gs.L + np.arange(gs.N) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X ** 2 + Y ** 2)
    PHI = np.arctan2(Y, X)

    fx = np.fft.fftshift(np.fft.fftfreq(gs.N, d=dx))
    FX, FY = np.meshgrid(fx, fx)

    return Grid(
        N=gs.N,
        L=gs.L,
        dx=dx,
        x=x,
        X=X,
        Y=Y,
        R=R,
        PHI=PHI,
        FX=FX,
        FY=FY,
    )


def transfer_function(
    wavelength: float,
    FX: np.ndarray,
    FY: np.ndarray,
    z: float,
    include_evanescent: bool = True,
) -> np.ndarray:
    arg = (2 * math.pi) ** 2 * ((1.0 / wavelength) ** 2 - FX ** 2 - FY ** 2)
    tmp = np.sqrt(np.abs(arg))
    kz = np.where(arg >= 0, tmp, 1j * tmp) if include_evanescent else np.where(arg >= 0, tmp, 0.0)
    return np.exp(1j * kz * z)


def fft2_shifted(U: np.ndarray) -> np.ndarray:
    return np.fft.fft2(np.fft.fftshift(U))


def propagate_ASM(U: np.ndarray, H: np.ndarray) -> np.ndarray:
    A = np.fft.fft2(np.fft.fftshift(U))
    Uz = np.fft.ifftshift(np.fft.ifft2(A * np.fft.fftshift(H)))
    return Uz


def propagate_from_A0(A0: np.ndarray, H: np.ndarray) -> np.ndarray:
    return np.fft.ifftshift(np.fft.ifft2(A0 * np.fft.fftshift(H)))


def build_propagator(field: ScalarField, sim: SimSpec) -> Callable[[float], ScalarField]:
    """
    Build an angular-spectrum propagator.

    Sampling note:
    We optionally apply the band-limited angular-spectrum clipping advocated by
    Matsushima & Shimobaba, Opt. Express 17, 19662-19673 (2009),
    doi:10.1364/OE.17.019662, to suppress transfer-function aliasing when high
    spatial frequencies approach the numerical Nyquist limit.
    """
    grid = field.grid
    A0 = fft2_shifted(field.U)

    arg = (2 * math.pi) ** 2 * ((1.0 / field.wavelength) ** 2 - grid.FX ** 2 - grid.FY ** 2)
    tmp = np.sqrt(np.abs(arg))
    kz = np.where(arg >= 0, tmp, 1j * tmp) if sim.include_evanescent else np.where(arg >= 0, tmp, 0.0)

    # NumPy fftfreq returns frequency bins in cycles / unit spacing, so the FX/FY
    # bin spacing is Δu = 1 / (N dx).
    du = 1.0 / (grid.N * grid.dx)

    def bandlimit_mask(z: float) -> np.ndarray:
        lam = float(field.wavelength)
        zz = float(abs(z))
        u_lim = 1.0 / (lam * math.sqrt((2.0 * du * zz) ** 2 + 1.0))
        return ((np.abs(grid.FX) <= u_lim) & (np.abs(grid.FY) <= u_lim)).astype(float)

    def propagate(z: float) -> ScalarField:
        z = float(z)
        H = np.exp(1j * kz * z)
        if getattr(sim, "bandlimit_asm", True):
            H = H * bandlimit_mask(z)
        Uz = propagate_from_A0(A0, H)
        if sim.alpha_np_per_m > 0.0:
            Uz = Uz * np.exp(-0.5 * sim.alpha_np_per_m * z)
        return ScalarField(U=Uz, wavelength=field.wavelength, n_medium=field.n_medium, grid=grid)

    return propagate

def compute_kr(k: float, n_axicon: float, n_medium: float, gamma_rad: float, mode: str = "tan") -> float:
    if mode == "small_angle":
        return k * (n_axicon - n_medium) * gamma_rad
    return k * (n_axicon - n_medium) * np.tan(gamma_rad)


def zmax_baliyan(w0: float, k: float, kr: float) -> float:
    return w0 * k / kr


def core_radius_estimate(kr: float) -> float:
    return 2.405 / kr


def gaussian_input(R: np.ndarray, w0: float) -> np.ndarray:
    return np.exp(-(R ** 2) / (w0 ** 2))


def build_beta_eq6(
    R: np.ndarray,
    PHI: np.ndarray,
    k: float,
    n_axicon: float,
    n_medium: float,
    gamma_rad: float,
    ell: int,
    signum_pi_flip: bool = False,
    kr_mode: str = "tan",
) -> np.ndarray:
    beta = -k * (n_axicon - n_medium) * R * np.tan(gamma_rad) + ell * PHI

    if signum_pi_flip:
        kr = compute_kr(k, n_axicon, n_medium, gamma_rad, mode=kr_mode)
        J = sp.jv(int(abs(ell)), kr * R)
        beta = beta + np.where(J < 0, np.pi, 0.0)

    return np.mod(beta, 2 * np.pi)


def quantize_phase(phi: np.ndarray, bits: int) -> np.ndarray:
    """
    Quantise wrapped phase to 2**bits discrete levels in [0, 2π).

    Reference note:
    HOLOEYE PLUTO-2.1 device pages list 8-bit addressing (256 phase levels)
    for the PLUTO family used as the baseline realism model in this project.
    """
    b = int(bits)
    if b <= 0:
        raise ValueError("quantize_phase: bits must be positive")
    levels = 1 << b
    wrapped = np.mod(phi, 2 * np.pi)
    idx = (np.floor(wrapped / (2 * np.pi) * levels + 0.5).astype(np.int64)) % levels
    return idx.astype(float) * (2 * np.pi) / float(levels)

def pixelate_phase(phi: np.ndarray, dx: float, pixel_pitch: Optional[float]) -> np.ndarray:
    """
    Convert a continuous phase map to an SLM pixelated phase map.

    The phase is block-averaged using a circular mean over one simulated SLM
    pixel. If pixel_pitch / dx is not close to an integer, we warn because the
    numerical SLM pitch is no longer faithfully represented.
    """
    if pixel_pitch is None:
        return phi

    dx = float(dx)
    p = float(pixel_pitch)
    if p <= dx:
        return phi

    ratio = p / dx
    b = int(round(ratio))
    if b <= 1:
        return phi

    if abs(ratio - b) > 0.01:
        print(f"[pixelate_phase] WARNING: pixel_pitch/dx = {ratio:.4f} not near integer; using b={b} so p_eff={b*dx:.3e} m")

    N = phi.shape[0]
    Nb = int(np.ceil(N / b) * b)
    pad = Nb - N
    if pad > 0:
        ph = np.pad(phi, ((0, pad), (0, pad)), mode="edge")
    else:
        ph = phi

    U = np.exp(1j * ph)
    Uc = U.reshape(Nb // b, b, Nb // b, b).mean(axis=(1, 3))
    Uu = np.repeat(np.repeat(Uc, b, axis=0), b, axis=1)
    return np.angle(Uu[:N, :N])

def _fill_factor_1d_cell_average(x: np.ndarray, dx: float, pixel_pitch: float, fill_factor: float) -> np.ndarray:
    """
    Anti-aliased 1D pixel fill function.

    Each simulation sample represents the cell [x-dx/2, x+dx/2]. We return the
    fraction of that cell covered by the active LC region of one SLM pixel.

    Why this matters here:
    the HOLOEYE PLUTO-2.1 family used as the realism baseline has 8 µm pitch and
    93% fill factor, so the inactive border is sub-micron. With dx = 4 µm a
    point-sampled 0/1 mask aliases that tiny gap into unrealistically thick dead
    lines. Cell-averaging preserves the effective fill factor instead.
    """
    x = np.asarray(x, float)
    dx = float(dx)
    p = float(pixel_pitch)
    ff = float(fill_factor)

    if p <= 0.0 or dx <= 0.0 or ff >= 1.0:
        return np.ones_like(x, dtype=float)

    a = np.sqrt(ff) * p
    start = 0.5 * (p - a)
    end = start + a

    x0 = x - 0.5 * dx
    x1 = x + 0.5 * dx
    u0 = np.mod(x0, p)
    u1 = np.mod(x1, p)

    def overlap(u_lo, u_hi):
        return np.maximum(0.0, np.minimum(u_hi, end) - np.maximum(u_lo, start))

    wraps = u1 < u0
    ov = overlap(u0, u1)
    ov_wrap = overlap(u0, np.full_like(u0, p)) + overlap(np.zeros_like(u0), u1)
    ov = np.where(wraps, ov_wrap, ov)
    return ov / dx


def slm_fill_factor_mask(
    X: np.ndarray,
    Y: np.ndarray,
    pixel_pitch: Optional[float],
    fill_factor: Optional[float],
    dx: Optional[float] = None,
    antialias: bool = True,
) -> np.ndarray:
    """
    Square-pixel fill-factor mask on the SLM plane.

    If antialias=True, compute a cell-averaged mask using dx. This is the right
    model whenever the real inter-pixel gap is much smaller than the numerical
    sample pitch.
    """
    if pixel_pitch is None or fill_factor is None or float(fill_factor) >= 1.0:
        return np.ones_like(X, dtype=float)

    p = float(pixel_pitch)
    ff = float(fill_factor)

    if (dx is None) or (not antialias):
        a = np.sqrt(ff) * p
        xmod = np.mod(X + 0.5 * p, p) - 0.5 * p
        ymod = np.mod(Y + 0.5 * p, p) - 0.5 * p
        return ((np.abs(xmod) <= 0.5 * a) & (np.abs(ymod) <= 0.5 * a)).astype(float)

    dx = float(dx)
    x1d = np.asarray(X[0, :], float)
    y1d = np.asarray(Y[:, 0], float)
    mx = _fill_factor_1d_cell_average(x1d, dx, p, ff)
    my = _fill_factor_1d_cell_average(y1d, dx, p, ff)
    return my[:, None] * mx[None, :]

class OpticalElement:
    def __init__(self, name: str):
        self.name = name

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        return field, {}


class SLMPhase(OpticalElement):
    def __init__(
        self,
        beam: BeamSpec,
        axicon: AxiconSpec,
        mask: MaskSpec,
        realism: SLMRealism,
        name: str = "SLM",
    ):
        super().__init__(name=name)
        self.beam = beam
        self.axicon = axicon
        self.mask = mask
        self.realism = realism

    def phase_pattern(self, field: ScalarField) -> np.ndarray:
        grid = field.grid
        k = 2 * np.pi / field.wavelength
        gamma = np.deg2rad(self.axicon.gamma_deg)

        beta = build_beta_eq6(
            grid.R,
            grid.PHI,
            k,
            self.axicon.n_axicon,
            self.beam.n_medium,
            gamma,
            self.beam.ell,
            signum_pi_flip=self.mask.signum_pi_flip,
            kr_mode=self.axicon.kr_mode,
        )

        phi = beta

        tx_mrad, ty_mrad = self.realism.tilt_mrad
        if tx_mrad != 0.0 or ty_mrad != 0.0:
            phi = phi + k * ((tx_mrad * 1e-3) * grid.X + (ty_mrad * 1e-3) * grid.Y)

        if self.realism.carrier_lpmm != 0.0:
            f = self.realism.carrier_lpmm * 1e3
            phi = phi + 2 * np.pi * f * grid.X

        phi = pixelate_phase(phi, grid.dx, self.realism.pixel_pitch)

        if self.realism.quantize_bits is not None:
            phi = quantize_phase(phi, self.realism.quantize_bits)

        return phi

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        grid = field.grid
        phi = self.phase_pattern(field)

        U = field.U * np.exp(1j * phi)

        meta = {"phase": phi}
        if self.realism.active_size is not None:
            sx, sy = self.realism.active_size
            aperture = ((np.abs(grid.X) <= 0.5 * sx) & (np.abs(grid.Y) <= 0.5 * sy)).astype(float)
            U = U * aperture
            meta["active_aperture"] = aperture

        if self.realism.pixel_pitch is not None and self.realism.fill_factor is not None and self.realism.fill_factor < 1.0:
            fill_mask = slm_fill_factor_mask(
                grid.X,
                grid.Y,
                self.realism.pixel_pitch,
                self.realism.fill_factor,
                dx=grid.dx,
                antialias=True,
            )
            # Device-model note:
            # at the current grid pitch we use the anti-aliased *effective*
            # fill-factor mask rather than an explicitly resolved sub-micron gap.
            U = U * fill_mask
            meta["fill_mask"] = fill_mask
            meta["fill_mask_mean"] = float(np.mean(fill_mask))
            meta["fill_factor_requested"] = float(self.realism.fill_factor)

        out = ScalarField(U=U, wavelength=field.wavelength, n_medium=field.n_medium, grid=grid)
        return out, meta


class FreeSpace(OpticalElement):
    def __init__(self, z: float, sim: SimSpec, name: str = "FreeSpace"):
        super().__init__(name=name)
        self.z = z
        self.sim = sim

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        H = transfer_function(
            field.wavelength,
            field.grid.FX,
            field.grid.FY,
            self.z,
            include_evanescent=self.sim.include_evanescent,
        )
        Uz = propagate_ASM(field.U, H)
        if self.sim.alpha_np_per_m > 0.0:
            Uz = Uz * np.exp(-0.5 * self.sim.alpha_np_per_m * self.z)
        out = ScalarField(U=Uz, wavelength=field.wavelength, n_medium=field.n_medium, grid=field.grid)
        return out, {}


class Lens(OpticalElement):
    def __init__(self, f: float, name: str = "Lens"):
        super().__init__(name=name)
        self.f = f

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        k = 2 * np.pi / field.wavelength
        quad = np.exp(-1j * k * (field.grid.X ** 2 + field.grid.Y ** 2) / (2 * self.f))
        out = ScalarField(U=field.U * quad, wavelength=field.wavelength, n_medium=field.n_medium, grid=field.grid)
        return out, {}


class Aperture(OpticalElement):
    def __init__(
        self,
        radius: Optional[float] = None,
        rect: Optional[Tuple[float, float]] = None,
        name: str = "Aperture",
    ):
        super().__init__(name=name)
        self.radius = radius
        self.rect = rect

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        mask = np.ones_like(field.U, dtype=float)
        if self.radius is not None:
            mask = mask * (field.grid.R <= self.radius).astype(float)
        if self.rect is not None:
            sx, sy = self.rect
            mask = mask * ((np.abs(field.grid.X) <= 0.5 * sx) & (np.abs(field.grid.Y) <= 0.5 * sy)).astype(float)
        out = ScalarField(U=field.U * mask, wavelength=field.wavelength, n_medium=field.n_medium, grid=field.grid)
        return out, {"mask": mask}


class CameraElement(OpticalElement):
    def __init__(self, name: str):
        super().__init__(name=name)

    def apply(self, field: ScalarField) -> Tuple[ScalarField, Dict[str, np.ndarray]]:
        return field, {"capture": np.array([1.0])}


class OpticalBench:
    def __init__(self):
        self.elements: List[OpticalElement] = []
        self.camera_registry: List[str] = []

    def add(self, element: OpticalElement) -> None:
        self.elements.append(element)

    def run(self, input_field: ScalarField) -> Tuple[ScalarField, Dict[str, ScalarField], Dict[str, Dict[str, np.ndarray]]]:
        field = input_field.copy()
        captures: Dict[str, ScalarField] = {}
        meta: Dict[str, Dict[str, np.ndarray]] = {}

        for element in self.elements:
            field, element_meta = element.apply(field)
            meta[element.name] = element_meta
            if isinstance(element, CameraElement):
                captures[element.name] = field.copy()
                self.camera_registry.append(element.name)

        return field, captures, meta


def plot_xy(
    I: np.ndarray,
    extent_mm: List[float],
    title: str,
    lim_mm: float = 1.0,
    cmap: str = "hot",
    normalize_to_self: bool = True,
) -> None:
    V = np.asarray(I, float)
    if normalize_to_self:
        V = V / (V.max() + 1e-15)

    plt.figure(figsize=(8.0, 7.0))
    plt.imshow(
        V,
        interpolation="spline36",
        aspect=1.0,
        extent=extent_mm,
        cmap=cmap,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.xlim(-lim_mm, lim_mm)
    plt.ylim(-lim_mm, lim_mm)
    plt.colorbar(label="I/Imax")
    plt.tight_layout()
    plt.show()


def montage_xy(
    images: List[np.ndarray],
    extent_mm: List[float],
    titles: List[str],
    lim_mm: float = 1.0,
    cmap: str = "hot",
    normalize_each: bool = True,
    show_axes: bool = True,
    title: Optional[str] = None,
) -> None:
    if len(images) == 0:
        return

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, img, t in zip(axes, images, titles):
        V = np.asarray(img, float)
        if normalize_each:
            V = V / (V.max() + 1e-15)

        im = ax.imshow(
            V,
            interpolation="spline36",
            aspect=1.0,
            extent=extent_mm,
            cmap=cmap,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(t)
        ax.set_xlim(-lim_mm, lim_mm)
        ax.set_ylim(-lim_mm, lim_mm)
        if show_axes:
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    if title is not None:
        fig.suptitle(title)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("I/Imax")
    plt.show()


def plot_xz(
    profile_xz: np.ndarray,
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    title: str,
    lim_mm: float = 1.0,
    cmap: str = "turbo",
    zmin_mm: Optional[float] = None,
    zmax_mm: Optional[float] = None,
    zmax_line_mm: Optional[float] = None,
    marks_mm: Optional[List[Tuple[str, float, str]]] = None,
) -> None:
    P = np.asarray(profile_xz, float)
    Pn = P / (P.max() + 1e-15)

    i0 = int(np.searchsorted(x_mm, -lim_mm, side="left"))
    i1 = int(np.searchsorted(x_mm, lim_mm, side="right"))

    if zmin_mm is None:
        j0 = 0
    else:
        j0 = int(np.searchsorted(z_mm, zmin_mm, side="left"))

    if zmax_mm is None:
        j1 = len(z_mm)
    else:
        j1 = int(np.searchsorted(z_mm, zmax_mm, side="right"))

    plt.figure(figsize=(10.5, 7.2))
    plt.imshow(
        Pn[i0:i1, j0:j1],
        interpolation="spline36",
        aspect="auto",
        cmap=cmap,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        extent=[z_mm[j0], z_mm[j1 - 1], x_mm[i0], x_mm[i1 - 1]],
    )

    if zmax_line_mm is not None and z_mm[j0] <= zmax_line_mm <= z_mm[j1 - 1]:
        plt.axvline(zmax_line_mm, color="white", linestyle="--", linewidth=1.8, label="Eq.(5) zmax")

    if marks_mm is not None:
        ymax = x_mm[i1 - 1]
        for name, zm, kind in marks_mm:
            if z_mm[j0] <= zm <= z_mm[j1 - 1]:
                col = "cyan" if kind == "probe" else "white"
                plt.axvline(zm, color=col, linestyle=":", linewidth=1.2)
                plt.text(zm, 0.95 * ymax, name, rotation=90, ha="right", va="top", fontsize=8, color=col)

    plt.xlabel("z (mm)")
    plt.ylabel("x (mm)")
    plt.title(title)
    plt.colorbar(label="I/Imax")
    plt.grid(alpha=0.15)
    if zmax_line_mm is not None:
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_onaxis_vs_distance(x: np.ndarray, s: np.ndarray, I_xs: np.ndarray, zmax: Optional[float] = None) -> None:
    x0 = int(np.argmin(np.abs(x)))
    on = I_xs[x0, :]
    on_n = on / (on.max() + 1e-15)

    plt.figure(figsize=(9.0, 4.8))
    plt.plot(s / mm, on_n, linewidth=2.0, color="tab:red", label="On-axis I/Imax")
    if zmax is not None:
        plt.axvline(zmax / mm, color="k", linestyle="--", linewidth=1.2, label="Eq.(5) zmax")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Propagation distance from SLM, z (mm)")
    plt.ylabel("On-axis normalized intensity")
    plt.title("On-axis intensity vs propagation (linear)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def auto_crop_mm_from_images(
    images: List[np.ndarray],
    x: np.ndarray,
    energy_frac: float = 0.997,
    pad: float = 1.15,
    min_mm: float = 0.25,
    max_mm: float = 1.2,
) -> float:
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    Rf = R.ravel()

    radii = []
    for I in images:
        W = np.maximum(I, 0).ravel()
        order = np.argsort(Rf)
        r_sorted = Rf[order]
        w_sorted = W[order]
        c = np.cumsum(w_sorted)
        if c[-1] <= 0:
            continue
        j = np.searchsorted(c, energy_frac * c[-1])
        j = min(j, len(r_sorted) - 1)
        radii.append(r_sorted[j])

    if len(radii) == 0:
        return 1.0

    crop_mm = (max(radii) * pad) / mm
    return float(np.clip(crop_mm, min_mm, max_mm))


def path_map(geom: BenchGeom, s_vals: np.ndarray) -> Tuple[np.ndarray, float, float]:
    slm = np.array(geom.slm, float)
    bs = np.array(geom.bs, float)
    cam = np.array(geom.cam, float)

    d1_mm = np.linalg.norm(bs - slm)
    d2_mm = np.linalg.norm(cam - bs)

    pts = []
    for s in np.atleast_1d(s_vals):
        smm = float(s / mm)
        if smm <= d1_mm:
            t = smm / d1_mm if d1_mm > 0 else 0.0
            p = slm + t * (bs - slm)
        else:
            t = (smm - d1_mm) / d2_mm if d2_mm > 0 else 0.0
            t = np.clip(t, 0.0, 1.0)
            p = bs + t * (cam - bs)
        pts.append(p)

    return np.asarray(pts), d1_mm * mm, (d1_mm + d2_mm) * mm


def draw_bench_layout(
    geom: BenchGeom,
    probes: Optional[List[CameraProbe]] = None,
    s_region: Optional[Tuple[float, float]] = None,
    title: str = "Optical bench top-down (x-z)",
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    laser = np.array(geom.laser, float)
    be = np.array(geom.be, float)
    hwp = np.array(geom.hwp, float)
    bs = np.array(geom.bs, float)
    slm = np.array(geom.slm, float)
    cam = np.array(geom.cam, float)

    route = np.vstack([laser, be, hwp, bs, slm, bs, cam])
    ax.plot(route[:, 0], route[:, 1], color="#58bf73", linewidth=4, alpha=0.95)

    ax.scatter([laser[0]], [laser[1]], s=560, marker="s", color="#a1a1a1", edgecolor="k", zorder=4)
    ax.scatter([be[0]], [be[1]], s=330, marker="o", color="#79c9ff", edgecolor="k", zorder=4)
    ax.scatter([hwp[0]], [hwp[1]], s=330, marker="o", color="#9fdcff", edgecolor="k", zorder=4)
    ax.scatter([bs[0]], [bs[1]], s=700, marker="D", color="#6ccfc6", edgecolor="k", zorder=4)
    ax.scatter([slm[0]], [slm[1]], s=700, marker="s", color="#efd56c", edgecolor="k", zorder=4)
    ax.scatter([cam[0]], [cam[1]], s=450, marker="o", color="#ddb66f", edgecolor="k", zorder=4)

    for name, p in [("LASER", laser), ("BE", be), ("HWP", hwp), ("BS", bs), ("SLM", slm), ("CMOS", cam)]:
        ax.text(p[0], p[1] + 9, name, ha="center", va="bottom", fontsize=10, fontweight="bold")

    if s_region is not None:
        s = np.linspace(s_region[0], s_region[1], 300)
        pts, _, _ = path_map(geom, s)
        ax.plot(pts[:, 0], pts[:, 1], color="magenta", linewidth=3, alpha=0.9, label="Simulated BG region")

    if probes is not None and len(probes) > 0:
        ps = np.array([p.z for p in probes])
        pts, _, _ = path_map(geom, ps)
        ax.scatter(pts[:, 0], pts[:, 1], s=80, color="white", edgecolor="k", zorder=5)
        for pr, pt in zip(probes, pts):
            ax.text(pt[0] + 4, pt[1] - 4, pr.name, fontsize=8)

    pad = 18.0
    ax.set_xlim(route[:, 0].min() - pad, route[:, 0].max() + pad)
    ax.set_ylim(route[:, 1].min() - pad, route[:, 1].max() + pad)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    if s_region is not None:
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def build_input_field(gs: GridSpec, beam: BeamSpec) -> ScalarField:
    grid = make_grid(gs)
    Uin = gaussian_input(grid.R, beam.w0)
    return ScalarField(U=Uin, wavelength=beam.wavelength, n_medium=beam.n_medium, grid=grid)


def print_sanity_block(
    field: ScalarField,
    beam: BeamSpec,
    axicon: AxiconSpec,
    realism: SLMRealism,
) -> Tuple[float, float, float]:
    k = 2 * np.pi / beam.wavelength
    gamma = np.deg2rad(axicon.gamma_deg)
    kr = compute_kr(k, axicon.n_axicon, beam.n_medium, gamma, mode=axicon.kr_mode)
    zmax = zmax_baliyan(beam.w0, k, kr)
    r0 = core_radius_estimate(kr)

    dx = field.grid.dx
    px_per_r0 = r0 / dx

    print("=== Scalar Baliyan sanity block ===")
    print(f"lambda = {beam.wavelength / nm:.1f} nm")
    print(f"n_axicon = {axicon.n_axicon:.4f}, n_medium = {beam.n_medium:.4f}, gamma = {axicon.gamma_deg:.4f} deg")
    print(f"k_r = {kr:.3e} 1/m")
    print(f"predicted core radius r0 ~= 2.405/k_r = {r0 / um:.2f} um")
    print(f"predicted zmax Eq.(5) = {zmax / mm:.2f} mm")
    print(f"grid: N={field.grid.N}, L={field.grid.L / mm:.2f} mm, dx={dx / um:.2f} um")
    print(f"sampling: r0/dx = {px_per_r0:.2f} pixels")

    if px_per_r0 < 8.0:
        print("WARNING: core under-resolved (<8 px). Increase N or reduce L.")

    if realism.active_size is not None:
        sx, sy = realism.active_size
        if 2 * field.grid.L < max(sx, sy):
            print("WARNING: computational window smaller than active SLM aperture.")

    return kr, zmax, r0


def run_scalar_pipeline(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
    scan: PathScanSpec,
    probes: List[CameraProbe],
    show_debug: bool = True,
) -> Dict[str, object]:
    field_in = build_input_field(gs, beam)

    bench = OpticalBench()
    bench.add(SLMPhase(beam=beam, axicon=axicon, mask=mask, realism=realism, name="SLM"))
    bench.add(CameraElement(name="After SLM"))

    field_after_slm, captures, meta = bench.run(field_in)
    phase_slm = meta["SLM"]["phase"]

    kr, zmax, r0 = print_sanity_block(field_after_slm, beam, axicon, realism) if show_debug else (None, None, None)
    if not show_debug:
        k = 2 * np.pi / beam.wavelength
        gamma = np.deg2rad(axicon.gamma_deg)
        kr = compute_kr(k, axicon.n_axicon, beam.n_medium, gamma, mode=axicon.kr_mode)
        zmax = zmax_baliyan(beam.w0, k, kr)
        r0 = core_radius_estimate(kr)

    propagator = build_propagator(field_after_slm, sim)

    s = np.linspace(scan.s_i, scan.s_f, scan.ns)
    xs = np.zeros((field_after_slm.grid.N, scan.ns), dtype=np.float32)
    y0 = field_after_slm.grid.N // 2

    for i, zi in enumerate(s):
        Iz = np.abs(propagator(float(zi)).U) ** 2
        xs[:, i] = Iz[y0, :].astype(np.float32)

    probe_images: List[np.ndarray] = []
    probe_titles: List[str] = []
    for pr in probes:
        Iz = np.abs(propagator(pr.z).U) ** 2
        probe_images.append(Iz.astype(np.float32))
        probe_titles.append(f"{pr.name} ({pr.z / mm:.1f} mm)")

    return {
        "field_after_slm": field_after_slm,
        "captures": captures,
        "phase_slm": phase_slm,
        "slm_meta": meta.get("SLM", {}),
        "propagator": propagator,
        "x": field_after_slm.grid.x,
        "s": s,
        "xs": xs,
        "grid": field_after_slm.grid,
        "probe_images": probe_images,
        "probe_titles": probe_titles,
        "kr": kr,
        "zmax": zmax,
        "r0": r0,
    }


def scalar_l_sweep(
    gs: GridSpec,
    beam_base: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
    ells: List[int],
    z_view: float,
) -> Tuple[List[np.ndarray], List[str], List[float], np.ndarray]:
    imgs: List[np.ndarray] = []
    titles: List[str] = []
    extent_mm: Optional[List[float]] = None
    x_ref: Optional[np.ndarray] = None

    for ell in ells:
        b = BeamSpec(
            wavelength=beam_base.wavelength,
            n_medium=beam_base.n_medium,
            w0=beam_base.w0,
            ell=int(ell),
        )
        out = run_scalar_pipeline(
            gs=gs,
            beam=b,
            axicon=axicon,
            mask=mask,
            realism=realism,
            sim=sim,
            scan=PathScanSpec(s_i=z_view, s_f=z_view, ns=1, x_crop_mm=1.0),
            probes=[CameraProbe(name="view", z=z_view)],
            show_debug=False,
        )
        imgs.append(out["probe_images"][0])
        titles.append(f"l={ell}")
        extent_mm = out["grid"].extent_mm
        x_ref = out["x"]

    return imgs, titles, extent_mm, x_ref


def radial_profile(I: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)

    dr = x[1] - x[0]
    edges = np.arange(0, R.max() + dr, dr)
    ind = np.digitize(R.ravel(), edges) - 1
    valid = (ind >= 0) & (ind < len(edges) - 1)

    sums = np.bincount(ind[valid], weights=I.ravel()[valid], minlength=len(edges) - 1)
    cnts = np.bincount(ind[valid], minlength=len(edges) - 1)
    prof = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof


def estimate_main_ring_radius(I: np.ndarray, x: np.ndarray, ell: int, kr: float) -> float:
    r, p = radial_profile(I, x)
    p = np.asarray(p, float)
    p = p / (p.max() + 1e-15)

    # Use Bessel-theory guidance: first intensity annulus near first J'_l zero.
    l_abs = int(abs(ell))
    if l_abs == 0:
        r_pred = 2.405 / kr
    else:
        r_pred = sp.jnp_zeros(l_abs, 1)[0] / kr

    r_lo = max(0.0, 0.55 * r_pred)
    r_hi = min(r.max(), 1.70 * r_pred)

    m = (r >= r_lo) & (r <= r_hi)
    if np.any(m):
        rr = r[m]
        pp = p[m]
        return rr[int(np.argmax(pp))]

    # Fallback window near center.
    m2 = (r >= 0.0) & (r <= min(r.max(), 0.9 * mm))
    if np.any(m2):
        rr2 = r[m2]
        pp2 = p[m2]
        return rr2[int(np.argmax(pp2))]

    return r[int(np.argmax(p))]


def paper_similarity_report(images: List[np.ndarray], ells: List[int], x: np.ndarray, kr: float) -> None:
    ring_mm = []
    center_ratio = []
    pred_mm = []

    for I, ell in zip(images, ells):
        rm = estimate_main_ring_radius(I, x, ell=ell, kr=kr) / mm
        ring_mm.append(rm)

        l_abs = int(abs(ell))
        if l_abs == 0:
            rp = 2.405 / kr
        else:
            rp = sp.jnp_zeros(l_abs, 1)[0] / kr
        pred_mm.append(rp / mm)

        c = I[I.shape[0] // 2, I.shape[1] // 2]
        center_ratio.append(float(c / (I.max() + 1e-15)))

    corr = np.corrcoef(np.asarray(ells, float), np.asarray(ring_mm, float))[0, 1]
    slope, intercept = np.polyfit(np.asarray(ells, float), np.asarray(ring_mm, float), 1)

    print("=== Scalar-vs-paper quick check (qualitative) ===")
    print("Expected trend: annulus radius increases with l.")
    print(f"Measured fit: r_mm ~= {slope:.4f} * l + {intercept:.4f}, corr={corr:.4f}")
    print("l | measured_mm | theory_mm | center_I/Imax")
    for l, rm, rp, cr in zip(ells, ring_mm, pred_mm, center_ratio):
        print(f"{l:2d} | {rm:10.4f} | {rp:9.4f} | {cr:10.4f}")

def onaxis_I_vs_z_from_propagator(
    propagator: Callable[[float], ScalarField],
    x: np.ndarray,
    z_values: np.ndarray,
) -> np.ndarray:
    x0 = int(np.argmin(np.abs(x)))
    y0 = x0
    I = np.zeros_like(z_values, dtype=float)
    for i, z in enumerate(z_values):
        U = propagator(float(z)).U
        I[i] = np.abs(U[y0, x0]) ** 2
    return I


def ring_mean_I_vs_z_from_propagator(
    propagator: Callable[[float], ScalarField],
    R: np.ndarray,
    z_values: np.ndarray,
    r_center: float,
    dr_frac: float = 0.12,
) -> np.ndarray:
    mask = (R >= (1.0 - dr_frac) * r_center) & (R <= (1.0 + dr_frac) * r_center)
    I = np.zeros_like(z_values, dtype=float)
    for i, z in enumerate(z_values):
        U = propagator(float(z)).U
        Iz = np.abs(U) ** 2
        I[i] = float(Iz[mask].mean())
    return I


def plot_baliyan_diagnostics(
    out: Dict[str, object],
    beam: BeamSpec,
    z_i: float = 0.0,
    z_f: Optional[float] = None,
    nz: int = 90,
    dr_frac: float = 0.12,
) -> Dict[str, object]:
    if z_f is None:
        z_f = float(out["s"][-1])

    z_values = np.linspace(z_i, z_f, int(nz))
    I_axis = onaxis_I_vs_z_from_propagator(out["propagator"], out["x"], z_values)

    I_theory = None
    if int(beam.ell) == 0:
        zz = z_values
        I_theory = zz * np.exp(-2.0 * (zz / out["zmax"]) ** 2)
        I_theory = I_theory / (I_theory.max() + 1e-15)

    I_ring = None
    r_peak = None
    if int(abs(beam.ell)) > 0:
        rho_peak = sp.jnp_zeros(int(abs(beam.ell)), 1)[0]
        r_peak = rho_peak / out["kr"]
        I_ring = ring_mean_I_vs_z_from_propagator(
            out["propagator"], out["grid"].R, z_values, r_peak, dr_frac=dr_frac
        )
        print(f"Estimated main-ring radius ~ {r_peak / mm:.4f} mm (from J'_l zero)")
        print("Note: for l>0, on-axis is expected near zero (dark vortex core).")

    plt.figure(figsize=(9.0, 4.8))
    plt.plot(z_values / mm, I_axis / (I_axis.max() + 1e-15), lw=2, label="On-axis I(z) / max")

    if I_theory is not None:
        plt.plot(
            z_values / mm,
            I_theory,
            lw=2,
            ls="--",
            label="Theory (l=0): z exp[-2(z/zmax)^2]",
        )

    if I_ring is not None:
        plt.plot(
            z_values / mm,
            I_ring / (I_ring.max() + 1e-15),
            lw=2,
            label="Main-ring mean I(z) / max",
        )

    plt.axvline(out["zmax"] / mm, color="k", ls="--", lw=1.2, label="zmax (Eq.5)")
    plt.xlabel("z (mm)")
    plt.ylabel("Normalized intensity metric")
    plt.title("Diagnostics: on-axis (and ring metric for vortex)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "z_values": z_values,
        "I_axis": I_axis,
        "I_theory": I_theory,
        "I_ring": I_ring,
        "r_peak": r_peak,
    }


from dataclasses import replace
from scipy.ndimage import gaussian_filter1d
from IPython.display import Markdown, display


def normalized_overlap(I_ref: np.ndarray, I_test: np.ndarray, dx: float, roi_mask: Optional[np.ndarray] = None) -> float:
    if roi_mask is None:
        roi_mask = np.ones_like(I_ref, dtype=bool)

    A = np.maximum(I_ref[roi_mask], 0.0)
    B = np.maximum(I_test[roi_mask], 0.0)

    A = A / (A.sum() * dx * dx + 1e-30)
    B = B / (B.sum() * dx * dx + 1e-30)
    return float(np.sum(np.sqrt(A * B)) * dx * dx)


def radial_profile_smoothed(
    I: np.ndarray,
    x: np.ndarray,
    smooth_sigma_bins: float = 2.0,
) -> Dict[str, np.ndarray]:
    r, profile = radial_profile(I, x)
    profile = np.asarray(profile, float)
    profile_norm = profile / (profile.max() + 1e-15)
    profile_smooth = gaussian_filter1d(profile_norm, smooth_sigma_bins, mode="nearest")
    return {
        "r": r,
        "profile": profile,
        "profile_norm": profile_norm,
        "profile_smooth": profile_smooth,
    }


def refine_radial_profile(
    r: np.ndarray,
    y: np.ndarray,
    refine_factor: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    refine_factor = max(1, int(refine_factor))
    if refine_factor == 1 or len(r) < 2:
        return np.asarray(r, float), np.asarray(y, float)

    n_fine = (len(r) - 1) * refine_factor + 1
    r_fine = np.linspace(float(r[0]), float(r[-1]), int(n_fine))
    y_fine = np.interp(r_fine, r, y)
    return r_fine, y_fine


def _interp_level(r1: float, p1: float, r2: float, p2: float, target: float) -> float:
    if np.isclose(p2, p1):
        return float(0.5 * (r1 + r2))
    return float(r1 + (target - p1) * (r2 - r1) / (p2 - p1))


def _local_maxima_indices(y: np.ndarray) -> np.ndarray:
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1


def _local_minima_indices(y: np.ndarray) -> np.ndarray:
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:]))[0] + 1


def _find_first_right_crossing(r: np.ndarray, y: np.ndarray, start_idx: int, level: float) -> float:
    idx = int(start_idx)
    while idx < len(y) - 1 and y[idx] >= level:
        idx += 1

    if idx == 0:
        return float(r[0])
    if idx >= len(y):
        return float(r[-1])
    return float(_interp_level(r[idx - 1], y[idx - 1], r[idx], y[idx], level))


def _find_first_left_crossing(r: np.ndarray, y: np.ndarray, start_idx: int, level: float) -> float:
    idx = int(start_idx)
    while idx > 0 and y[idx] >= level:
        idx -= 1

    if idx == start_idx:
        return float(r[start_idx])
    return float(_interp_level(r[idx], y[idx], r[idx + 1], y[idx + 1], level))


def predicted_ring_peak_radii(ell: int, kr: float, count: int = 2) -> np.ndarray:
    l_abs = int(abs(ell))
    if l_abs == 0 or count <= 0:
        return np.asarray([], dtype=float)
    return np.asarray(sp.jnp_zeros(l_abs, int(count)), dtype=float) / float(kr)


def _peak_index_near_radius(
    r: np.ndarray,
    y: np.ndarray,
    r_center: float,
    half_window: float,
    floor: float = 0.0,
) -> Optional[int]:
    lo = max(0.0, float(r_center - half_window))
    hi = float(r_center + half_window)
    mask = (r >= lo) & (r <= hi) & (y >= floor)
    if not np.any(mask):
        return None
    idx_local = int(np.argmax(y[mask]))
    return int(np.flatnonzero(mask)[idx_local])


def _peak_index_in_bounds(
    r: np.ndarray,
    y: np.ndarray,
    r_min: float,
    r_max: float,
    floor: float = 0.0,
) -> Optional[int]:
    lo = float(min(r_min, r_max))
    hi = float(max(r_min, r_max))
    mask = (r >= lo) & (r <= hi) & (y >= floor)
    if not np.any(mask):
        return None
    idx_local = int(np.argmax(y[mask]))
    return int(np.flatnonzero(mask)[idx_local])


def threshold_interval_around_reference(
    z: np.ndarray,
    y: np.ndarray,
    ref_idx: int,
    level: float,
) -> Tuple[float, float, int, int]:
    ref_idx = int(ref_idx)
    if len(z) != len(y):
        raise ValueError("z and y must have the same length")
    if not (0 <= ref_idx < len(z)):
        raise IndexError("ref_idx is out of range")
    if y[ref_idx] < level:
        z_ref = float(z[ref_idx])
        return z_ref, z_ref, ref_idx, ref_idx

    i_start = ref_idx
    while i_start > 0 and y[i_start - 1] >= level:
        i_start -= 1

    if i_start == 0:
        z_start = float(z[0])
    else:
        z_start = float(_interp_level(z[i_start - 1], y[i_start - 1], z[i_start], y[i_start], level))

    i_end = ref_idx
    while i_end < len(y) - 1 and y[i_end + 1] >= level:
        i_end += 1

    if i_end == len(y) - 1:
        z_end = float(z[-1])
    else:
        z_end = float(_interp_level(z[i_end], y[i_end], z[i_end + 1], y[i_end + 1], level))

    return z_start, z_end, i_start, i_end


def extract_inner_ring_metrics(
    I: np.ndarray,
    x: np.ndarray,
    ell: int,
    kr: float,
    halfmax_level: float = 0.5,
    smooth_sigma_bins: float = 2.0,
    min_prominence: float = 0.02,
    radial_refine_factor: int = 8,
    min_second_peak_ratio: float = 0.02,
    min_second_peak_contrast: float = 0.005,
) -> Dict[str, object]:
    profile_data = radial_profile_smoothed(I, x, smooth_sigma_bins=smooth_sigma_bins)
    r_raw = np.asarray(profile_data["r"], float)
    profile_smooth_raw = np.asarray(profile_data["profile_smooth"], float)
    r, profile_smooth = refine_radial_profile(r_raw, profile_smooth_raw, refine_factor=radial_refine_factor)
    profile_norm = np.interp(r, r_raw, np.asarray(profile_data["profile_norm"], float))
    dr = float(r[1] - r[0])

    if int(ell) == 0:
        first_peak_idx = 0
        second_peak_idx = None
        r_inner = 0.0
        r_outer = 0.0
        first_peak_radius = 0.0
        first_peak_diameter = 0.0
        ring_width = float("nan")
        expected_second_peak_radius = float("nan")
    else:
        predicted_radii = predicted_ring_peak_radii(ell, kr, count=3)
        predicted_first_ring = float(predicted_radii[0])
        predicted_second_ring = float(predicted_radii[1]) if len(predicted_radii) > 1 else float("nan")

        first_window = max(3.0 * dr, 0.40 * predicted_first_ring)
        first_peak_idx = _peak_index_near_radius(
            r,
            profile_smooth,
            predicted_first_ring,
            half_window=first_window,
            floor=0.01,
        )
        if first_peak_idx is None:
            first_peak_idx = int(np.argmin(np.abs(r - predicted_first_ring)))

        peak_height = float(profile_smooth[first_peak_idx])
        half_level = float(max(1e-6, halfmax_level * peak_height))
        r_inner = _find_first_left_crossing(r, profile_smooth, first_peak_idx, half_level)
        r_outer = _find_first_right_crossing(r, profile_smooth, first_peak_idx, half_level)
        first_peak_radius = float(r[first_peak_idx])
        first_peak_diameter = float(2.0 * first_peak_radius)
        ring_width = float(max(0.0, r_outer - r_inner))
        if np.isfinite(predicted_second_ring) and predicted_first_ring > 0:
            expected_second_peak_radius = float(first_peak_radius * predicted_second_ring / predicted_first_ring)
        else:
            expected_second_peak_radius = float("nan")

        if np.isfinite(predicted_second_ring):
            predicted_third_ring = float(predicted_radii[2]) if len(predicted_radii) > 2 else float("nan")
            predicted_spacing = float(max(dr, predicted_second_ring - predicted_first_ring))
            second_window = max(6.0 * dr, 0.80 * predicted_spacing)
            search_hi = predicted_third_ring if np.isfinite(predicted_third_ring) else (predicted_second_ring + 1.25 * predicted_spacing)
            i_hi = min(len(r) - 1, int(np.searchsorted(r, search_hi, side="right")) - 1)
            if i_hi <= first_peak_idx:
                # Under strong distortion / truncation the detected "first ring"
                # can drift beyond the ideal second-ring search window. In that
                # regime the profile no longer supports a reliable second-ring
                # measurement, so we fall back to the geometric spacing estimate
                # below instead of inventing a peak from an empty interval.
                second_peak_idx = None
            else:
                search_slice = profile_smooth[first_peak_idx:i_hi + 1]
                valley_idx = first_peak_idx + int(np.argmin(search_slice))
                valley_height = float(np.min(search_slice))
                search_lo = max(r[valley_idx] + dr, predicted_second_ring - second_window)
                if search_lo >= search_hi:
                    second_peak_idx = None
                else:
                    maxima = _local_maxima_indices(profile_smooth)
                    candidate_maxima = [
                        int(idx)
                        for idx in maxima
                        if idx > valley_idx
                        and r[idx] >= search_lo
                        and r[idx] <= search_hi
                        and profile_smooth[idx] >= max(0.01, min_prominence * peak_height)
                    ]
                    if candidate_maxima:
                        second_peak_idx = min(candidate_maxima, key=lambda idx: abs(r[idx] - predicted_second_ring))
                    else:
                        second_peak_idx = _peak_index_in_bounds(
                            r,
                            profile_smooth,
                            search_lo,
                            search_hi,
                            floor=max(0.01, min_prominence * peak_height),
                        )
                    if second_peak_idx is not None:
                        second_peak_height = float(profile_smooth[second_peak_idx])
                        contrast = second_peak_height - valley_height
                        if second_peak_height < min_second_peak_ratio * peak_height:
                            second_peak_idx = None
                        elif contrast < min_second_peak_contrast * peak_height:
                            second_peak_idx = None
        else:
            second_peak_idx = None

    if second_peak_idx is None:
        second_peak_radius = float("nan")
        if np.isfinite(expected_second_peak_radius):
            ring_spacing = float(max(0.0, expected_second_peak_radius - first_peak_radius))
            spacing_method = "scaled_from_first_peak"
        else:
            ring_spacing = float("nan")
            spacing_method = "unavailable"
    else:
        second_peak_radius_direct = float(r[second_peak_idx])
        if np.isfinite(expected_second_peak_radius):
            mismatch = abs(second_peak_radius_direct - expected_second_peak_radius)
            mismatch_limit = max(3.0 * dr, 0.25 * max(dr, expected_second_peak_radius - first_peak_radius))
            if mismatch > mismatch_limit:
                second_peak_radius = float("nan")
                ring_spacing = float(max(0.0, expected_second_peak_radius - first_peak_radius))
                spacing_method = "scaled_from_first_peak"
            else:
                second_peak_radius = second_peak_radius_direct
                ring_spacing = float(second_peak_radius - first_peak_radius)
                spacing_method = "profile"
        else:
            second_peak_radius = second_peak_radius_direct
            ring_spacing = float(second_peak_radius - first_peak_radius)
            spacing_method = "profile"

    return {
        "r": r,
        "profile_norm": profile_norm,
        "profile_smooth": profile_smooth,
        "first_peak_idx": int(first_peak_idx),
        "second_peak_idx": second_peak_idx,
        "r_first_peak": float(first_peak_radius),
        "r_second_peak": float(second_peak_radius),
        "diameter": float(first_peak_diameter),
        "width": float(ring_width),
        "spacing": float(ring_spacing),
        "spacing_method": spacing_method,
        "r_half_inner": float(r_inner),
        "r_half_outer": float(r_outer),
    }


def choose_similarity_reference_z(
    zmax: float,
    z_eval: float,
    mode: str = "fraction_of_zmax",
    fraction: float = 0.25,
) -> float:
    if mode == "fixed_z_eval":
        return float(z_eval)
    if mode == "fraction_of_zmax":
        return float(fraction * zmax)
    if mode == "earlier_of_eval_and_fraction":
        return float(min(z_eval, fraction * zmax))
    if mode == "later_of_eval_and_fraction":
        return float(max(z_eval, fraction * zmax))
    raise ValueError("Unsupported similarity-reference mode")


def similarity_roi_mask(
    grid: Grid,
    ring_metrics: Dict[str, object],
    min_radius: float = 0.35 * mm,
    scale: float = 1.6,
) -> Tuple[np.ndarray, float]:
    candidates = [
        float(ring_metrics["r_first_peak"]),
        float(ring_metrics["r_half_outer"]),
    ]
    if np.isfinite(ring_metrics["r_second_peak"]):
        candidates.append(float(ring_metrics["r_second_peak"]))

    roi_radius = float(max(min_radius, scale * max(candidates)))
    return grid.R <= roi_radius, roi_radius


def build_scalar_case(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
    show_debug: bool = False,
) -> Dict[str, object]:
    field_in = build_input_field(gs, beam)

    bench = OpticalBench()
    bench.add(SLMPhase(beam=beam, axicon=axicon, mask=mask, realism=realism, name="SLM"))
    bench.add(CameraElement(name="After SLM"))

    field_after_slm, captures, meta = bench.run(field_in)
    phase_slm = meta["SLM"]["phase"]

    if show_debug:
        kr, zmax, r0 = print_sanity_block(field_after_slm, beam, axicon, realism)
    else:
        k = 2 * np.pi / beam.wavelength
        gamma = np.deg2rad(axicon.gamma_deg)
        kr = compute_kr(k, axicon.n_axicon, beam.n_medium, gamma, mode=axicon.kr_mode)
        zmax = zmax_baliyan(beam.w0, k, kr)
        r0 = core_radius_estimate(kr)

    return {
        "field_after_slm": field_after_slm,
        "captures": captures,
        "phase_slm": phase_slm,
        "slm_meta": meta.get("SLM", {}),
        "propagator": build_propagator(field_after_slm, sim),
        "grid": field_after_slm.grid,
        "x": field_after_slm.grid.x,
        "kr": float(kr),
        "zmax": float(zmax),
        "r0": float(r0),
    }


def scan_distance_for_case(
    zmax: float,
    z_eval: float,
    floor_distance: float,
    zmax_factor: float,
) -> float:
    return float(max(floor_distance, 1.15 * z_eval, zmax_factor * zmax))


def run_full_scan_case(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
    z_eval: float,
    ns_points: int,
    floor_distance: float,
    zmax_factor: float,
    similarity_threshold: float,
    z_ref_mode: str,
    z_ref_fraction: float,
    smooth_sigma_bins: float,
    min_prominence: float,
    radial_refine_factor: int,
    similarity_smooth_sigma: float,
    show_debug: bool = False,
) -> Dict[str, object]:
    case = build_scalar_case(gs, beam, axicon, mask, realism, sim, show_debug=show_debug)
    s_f = scan_distance_for_case(case["zmax"], z_eval, floor_distance, zmax_factor)
    z_values = np.linspace(0.0, s_f, int(ns_points))
    z_eval_use = float(np.clip(z_eval, z_values[0], z_values[-1]))
    z_metric_ref = choose_similarity_reference_z(
        case["zmax"],
        z_eval_use,
        mode=z_ref_mode,
        fraction=z_ref_fraction,
    )
    z_metric_ref = float(np.clip(z_metric_ref, z_values[0], z_values[-1]))

    I_eval = np.abs(case["propagator"](z_eval_use).U) ** 2
    ring_metrics = extract_inner_ring_metrics(
        I_eval,
        case["x"],
        ell=beam.ell,
        kr=case["kr"],
        halfmax_level=0.5,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
    )
    I_ref = np.abs(case["propagator"](z_metric_ref).U) ** 2
    ring_metrics_ref = extract_inner_ring_metrics(
        I_ref,
        case["x"],
        ell=beam.ell,
        kr=case["kr"],
        halfmax_level=0.5,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
    )
    roi_mask, roi_radius = similarity_roi_mask(case["grid"], ring_metrics_ref)

    xs = np.zeros((case["grid"].N, len(z_values)), dtype=np.float32)
    similarity_raw = np.zeros(len(z_values), dtype=float)
    y0 = case["grid"].N // 2

    for idx, z_now in enumerate(z_values):
        I_now = np.abs(case["propagator"](float(z_now)).U) ** 2
        xs[:, idx] = I_now[y0, :].astype(np.float32)
        similarity_raw[idx] = normalized_overlap(I_ref, I_now, case["grid"].dx, roi_mask=roi_mask)

    if similarity_smooth_sigma > 0:
        similarity = gaussian_filter1d(similarity_raw, similarity_smooth_sigma, mode="nearest")
    else:
        similarity = similarity_raw.copy()

    ref_idx = int(np.argmin(np.abs(z_values - z_metric_ref)))
    z_bessel_start, z_bessel_end, i_start, i_end = threshold_interval_around_reference(
        z_values,
        similarity,
        ref_idx=ref_idx,
        level=similarity_threshold,
    )

    return {
        **case,
        "beam": beam,
        "axicon": axicon,
        "z_values": z_values,
        "z_eval": float(z_eval_use),
        "z_metric_ref": float(z_metric_ref),
        "I_eval": I_eval.astype(np.float32),
        "I_ref": I_ref.astype(np.float32),
        "xs": xs,
        "ring_metrics": ring_metrics,
        "ring_metrics_ref": ring_metrics_ref,
        "similarity_raw": similarity_raw,
        "similarity": similarity,
        "similarity_threshold": float(similarity_threshold),
        "roi_radius": float(roi_radius),
        "bessel_start": float(z_bessel_start),
        "bessel_end": float(z_bessel_end),
        "bessel_length": float(z_bessel_end - z_bessel_start),
    }


def run_single_plane_case(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
    z_eval: float,
    smooth_sigma_bins: float,
    min_prominence: float,
    radial_refine_factor: int,
    show_debug: bool = False,
) -> Dict[str, object]:
    case = build_scalar_case(gs, beam, axicon, mask, realism, sim, show_debug=show_debug)
    I_eval = np.abs(case["propagator"](float(z_eval)).U) ** 2
    ring_metrics = extract_inner_ring_metrics(
        I_eval,
        case["x"],
        ell=beam.ell,
        kr=case["kr"],
        halfmax_level=0.5,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
    )
    return {
        **case,
        "beam": beam,
        "axicon": axicon,
        "z_eval": float(z_eval),
        "I_eval": I_eval.astype(np.float32),
        "ring_metrics": ring_metrics,
    }


def metric_limit_mm_from_results(results: List[Dict[str, object]]) -> float:
    images = [row["I_eval"] for row in results]
    x = results[0]["x"]
    return auto_crop_mm_from_images(
        images,
        x,
        energy_frac=0.997,
        pad=1.12,
        min_mm=0.35,
        max_mm=1.30,
    )


def rows_by_value(
    rows: List[Dict[str, object]],
    values_or_getter: Union[List[float], str, Callable[[Dict[str, object]], float]],
    getter_or_values: Union[List[float], str, Callable[[Dict[str, object]], float]],
    atol: float = 1e-12,
) -> List[Dict[str, object]]:
    """
    Backward-compatible row selector. Supports BOTH call styles:

        rows_by_value(rows, [targets], lambda row: ...)
        rows_by_value(rows, "beam.ell", [targets])

    The dotted-path string form is convenient for notebooks.
    """

    def dotted_get(row: Dict[str, object], path: str) -> float:
        obj = row
        for part in path.split('.'):
            if isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        return float(obj)

    if callable(values_or_getter) and not callable(getter_or_values):
        value_getter = values_or_getter
        values = getter_or_values
    elif callable(getter_or_values):
        values = values_or_getter
        value_getter = getter_or_values
    elif isinstance(values_or_getter, str):
        values = getter_or_values
        value_getter = lambda row, path=values_or_getter: dotted_get(row, path)
    elif isinstance(getter_or_values, str):
        values = values_or_getter
        value_getter = lambda row, path=getter_or_values: dotted_get(row, path)
    else:
        raise TypeError("rows_by_value expected either (rows, values, getter) or (rows, dotted_path, values)")

    out = []
    for target in values:
        found = False
        for row in rows:
            if abs(float(value_getter(row)) - float(target)) <= atol:
                out.append(row)
                found = True
                break
        if not found:
            raise KeyError(f"Could not find row for value {target}")
    return out


def plot_xz_comparison(
    rows: List[Dict[str, object]],
    labels: List[str],
    title: str,
    x_lim_mm: float = 0.8,
) -> None:
    if len(rows) == 0:
        return

    fig, axes = plt.subplots(1, len(rows), figsize=(4.3 * len(rows), 4.0), constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]

    im = None
    for ax, row, label in zip(axes, rows, labels):
        x_mm = row["x"] / mm
        z_mm = row["z_values"] / mm
        xs_norm = row["xs"] / (np.max(row["xs"]) + 1e-15)
        i0 = int(np.searchsorted(x_mm, -x_lim_mm, side="left"))
        i1 = int(np.searchsorted(x_mm, x_lim_mm, side="right"))
        if i1 <= i0:
            i0, i1 = 0, len(x_mm)

        im = ax.imshow(
            xs_norm[i0:i1, :],
            extent=[z_mm[0], z_mm[-1], x_mm[i0], x_mm[i1 - 1]],
            origin="lower",
            aspect="auto",
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            interpolation="spline36",
        )
        ax.set_title(label)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("x (mm)")

    fig.suptitle(title)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("I / Imax")
    plt.show()


def plot_metric_diagnostic(
    row: Dict[str, object],
    title: str,
    x_lim_mm: Optional[float] = None,
) -> None:
    if "z_values" not in row or "similarity" not in row:
        return

    ring = row["ring_metrics"]
    first_radius_mm = float(ring["r_first_peak"] / mm)
    second_radius_mm = float((ring["r_first_peak"] + ring["spacing"]) / mm) if np.isfinite(ring["spacing"]) else float("nan")
    inner_half_mm = float(ring["r_half_inner"] / mm)
    outer_half_mm = float(ring["r_half_outer"] / mm)
    beam_lim_mm = float(max(0.08, 1.55 * max(first_radius_mm, outer_half_mm)))
    profile_lim_mm = float(max(0.12, 1.35 * max(outer_half_mm, second_radius_mm if np.isfinite(second_radius_mm) else outer_half_mm)))
    xz_lim_mm = float(max(0.14, 2.5 * max(first_radius_mm, outer_half_mm)))
    if x_lim_mm is not None:
        xz_lim_mm = min(float(x_lim_mm), xz_lim_mm)

    fig = plt.figure(figsize=(12.0, 7.8), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_xy = fig.add_subplot(grid_spec[0, 0])
    ax_profile = fig.add_subplot(grid_spec[0, 1])
    ax_xz = fig.add_subplot(grid_spec[1, 0])
    ax_similarity = fig.add_subplot(grid_spec[1, 1])

    x_mm = row["x"] / mm
    I_norm = row["I_eval"] / (np.max(row["I_eval"]) + 1e-15)
    I_view = np.power(I_norm, 0.55)
    ax_xy.imshow(
        I_view,
        extent=row["grid"].extent_mm,
        origin="lower",
        cmap="hot",
        vmin=0.0,
        vmax=1.0,
        interpolation="spline36",
    )
    ax_xy.set_xlim(-beam_lim_mm, beam_lim_mm)
    ax_xy.set_ylim(-beam_lim_mm, beam_lim_mm)
    ax_xy.set_aspect("equal")
    ax_xy.set_title("Inner-ring geometry")
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")

    circle_specs = [
        (inner_half_mm, "#58a6ff", "--"),
        (first_radius_mm, "#f6c945", "-"),
        (outer_half_mm, "#58a6ff", "--"),
    ]

    for radius_mm, color, linestyle in circle_specs:
        if radius_mm > 0:
            ax_xy.add_patch(plt.Circle((0.0, 0.0), radius_mm, fill=False, color=color, linewidth=1.8, linestyle=linestyle, alpha=0.95))

    profile_peak = float(ring["profile_smooth"][ring["first_peak_idx"]])
    profile_half_level = 0.5 * profile_peak
    r_profile_mm = ring["r"] / mm
    ax_profile.plot(r_profile_mm, ring["profile_norm"], color="#b7c5d3", linewidth=1.0, alpha=0.9)
    ax_profile.plot(r_profile_mm, ring["profile_smooth"], color="#1f4e79", linewidth=2.3)
    ax_profile.axvline(first_radius_mm, color="#f6c945", linewidth=1.8, label="first ring radius")
    ax_profile.axvline(inner_half_mm, color="#58a6ff", linewidth=1.4, linestyle="--", label="FWHM edges")
    ax_profile.axvline(outer_half_mm, color="#58a6ff", linewidth=1.4, linestyle="--")
    ax_profile.hlines(profile_half_level, inner_half_mm, outer_half_mm, color="#58a6ff", linewidth=1.6)
    first_peak_height = float(np.interp(first_radius_mm, r_profile_mm, ring["profile_smooth"]))
    ax_profile.plot(first_radius_mm, first_peak_height, marker="o", color="#f6c945", markersize=5)
    if np.isfinite(second_radius_mm):
        second_style = "-." if ring["spacing_method"] == "profile" else ":"
        second_peak_height = float(np.interp(second_radius_mm, r_profile_mm, ring["profile_smooth"]))
        spacing_label = "second ring radius" if ring["spacing_method"] == "profile" else "guided second-ring radius"
        ax_profile.axvline(second_radius_mm, color="#ff8c42", linewidth=1.6, linestyle=second_style, label=spacing_label)
        ax_profile.plot(second_radius_mm, second_peak_height, marker="o", color="#ff8c42", markersize=5)
        y_spacing = max(0.10, 0.18 * profile_peak)
        ax_profile.annotate(
            "",
            xy=(second_radius_mm, y_spacing),
            xytext=(first_radius_mm, y_spacing),
            arrowprops=dict(arrowstyle="<->", color="#ff8c42", lw=1.5),
        )
    ax_profile.annotate(
        "",
        xy=(outer_half_mm, profile_half_level),
        xytext=(inner_half_mm, profile_half_level),
        arrowprops=dict(arrowstyle="<->", color="#58a6ff", lw=1.5),
    )
    ax_profile.set_xlim(0.0, profile_lim_mm)
    ax_profile.set_ylim(0.0, 1.05)
    ax_profile.set_title("Radial profile")
    ax_profile.set_xlabel("radius (mm)")
    ax_profile.set_ylabel("normalized intensity")
    ax_profile.grid(alpha=0.28)
    ax_profile.legend(frameon=False, fontsize=9, loc="upper right")

    z_mm = row["z_values"] / mm
    xs_norm = row["xs"] / (np.max(row["xs"]) + 1e-15)
    xs_view = np.power(xs_norm, 0.60)
    i0 = int(np.searchsorted(x_mm, -xz_lim_mm, side="left"))
    i1 = int(np.searchsorted(x_mm, xz_lim_mm, side="right"))
    if i1 <= i0:
        i0, i1 = 0, len(x_mm)

    ax_xz.imshow(
        xs_view[i0:i1, :],
        extent=[z_mm[0], z_mm[-1], x_mm[i0], x_mm[i1 - 1]],
        origin="lower",
        aspect="auto",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        interpolation="spline36",
    )
    ax_xz.axvspan(row["bessel_start"] / mm, row["bessel_end"] / mm, color="#d9ffff", alpha=0.12)
    ax_xz.axvline(row["bessel_start"] / mm, color="#7fd3d4", linewidth=1.6, label="start / end")
    ax_xz.axvline(row["bessel_end"] / mm, color="#7fd3d4", linewidth=1.6)
    ax_xz.axvline(row["z_metric_ref"] / mm, color="white", linewidth=1.4, linestyle="--", label="reference slice")
    ax_xz.set_title("x-z intensity")
    ax_xz.set_xlabel("z (mm)")
    ax_xz.set_ylabel("x (mm)")
    ax_xz.legend(frameon=False, fontsize=9, loc="upper right")

    similarity_raw = np.asarray(row.get("similarity_raw", row["similarity"]), float)
    similarity = np.asarray(row["similarity"], float)
    threshold = float(row["similarity_threshold"])
    mask = similarity >= threshold
    ax_similarity.fill_between(z_mm, threshold, similarity, where=mask, color="#9dd9d2", alpha=0.24, label="accepted interval")
    ax_similarity.plot(z_mm, similarity_raw, color="#b7c5d3", linewidth=1.0, alpha=0.9, label="raw overlap")
    ax_similarity.plot(z_mm, similarity, color="#1f4e79", linewidth=2.2, label="smoothed overlap")
    ax_similarity.axhline(threshold, color="#c23b22", linewidth=1.5, linestyle="--", label=f"threshold = {threshold:.2f}")
    ax_similarity.axvline(row["z_metric_ref"] / mm, color="#4c956c", linewidth=1.5, linestyle="--", label="reference slice")
    ax_similarity.axvline(row["bessel_start"] / mm, color="#7fd3d4", linewidth=1.5, label="start / end")
    ax_similarity.axvline(row["bessel_end"] / mm, color="#7fd3d4", linewidth=1.5)
    y_min = min(float(np.min(similarity_raw)), float(row["similarity_threshold"])) - 0.04
    ax_similarity.set_ylim(max(0.0, y_min), 1.01)
    ax_similarity.set_title("Similarity to the reference slice")
    ax_similarity.set_xlabel("z (mm)")
    ax_similarity.set_ylabel("overlap")
    ax_similarity.grid(alpha=0.28)
    ax_similarity.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle(title)
    fig.text(
        0.5,
        0.965,
        f"Bessel length here = the continuous z interval around the reference slice where overlap stays above {threshold:.2f}.",
        ha="center",
        fontsize=10,
    )
    plt.show()


def plot_metric_grid(
    x_values: np.ndarray,
    curves: List[Tuple[np.ndarray, str, str]],
    xlabel: str,
    figure_title: str,
    ncols: int = 2,
) -> None:
    nplots = len(curves)
    ncols = min(max(1, ncols), nplots)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 3.9 * nrows), constrained_layout=True, squeeze=False)

    for ax, (y_values, ylabel, title) in zip(axes.flat, curves):
        ax.plot(x_values, y_values, marker="o", linewidth=2.2, color="#1f4e79")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.28)

    for ax in axes.flat[nplots:]:
        ax.axis("off")

    fig.suptitle(figure_title)
    plt.show()


def summary_table_from_results(rows: List[Dict[str, object]]) -> List[Dict[str, float]]:
    table = []
    for row in rows:
        ring = row["ring_metrics"]
        table.append({
            "inner_diameter_mm": float(ring["diameter"] / mm),
            "ring_width_mm": float(ring["width"] / mm),
            "ring_spacing_mm": float(ring["spacing"] / mm),
            "bessel_length_mm": float(row.get("bessel_length", np.nan) / mm),
            "eq5_zmax_mm": float(row["zmax"] / mm),
        })
    return table




# -----------------------------------------------------------------------------
# Shared defaults and convenience helpers for the split notebook workflow
# -----------------------------------------------------------------------------

QUALITY_MAP = {
    "fast": {"N": 1024, "ns": 56},
    "balanced": {"N": 1536, "ns": 72},
    "paper": {"N": 2048, "ns": 108},
}


def make_default_context(
    quality: str = "balanced",
    metric_quality: str = "paper",
    showcase_quality: str = "paper",
    use_showcase_quality: bool = False,
    grid_half_width: float = 5.0 * mm,
    wavelength: float = 532 * nm,
    n_medium: float = 1.0,
    w0: float = 2.0 * mm,
    ell: int = 3,
    n_axicon: float = 1.5,
    gamma_deg: float = 1.0,
    kr_mode: str = "tan",
):
    """Build a reusable dictionary of shared defaults.

    The split notebooks all start from this helper so they share the same
    numerical conventions while remaining independently runnable.
    """
    if quality not in QUALITY_MAP:
        raise ValueError(f"Unknown quality={quality!r}")
    if metric_quality not in QUALITY_MAP:
        raise ValueError(f"Unknown metric_quality={metric_quality!r}")
    if showcase_quality not in QUALITY_MAP:
        raise ValueError(f"Unknown showcase_quality={showcase_quality!r}")

    primary = QUALITY_MAP[quality]
    metric = QUALITY_MAP[metric_quality]
    showcase = QUALITY_MAP[showcase_quality]
    gs = GridSpec(N=int(primary["N"]), L=grid_half_width)
    gs_metric = GridSpec(N=int(metric["N"]), L=grid_half_width)
    gs_showcase = GridSpec(N=int(showcase["N"]), L=grid_half_width)
    gs_display = gs_showcase if use_showcase_quality else gs

    beam_base = BeamSpec(wavelength=wavelength, n_medium=n_medium, w0=w0, ell=ell)
    axicon_base = AxiconSpec(n_axicon=n_axicon, gamma_deg=gamma_deg, kr_mode=kr_mode)
    mask = MaskSpec(signum_pi_flip=False)
    realism = SLMRealism(
        quantize_bits=None,
        pixel_pitch=None,
        carrier_lpmm=0.0,
        tilt_mrad=(0.0, 0.0),
        active_size=None,
        fill_factor=None,
    )
    sim = SimSpec(include_evanescent=True, alpha_np_per_m=0.0)

    slm_active_size = (15.36 * mm, 8.64 * mm)
    active_short_side = slm_active_size[1]
    w0_min = 1.0 * mm
    w0_max = min(3.2 * mm, 0.37 * active_short_side)

    ctx = {
        "quality": quality,
        "metric_quality": metric_quality,
        "showcase_quality": showcase_quality,
        "use_showcase_quality": use_showcase_quality,
        "gs": gs,
        "gs_metric": gs_metric,
        "gs_showcase": gs_showcase,
        "gs_display": gs_display,
        "beam_base": beam_base,
        "axicon_base": axicon_base,
        "mask": mask,
        "realism": realism,
        "sim": sim,
        "slm_active_size": slm_active_size,
        "z_eval": 80 * mm,
        "similarity_threshold": 0.93,
        "bessel_ref_mode": "later_of_eval_and_fraction",
        "bessel_ref_fraction": 0.25,
        "similarity_smooth_sigma": 1.0,
        "smooth_sigma_bins": 2.0,
        "radial_refine_factor": 8,
        "min_peak_prominence": 0.018,
        "z_scan_floor": 220 * mm,
        "zmax_scan_factor": 1.15,
        "xz_lim_mm": 0.85,
        "gamma_show_values_deg": np.array([0.20, 0.45, 0.75, 1.10, 1.50], dtype=float),
        "gamma_values_deg": np.array([0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.10, 1.30, 1.50], dtype=float),
        "w0_show_values": w0_min + (w0_max - w0_min) * np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=float),
        "w0_values": w0_min + (w0_max - w0_min) * np.array([0.00, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.00], dtype=float),
        "l_show_values": [0, 1, 2, 3, 5, 10, 20, 30],
        "l_values": [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30],
    }
    ctx["gamma_diagnostic_value_deg"] = float(ctx["gamma_show_values_deg"][len(ctx["gamma_show_values_deg"]) // 2])
    ctx["w0_diagnostic_value"] = float(ctx["w0_show_values"][len(ctx["w0_show_values"]) // 2])
    ctx["l_diagnostic_value"] = int(beam_base.ell)
    return ctx


def row_to_summary(row: Dict[str, object]) -> Dict[str, float]:
    """Compact scalar metrics from a full-scan row.

    This helper mirrors the summary table values but keeps them easy to reuse
    inside the split notebooks.
    """
    rm = row["ring_metrics"]
    return {
        "inner_diameter_mm": float(rm["diameter"] / mm),
        "ring_width_mm": float(rm["width"] / mm),
        "ring_spacing_mm": float(rm["spacing"] / mm),
        "bessel_length_mm": float(row["bessel_length"] / mm),
        "eq5_zmax_mm": float(row["zmax"] / mm),
    }


def run_standard_case(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism: SLMRealism,
    sim: SimSpec,
    z_eval: float,
    ns_points: int,
    floor_distance: float,
    zmax_factor: float,
    similarity_threshold: float,
    z_ref_mode: str,
    z_ref_fraction: float,
    smooth_sigma_bins: float,
    min_prominence: float,
    radial_refine_factor: int,
    similarity_smooth_sigma: float,
    show_debug: bool = False,
) -> Dict[str, object]:
    row = run_full_scan_case(
        gs=gs,
        beam=beam,
        axicon=axicon,
        mask=mask,
        realism=realism,
        sim=sim,
        z_eval=z_eval,
        ns_points=ns_points,
        floor_distance=floor_distance,
        zmax_factor=zmax_factor,
        similarity_threshold=similarity_threshold,
        z_ref_mode=z_ref_mode,
        z_ref_fraction=z_ref_fraction,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
        similarity_smooth_sigma=similarity_smooth_sigma,
        show_debug=show_debug,
    )
    row.update(row_to_summary(row))
    return row


# -----------------------------------------------------------------------------
# Axial observables for Richard's requested direct measurements
# -----------------------------------------------------------------------------


def annulus_from_ring_metrics(
    metrics: Dict[str, object],
    x: Optional[np.ndarray] = None,
    fallback_halfwidth_factor: float = 0.75,
) -> Dict[str, float]:
    """
    Build a first-ring annulus from already-extracted ring metrics.

    The half-maximum edges are preferred. If those are unavailable, fall back
    to a width-based annulus around the detected first-ring peak.
    """
    ra = float(metrics.get("r_half_inner", np.nan))
    rb = float(metrics.get("r_half_outer", np.nan))
    rpk = float(metrics.get("r_first_peak", metrics.get("r_peak", np.nan)))
    width = float(metrics.get("width", np.nan))

    dx = 0.0
    if x is not None:
        x = np.asarray(x, float)
        if x.size > 1:
            dx = abs(float(x[1] - x[0]))

    if not np.isfinite(ra) or not np.isfinite(rb) or rb <= ra:
        half = fallback_halfwidth_factor * (
            width if np.isfinite(width) and width > 0 else max(abs(rpk) * 0.4, dx * 3.0)
        )
        ra = max(0.0, rpk - half)
        rb = rpk + half

    return {"ra": float(ra), "rb": float(rb), "r_peak": float(rpk)}


def fixed_first_ring_annulus(
    I: np.ndarray,
    x: np.ndarray,
    ell: int,
    kr: float,
    smooth_sigma_bins: float = 2.0,
    min_prominence: float = 0.02,
    radial_refine_factor: int = 8,
    fallback_halfwidth_factor: float = 0.75,
) -> Dict[str, float]:
    """Return a fixed annulus [ra, rb] for the first bright ring.

    The annulus is defined from the reference-plane radial profile.  Richard's
    email is easiest to honour when the annulus is fixed once at the design
    plane and then reused along z.
    """
    metrics = extract_inner_ring_metrics(
        I,
        x,
        ell,
        kr,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
    )
    ann = annulus_from_ring_metrics(metrics, x=x, fallback_halfwidth_factor=fallback_halfwidth_factor)
    out = dict(metrics)
    out.update(ann)
    return out



def annulus_mask(X: np.ndarray, Y: np.ndarray, ra: float, rb: float) -> np.ndarray:
    R = np.sqrt(X ** 2 + Y ** 2)
    return (R >= float(ra)) & (R <= float(rb))



def integrated_intensity_in_annulus(I: np.ndarray, grid: Grid, ra: float, rb: float) -> float:
    mask = annulus_mask(grid.X, grid.Y, ra, rb)
    return float(np.sum(np.asarray(I, float)[mask]) * grid.dx * grid.dx)



def peak_intensity_in_annulus(I: np.ndarray, grid: Grid, ra: float, rb: float) -> float:
    """
    Robust peak estimator inside the annulus.

    The raw max is very sensitive to one hot sample or one drifting hotspot.
    Using the 99th percentile keeps the metric physically interpretable while
    reducing spike artefacts.
    """
    mask = annulus_mask(grid.X, grid.Y, ra, rb)
    vals = np.asarray(I, float)[mask]
    return float(np.nanpercentile(vals, 99)) if vals.size else float("nan")

def peak_intensity_theta0(I: np.ndarray, x: np.ndarray, ra: float, rb: float) -> float:
    """Peak intensity along +x with y = 0.

    This is the practical simplification Richard mentioned.  It is not the main
    metric in simulation, but it is useful as an experimentally friendlier proxy.
    """
    y0 = I.shape[0] // 2
    x1d = np.asarray(x, float)
    Iline = np.asarray(I[y0, :], float)
    mask = (x1d >= 0.0) & (x1d >= float(ra)) & (x1d <= float(rb))
    vals = Iline[mask]
    return float(np.max(vals)) if vals.size else float("nan")



def axial_observables_from_case(
    row: Dict[str, object],
    smooth_sigma_bins: float = 2.0,
    min_prominence: float = 0.02,
    radial_refine_factor: int = 8,
    use_fixed_annulus_from_eval: bool = True,
    annulus_mode: str = "fixed",
) -> Dict[str, object]:
    """
    Compute Richard-style axial observables along z for one case.

    annulus_mode:
        - "fixed": use one reference annulus for all z slices
        - "tracked": re-extract the first ring at each z slice and build a
          fresh annulus for diagnostic comparison
    """
    grid = row["grid"]
    x = row["x"]
    ell = row["beam"].ell
    kr = row["kr"]
    annulus_mode = str(annulus_mode).lower().strip()
    if annulus_mode not in ("fixed", "tracked"):
        raise ValueError("annulus_mode must be 'fixed' or 'tracked'")

    reference_I = row["I_eval"] if use_fixed_annulus_from_eval else row["I_ref"]
    ann = fixed_first_ring_annulus(
        reference_I,
        x,
        ell,
        kr,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
    )
    ra_ref, rb_ref = ann["ra"], ann["rb"]

    z_values = np.asarray(row["z_values"], float)
    propagator = row["propagator"]
    E1 = np.zeros_like(z_values, dtype=float)
    Ipk = np.zeros_like(z_values, dtype=float)
    Ipk_theta0 = np.zeros_like(z_values, dtype=float)
    diameter = np.full_like(z_values, np.nan, dtype=float)
    width = np.full_like(z_values, np.nan, dtype=float)
    spacing = np.full_like(z_values, np.nan, dtype=float)
    ra_values = np.full_like(z_values, np.nan, dtype=float)
    rb_values = np.full_like(z_values, np.nan, dtype=float)
    r_peak_values = np.full_like(z_values, np.nan, dtype=float)
    spacing_method_values = np.full(z_values.shape, "", dtype=object)

    for i, zi in enumerate(z_values):
        Iz = np.abs(propagator(float(zi)).U) ** 2
        rm = extract_inner_ring_metrics(
            Iz, x, ell, kr,
            smooth_sigma_bins=smooth_sigma_bins,
            min_prominence=min_prominence,
            radial_refine_factor=radial_refine_factor,
        )
        if annulus_mode == "tracked":
            ann_now = annulus_from_ring_metrics(rm, x=x)
        else:
            ann_now = ann
        ra_now = float(ann_now["ra"])
        rb_now = float(ann_now["rb"])
        ra_values[i] = ra_now
        rb_values[i] = rb_now
        r_peak_values[i] = float(ann_now["r_peak"])
        E1[i] = integrated_intensity_in_annulus(Iz, grid, ra_now, rb_now)
        Ipk[i] = peak_intensity_in_annulus(Iz, grid, ra_now, rb_now)
        Ipk_theta0[i] = peak_intensity_theta0(Iz, x, ra_now, rb_now)
        diameter[i] = float(rm.get("diameter", np.nan))
        width[i] = float(rm.get("width", np.nan))
        spacing[i] = float(rm.get("spacing", np.nan))
        spacing_method_values[i] = str(rm.get("spacing_method", ""))

    z_ref = float(row["z_eval"])
    ref_idx = int(np.argmin(np.abs(z_values - z_ref)))
    E1_norm = E1 / (E1[ref_idx] + 1e-30)
    Ipk_norm = Ipk / (Ipk[ref_idx] + 1e-30)
    Ipk_theta0_norm = Ipk_theta0 / (Ipk_theta0[ref_idx] + 1e-30)

    return {
        "annulus_mode": annulus_mode,
        "reference_annulus_source": ("eval" if use_fixed_annulus_from_eval else "ref"),
        "ra": ra_ref,
        "rb": rb_ref,
        "r_peak": ann["r_peak"],
        "ra_values": ra_values,
        "rb_values": rb_values,
        "r_peak_values": r_peak_values,
        "reference_metrics": ann,
        "z_values": z_values,
        "z_mm": z_values / mm,
        "z_ref": z_ref,
        "E1": E1,
        "E1_norm": E1_norm,
        "I1_peak": Ipk,
        "I1_peak_norm": Ipk_norm,
        "I1_peak_theta0": Ipk_theta0,
        "I1_peak_theta0_norm": Ipk_theta0_norm,
        "diameter": diameter,
        "diameter_mm": diameter / mm,
        "width": width,
        "width_mm": width / mm,
        "spacing": spacing,
        "spacing_mm": spacing / mm,
        "spacing_method_values": spacing_method_values,
    }



def accepted_interval_mask(z_values: np.ndarray, z_start: float, z_end: float) -> np.ndarray:
    z = np.asarray(z_values, float)
    return (z >= float(z_start)) & (z <= float(z_end))



def relative_rms(y: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    vals = np.asarray(y, float)
    if mask is not None:
        vals = vals[np.asarray(mask, bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    mean = float(np.mean(vals))
    if abs(mean) < 1e-30:
        return float("nan")
    return float(np.sqrt(np.mean((vals - mean) ** 2)) / abs(mean))



def axial_stability_summary(
    row: Dict[str, object],
    axial: Dict[str, object],
    min_points: int = 4,
    min_length_mm: float = 1.0,
) -> Dict[str, float]:
    """
    Stability metrics over the accepted Bessel interval.

    If the accepted interval is degenerate or too short, return NaN for all
    epsilon metrics and make the failure explicit.
    """
    z = np.asarray(axial["z_values"], float)
    z0 = float(row["bessel_start"])
    z1 = float(row["bessel_end"])
    mask = accepted_interval_mask(z, z0, z1)

    n_valid = int(np.sum(mask))
    interval_len = float(z1 - z0)
    interval_len_mm = interval_len / mm

    fail_reason = ""
    if (not np.isfinite(interval_len)) or (interval_len <= 0.0):
        fail_reason = "degenerate_interval"
    elif (not np.isfinite(interval_len_mm)) or (interval_len_mm < float(min_length_mm)):
        fail_reason = "too_short_mm"
    elif n_valid < int(min_points):
        fail_reason = "too_few_points"

    if fail_reason:
        return {
            "epsilon_E": np.nan,
            "epsilon_Ipeak": np.nan,
            "epsilon_Ipeak_theta0": np.nan,
            "epsilon_D": np.nan,
            "epsilon_W": np.nan,
            "axial_interval_valid": False,
            "axial_interval_n": n_valid,
            "axial_interval_length_mm": interval_len_mm,
            "axial_fail_reason": fail_reason,
        }

    out = {
        "epsilon_E": relative_rms(axial["E1_norm"], mask),
        "epsilon_Ipeak": relative_rms(axial["I1_peak_norm"], mask),
        "epsilon_Ipeak_theta0": relative_rms(axial["I1_peak_theta0_norm"], mask),
        "epsilon_D": relative_rms(axial["diameter_mm"], mask),
        "epsilon_W": relative_rms(axial["width_mm"], mask),
        "axial_interval_valid": True,
        "axial_interval_n": n_valid,
        "axial_interval_length_mm": interval_len_mm,
        "axial_fail_reason": "",
    }

    keys = ["epsilon_E", "epsilon_Ipeak", "epsilon_Ipeak_theta0", "epsilon_D", "epsilon_W"]
    if not all(np.isfinite(out[k]) for k in keys):
        out.update({
            "epsilon_E": np.nan,
            "epsilon_Ipeak": np.nan,
            "epsilon_Ipeak_theta0": np.nan,
            "epsilon_D": np.nan,
            "epsilon_W": np.nan,
            "axial_interval_valid": False,
            "axial_fail_reason": "nonfinite_metric",
        })
    return out

def plot_axial_observables(
    row: Dict[str, object],
    axial: Dict[str, object],
    title: str = "Axial first-ring observables",
    show_theta0: bool = True,
) -> None:
    zmm = axial["z_mm"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(zmm, axial["E1_norm"], lw=2)
    axes[0].set_title("Normalised first-ring integrated energy")
    axes[0].set_xlabel("z [mm]")
    axes[0].set_ylabel(r"$E_1(z) / E_1(z_{\rm eval})$")

    axes[1].plot(zmm, axial["I1_peak_norm"], lw=2, label="full annulus")
    if show_theta0:
        axes[1].plot(zmm, axial["I1_peak_theta0_norm"], lw=1.5, ls="--", label=r"$\theta = 0$ cut")
        axes[1].legend(frameon=False)
    axes[1].set_title("Normalised first-ring peak intensity")
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel(r"$I_{\rm peak}(z) / I_{\rm peak}(z_{\rm eval})$")

    axes[2].plot(zmm, axial["diameter_mm"], lw=2)
    axes[2].set_title("First-ring diameter")
    axes[2].set_xlabel("z [mm]")
    axes[2].set_ylabel("Diameter [mm]")

    axes[3].plot(zmm, axial["width_mm"], lw=2)
    axes[3].set_title("First-ring width (FWHM)")
    axes[3].set_xlabel("z [mm]")
    axes[3].set_ylabel("Width [mm]")

    for ax in axes:
        ax.axvline(row["z_eval"] / mm, color="k", ls=":", lw=1)
        ax.axvline(row["bessel_start"] / mm, color="0.5", ls="--", lw=0.9)
        ax.axvline(row["bessel_end"] / mm, color="0.5", ls="--", lw=0.9)
        ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=14)
    plt.show()


# -----------------------------------------------------------------------------
# Pixelation realism helpers
# -----------------------------------------------------------------------------
from scipy.ndimage import map_coordinates, zoom


def bilinear_sample(I: np.ndarray, x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    dx = float(x[1] - x[0])
    x0 = float(x[0])
    coords_y = (yp - x0) / dx
    coords_x = (xp - x0) / dx
    coords = np.vstack([coords_y, coords_x])
    return map_coordinates(np.asarray(I, float), coords, order=1, mode="nearest")



def azimuthal_profile_at_radius(I: np.ndarray, x: np.ndarray, radius: float, n_theta: int = 360) -> Dict[str, np.ndarray]:
    theta = np.linspace(0.0, 2 * np.pi, int(n_theta), endpoint=False)
    xp = float(radius) * np.cos(theta)
    yp = float(radius) * np.sin(theta)
    vals = bilinear_sample(I, x, xp, yp)
    return {"theta": theta, "theta_deg": theta * 180.0 / np.pi, "profile": vals}



def ring_azimuthal_nonuniformity(I: np.ndarray, x: np.ndarray, radius: float, n_theta: int = 360) -> float:
    ap = azimuthal_profile_at_radius(I, x, radius, n_theta=n_theta)["profile"]
    mu = float(np.mean(ap))
    if abs(mu) < 1e-30:
        return float("nan")
    return float(np.std(ap) / abs(mu))



def sampling_feasibility_metric(r_peak: float, ell: int, pixel_pitch: Optional[float]) -> float:
    if ell == 0 or pixel_pitch is None or pixel_pitch <= 0:
        return float("inf")
    return float((2 * np.pi * float(r_peak) / abs(int(ell))) / float(pixel_pitch))



def run_realism_case(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    realism_base: SLMRealism,
    sim: SimSpec,
    z_eval: float,
    ns_points: int,
    floor_distance: float,
    zmax_factor: float,
    similarity_threshold: float,
    z_ref_mode: str,
    z_ref_fraction: float,
    smooth_sigma_bins: float,
    min_prominence: float,
    radial_refine_factor: int,
    similarity_smooth_sigma: float,
    pixel_pitch: Optional[float] = None,
    quantize_bits: Optional[int] = None,
    fill_factor: Optional[float] = None,
    active_size: Optional[Tuple[float, float]] = None,
) -> Dict[str, object]:
    realism_use = replace(realism_base, pixel_pitch=pixel_pitch, quantize_bits=quantize_bits, fill_factor=fill_factor if fill_factor is not None else realism_base.fill_factor, active_size=active_size if active_size is not None else realism_base.active_size)
    row = run_standard_case(
        gs=gs,
        beam=beam,
        axicon=axicon,
        mask=mask,
        realism=realism_use,
        sim=sim,
        z_eval=z_eval,
        ns_points=ns_points,
        floor_distance=floor_distance,
        zmax_factor=zmax_factor,
        similarity_threshold=similarity_threshold,
        z_ref_mode=z_ref_mode,
        z_ref_fraction=z_ref_fraction,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
        similarity_smooth_sigma=similarity_smooth_sigma,
        show_debug=False,
    )
    axial = axial_observables_from_case(
        row,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
    )
    stability = axial_stability_summary(row, axial)
    uniformity = ring_azimuthal_nonuniformity(row["I_eval"], row["x"], axial["r_peak"], n_theta=360)
    row.update(axial)
    row.update(stability)
    row["azimuthal_nonuniformity"] = uniformity
    row["sampling_feasibility"] = sampling_feasibility_metric(axial["r_peak"], beam.ell, pixel_pitch)
    row["effective_pixel_pitch"] = pixel_pitch
    row["quantize_bits"] = quantize_bits
    row["fill_factor"] = realism_use.fill_factor
    row["active_size"] = realism_use.active_size
    return row



def run_realism_grid(
    gs: GridSpec,
    beam_base: BeamSpec,
    axicon_base: AxiconSpec,
    mask: MaskSpec,
    realism_base: SLMRealism,
    sim: SimSpec,
    z_eval: float,
    ns_points: int,
    floor_distance: float,
    zmax_factor: float,
    similarity_threshold: float,
    z_ref_mode: str,
    z_ref_fraction: float,
    smooth_sigma_bins: float,
    min_prominence: float,
    radial_refine_factor: int,
    similarity_smooth_sigma: float,
    ell_values: List[int],
    pixel_pitch_values: List[Optional[float]],
    quantize_bits: Optional[int] = None,
    fill_factor: Optional[float] = None,
    active_size: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, object]]:
    rows = []
    for ell in ell_values:
        beam_use = replace(beam_base, ell=int(ell))
        for pixel_pitch in pixel_pitch_values:
            row = run_realism_case(
                gs=gs,
                beam=beam_use,
                axicon=axicon_base,
                mask=mask,
                realism_base=realism_base,
                sim=sim,
                z_eval=z_eval,
                ns_points=ns_points,
                floor_distance=floor_distance,
                zmax_factor=zmax_factor,
                similarity_threshold=similarity_threshold,
                z_ref_mode=z_ref_mode,
                z_ref_fraction=z_ref_fraction,
                smooth_sigma_bins=smooth_sigma_bins,
                min_prominence=min_prominence,
                radial_refine_factor=radial_refine_factor,
                similarity_smooth_sigma=similarity_smooth_sigma,
                pixel_pitch=pixel_pitch,
                quantize_bits=quantize_bits,
                fill_factor=fill_factor,
                active_size=active_size,
            )
            rows.append(row)
    return rows



def add_relative_to_ideal_metrics(rows: List[Dict[str, object]], key_names: List[str]) -> None:
    ideal_map = {}
    for row in rows:
        if row.get("effective_pixel_pitch") is None:
            ideal_map[row["beam"].ell] = row
    for row in rows:
        ideal = ideal_map.get(row["beam"].ell)
        if ideal is None:
            continue
        for key in key_names:
            ideal_val = float(ideal.get(key, np.nan))
            val = float(row.get(key, np.nan))
            rel = np.nan if not np.isfinite(ideal_val) or abs(ideal_val) < 1e-30 else (val - ideal_val) / ideal_val
            row[f"relerr_{key}"] = rel



def plot_xy_tile(rows_2d: List[List[Dict[str, object]]], col_labels: List[str], row_labels: List[str], lim_mm: float, title: str) -> None:
    nrows = len(rows_2d)
    ncols = len(rows_2d[0]) if nrows else 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, None]
    for i in range(nrows):
        for j in range(ncols):
            row = rows_2d[i][j]
            ax = axes[i, j]
            V = np.asarray(row["I_eval"], float)
            V = V / (V.max() + 1e-15)
            im = ax.imshow(
                V,
                extent=[row["x"][0] / mm, row["x"][-1] / mm, row["x"][0] / mm, row["x"][-1] / mm],
                origin="lower",
                cmap="hot",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            ax.set_xlim(-lim_mm, lim_mm)
            ax.set_ylim(-lim_mm, lim_mm)
            if i == 0:
                ax.set_title(col_labels[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(row_labels[i])
            if i < nrows - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="normalised intensity")
    fig.suptitle(title, fontsize=14)
    plt.show()



def plot_xz_tile(rows_2d: List[List[Dict[str, object]]], col_labels: List[str], row_labels: List[str], x_lim_mm: float, title: str) -> None:
    nrows = len(rows_2d)
    ncols = len(rows_2d[0]) if nrows else 0
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, None]
    for i in range(nrows):
        for j in range(ncols):
            row = rows_2d[i][j]
            ax = axes[i, j]
            x = row["x"] / mm
            z = row["z_values"] / mm
            xs = row["xs"]
            V = xs / (np.max(xs) + 1e-15)
            im = ax.imshow(
                V,
                extent=[z[0], z[-1], x[0], x[-1]],
                origin="lower",
                aspect="auto",
                cmap="turbo",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            ax.set_ylim(-x_lim_mm, x_lim_mm)
            if i == 0:
                ax.set_title(col_labels[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(row_labels[i])
            if i == nrows - 1:
                ax.set_xlabel("z [mm]")
            if j > 0:
                ax.set_yticklabels([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="normalised intensity")
    fig.suptitle(title, fontsize=14)
    plt.show()



def metric_matrix(rows: List[Dict[str, object]], ell_values: List[int], pixel_values: List[Optional[float]], key: str) -> np.ndarray:
    arr = np.full((len(pixel_values), len(ell_values)), np.nan, dtype=float)
    for i, pp in enumerate(pixel_values):
        for j, ell in enumerate(ell_values):
            for row in rows:
                if row["beam"].ell == ell and ((pp is None and row.get("effective_pixel_pitch") is None) or row.get("effective_pixel_pitch") == pp):
                    arr[i, j] = float(row.get(key, np.nan))
                    break
    return arr



def plot_metric_heatmap(matrix: np.ndarray, x_labels: List[str], y_labels: List[str], title: str, cbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(0.8 * len(x_labels) + 2.8, 0.65 * len(y_labels) + 2.2), constrained_layout=True)
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("topological charge l")
    ax.set_ylabel("pixelation level")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    plt.show()


# =============================================================================
# Kernel-first convenience wrappers
# =============================================================================

def make_research_context(
    quality: str = "balanced",
    metric_quality: str = "balanced",
    showcase_quality: str = "paper",
    use_showcase_quality: bool = False,
    grid_half_width: float = 5.0 * mm,
    wavelength: float = 1030 * nm,
    n_medium: float = 1.0,
    w0: float = 2.0 * mm,
    ell: int = 3,
    n_axicon: float = 1.5,
    gamma_deg: float = 1.0,
    kr_mode: str = "tan",
):
    """
    Build a notebook-friendly context dictionary.

    This wrapper is meant for a `%run ./balyian_shared_kernel.py` workflow.
    It starts from the shared defaults, then adds the explicit `ns_points`
    entries and realism-study defaults that the split notebooks need.
    """
    ctx = make_default_context(
        quality=quality,
        metric_quality=metric_quality,
        showcase_quality=showcase_quality,
        use_showcase_quality=use_showcase_quality,
        grid_half_width=grid_half_width,
        wavelength=wavelength,
        n_medium=n_medium,
        w0=w0,
        ell=ell,
        n_axicon=n_axicon,
        gamma_deg=gamma_deg,
        kr_mode=kr_mode,
    )
    ctx["primary_ns_points"] = int(QUALITY_MAP[ctx["quality"]]["ns"])
    ctx["metric_ns_points"] = int(QUALITY_MAP[ctx["metric_quality"]]["ns"])
    ctx["showcase_ns_points"] = int(QUALITY_MAP[ctx["showcase_quality"]]["ns"])
    ctx["pixel_pitch_um_nominal"] = 8.0
    ctx["pixel_pitch_scale_values"] = [None, 0.5, 1.0, 2.0, 4.0, 8.0]
    ctx["quantize_bits_values"] = [None, 10, 8, 6, 4]
    ctx["l_values_realism"] = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
    ctx["selected_pixel_pitch_scales_xy"] = [0.5, 1.0, 4.0, 8.0]
    ctx["selected_l_xy"] = [0, 1, 3, 5, 10, 20, 30]
    ctx["selected_l_xz"] = [0, 5, 15, 30]
    ctx["selected_pixel_pitch_scales_xz"] = [0.5, 1.0, 4.0, 8.0]
    return ctx


def _canonicalize_row_metrics(row):
    """
    Add a consistent set of metric keys without breaking the original row.

    Older helpers use keys like `inner_diameter_mm` / `bessel_length_mm`,
    while the later notebook ideas used names like `ring_diameter` / `L_env`.
    This helper makes both styles available.
    """
    if "ring_metrics" in row:
        rm = row["ring_metrics"]
        row.setdefault("ring_diameter", float(rm["diameter"]))
        row.setdefault("ring_width", float(rm["width"]))
        row.setdefault("ring_spacing", float(rm["spacing"]))
    if "inner_diameter_mm" in row:
        row.setdefault("ring_diameter_mm", float(row["inner_diameter_mm"]))
    elif "ring_diameter" in row:
        row.setdefault("ring_diameter_mm", float(row["ring_diameter"] / mm))
    if "ring_width_mm" in row:
        row.setdefault("ring_width_mm", float(row["ring_width_mm"]))
    elif "ring_width" in row:
        row["ring_width_mm"] = float(row["ring_width"] / mm)
    if "ring_spacing_mm" in row:
        row.setdefault("ring_spacing_mm", float(row["ring_spacing_mm"]))
    elif "ring_spacing" in row:
        row["ring_spacing_mm"] = float(row["ring_spacing"] / mm)
    if "bessel_length" in row:
        row.setdefault("L_env", float(row["bessel_length"]))
        row.setdefault("bessel_length_mm", float(row["bessel_length"] / mm))
    elif "L_env" in row:
        row.setdefault("bessel_length_mm", float(row["L_env"] / mm))
    if "zmax" in row:
        row.setdefault("zmax_geo", float(row["zmax"]))
        row.setdefault("eq5_zmax_mm", float(row["zmax"] / mm))
    elif "zmax_geo" in row:
        row.setdefault("eq5_zmax_mm", float(row["zmax_geo"] / mm))
    return row


def run_case(
    ctx,
    beam=None,
    axicon=None,
    mask=None,
    realism=None,
    sim=None,
    gs=None,
    z_eval=None,
    ns_points=None,
    floor_distance=None,
    zmax_factor=None,
    similarity_threshold=None,
    z_ref_mode=None,
    z_ref_fraction=None,
    smooth_sigma_bins=None,
    min_prominence=None,
    radial_refine_factor=None,
    similarity_smooth_sigma=None,
    show_debug: bool = False,
):
    """
    Run one full scalar case using the context dictionary.

    This is the main notebook-facing wrapper. It hides the long argument list
    of `run_standard_case(...)` and keeps the calling convention stable.
    """
    beam = ctx["beam_base"] if beam is None else beam
    axicon = ctx["axicon_base"] if axicon is None else axicon
    mask = ctx["mask"] if mask is None else mask
    realism = ctx["realism"] if realism is None else realism
    sim = ctx["sim"] if sim is None else sim
    gs = ctx["gs_metric"] if gs is None else gs
    z_eval = ctx["z_eval"] if z_eval is None else z_eval
    ns_points = ctx["metric_ns_points"] if ns_points is None else ns_points
    floor_distance = ctx["z_scan_floor"] if floor_distance is None else floor_distance
    zmax_factor = ctx["zmax_scan_factor"] if zmax_factor is None else zmax_factor
    similarity_threshold = ctx["similarity_threshold"] if similarity_threshold is None else similarity_threshold
    z_ref_mode = ctx["bessel_ref_mode"] if z_ref_mode is None else z_ref_mode
    z_ref_fraction = ctx["bessel_ref_fraction"] if z_ref_fraction is None else z_ref_fraction
    smooth_sigma_bins = ctx["smooth_sigma_bins"] if smooth_sigma_bins is None else smooth_sigma_bins
    min_prominence = ctx["min_peak_prominence"] if min_prominence is None else min_prominence
    radial_refine_factor = ctx["radial_refine_factor"] if radial_refine_factor is None else radial_refine_factor
    similarity_smooth_sigma = ctx["similarity_smooth_sigma"] if similarity_smooth_sigma is None else similarity_smooth_sigma

    row = run_standard_case(
        gs=gs,
        beam=beam,
        axicon=axicon,
        mask=mask,
        realism=realism,
        sim=sim,
        z_eval=z_eval,
        ns_points=ns_points,
        floor_distance=floor_distance,
        zmax_factor=zmax_factor,
        similarity_threshold=similarity_threshold,
        z_ref_mode=z_ref_mode,
        z_ref_fraction=z_ref_fraction,
        smooth_sigma_bins=smooth_sigma_bins,
        min_prominence=min_prominence,
        radial_refine_factor=radial_refine_factor,
        similarity_smooth_sigma=similarity_smooth_sigma,
        show_debug=show_debug,
    )
    return _canonicalize_row_metrics(row)


def run_parameter_sweep(ctx, parameter_name, values, ns_points=None, gs=None):
    """
    Generic sweep driver for gamma_deg, w0, or ell/l.

    parameter_name:
        - "gamma_deg"
        - "w0"
        - "ell" or "l"
    """
    rows = []
    for value in values:
        beam = ctx["beam_base"]
        axicon = ctx["axicon_base"]
        if parameter_name == "gamma_deg":
            axicon = replace(axicon, gamma_deg=float(value))
        elif parameter_name == "w0":
            beam = replace(beam, w0=float(value))
        elif parameter_name in ("ell", "l"):
            beam = replace(beam, ell=int(value))
        else:
            raise ValueError("parameter_name must be 'gamma_deg', 'w0', or 'ell'/'l'")
        row = run_case(ctx, beam=beam, axicon=axicon, ns_points=ns_points, gs=gs)
        rows.append(row)
    return rows


def enrich_with_axial_observables(
    row,
    ctx,
    use_fixed_annulus_from_eval: bool = True,
    annulus_mode: str = "fixed",
):
    """
    Add Richard-style first-ring axial observables to a row.

    Returns:
        axial: raw z-dependent traces and annulus metadata
        stability: compact stability summaries, including azimuthal ring nonuniformity
    """
    axial = axial_observables_from_case(
        row,
        smooth_sigma_bins=ctx["smooth_sigma_bins"],
        min_prominence=ctx["min_peak_prominence"],
        radial_refine_factor=ctx["radial_refine_factor"],
        use_fixed_annulus_from_eval=use_fixed_annulus_from_eval,
        annulus_mode=annulus_mode,
    )
    stability = axial_stability_summary(row, axial)
    uniformity = ring_azimuthal_nonuniformity(row["I_eval"], row["x"], axial["r_peak"], n_theta=360)
    stability["azimuthal_nonuniformity"] = uniformity
    stability["azimuthal_cv"] = uniformity
    ref_idx = int(np.argmin(np.abs(np.asarray(axial["z_values"], float) - float(row["z_eval"]))))
    I_eval = np.asarray(row["I_eval"], float)
    cy = int(I_eval.shape[0] // 2)
    cx = int(I_eval.shape[1] // 2)
    center_eval = float(I_eval[cy, cx])
    stability["center_to_ring_peak_ratio"] = center_eval / (float(axial["I1_peak"][ref_idx]) + 1e-30)
    row[f"axial_{annulus_mode}"] = axial
    row[f"axial_stability_{annulus_mode}"] = dict(stability)
    if annulus_mode == "fixed":
        row["axial"] = axial
        row["axial_stability"] = stability
        row.update(stability)
    else:
        for key, value in stability.items():
            row[f"{key}_{annulus_mode}"] = value
    return axial, stability


def run_realism_study(
    ctx,
    ell_values=None,
    pixel_pitch_scales=None,
    quantize_bits=None,
    ns_points=None,
    gs=None,
    floor_distance=None,
    zmax_factor=None,
    include_fill_factor: bool = False,
    include_active_area: bool = False,
):
    """
    Run the pixelation-vs-topological-charge study using scale factors.

    The scale factors are converted to physical pitches using
    `ctx["pixel_pitch_um_nominal"]`.
    """
    ell_values = ctx["l_values_realism"] if ell_values is None else list(ell_values)
    pixel_pitch_scales = ctx["pixel_pitch_scale_values"] if pixel_pitch_scales is None else list(pixel_pitch_scales)
    pitch_nom = float(ctx["pixel_pitch_um_nominal"]) * um
    pixel_pitch_values = [None if s is None else float(s) * pitch_nom for s in pixel_pitch_scales]
    ns_points = max(24, ctx["metric_ns_points"] // 2) if ns_points is None else ns_points
    gs = ctx["gs_metric"] if gs is None else gs
    floor_distance = (0.6 * ctx["z_scan_floor"]) if floor_distance is None else floor_distance
    zmax_factor = ctx["zmax_scan_factor"] if zmax_factor is None else zmax_factor

    rows = run_realism_grid(
        gs=gs,
        beam_base=ctx["beam_base"],
        axicon_base=ctx["axicon_base"],
        mask=ctx["mask"],
        realism_base=ctx["realism"],
        sim=ctx["sim"],
        z_eval=ctx["z_eval"],
        ns_points=ns_points,
        floor_distance=floor_distance,
        zmax_factor=zmax_factor,
        similarity_threshold=ctx["similarity_threshold"],
        z_ref_mode=ctx["bessel_ref_mode"],
        z_ref_fraction=ctx["bessel_ref_fraction"],
        smooth_sigma_bins=ctx["smooth_sigma_bins"],
        min_prominence=ctx["min_peak_prominence"],
        radial_refine_factor=ctx["radial_refine_factor"],
        similarity_smooth_sigma=ctx["similarity_smooth_sigma"],
        ell_values=ell_values,
        pixel_pitch_values=pixel_pitch_values,
        quantize_bits=quantize_bits,
        fill_factor=(ctx['slm_fill_factor'] if include_fill_factor else None),
        active_size=(ctx['slm_active_size'] if include_active_area else None),
    )
    for row in rows:
        _canonicalize_row_metrics(row)
        scale = None if row.get("effective_pixel_pitch") is None else float(row["effective_pixel_pitch"] / pitch_nom)
        row["pixel_pitch_scale"] = scale
        row["effective_pixel_pitch_um"] = None if row.get("effective_pixel_pitch") is None else float(row["effective_pixel_pitch"] / um)
    add_relative_to_ideal_metrics(
        rows,
        key_names=[
            "ring_diameter_mm",
            "ring_width_mm",
            "ring_spacing_mm",
            "bessel_length_mm",
            "epsilon_E",
            "epsilon_Ipeak",
            "azimuthal_cv",
        ],
    )
    return rows


def select_rows(rows, parameter_name, targets, atol=1e-12):
    """
    Notebook-friendly selection helper.

    Examples:
        select_rows(rows, "axicon.gamma_deg", [0.2, 0.75, 1.5])
        select_rows(rows, "beam.ell", [0, 5, 10, 20])
        select_rows(rows, "pixel_pitch_scale", [0.5, 1.0, 4.0, 8.0])
    """
    return rows_by_value(rows, parameter_name, targets, atol=atol)


def metric_array(rows, key):
    """Simple numeric array helper for line plots."""
    vals = []
    for row in rows:
        vals.append(row.get(key, np.nan))
    return np.asarray(vals, dtype=float)


def describe_original_slm_scaling(ctx):
    """
    Return a compact dictionary for the original beam-on-SLM sizing.

    This is useful for sanity checks when talking about SLM pitch.
    """
    w0 = float(ctx["beam_base"].w0)
    pitch_um = float(ctx["pixel_pitch_um_nominal"])
    return {
        "w0_mm": w0 / mm,
        "diameter_2w0_mm": 2.0 * w0 / mm,
        "diameter_1e_full_mm": 2.0 * w0 / mm,
        "radius_pixels_at_nominal_pitch": w0 / (pitch_um * um),
        "diameter_pixels_at_nominal_pitch": 2.0 * w0 / (pitch_um * um),
        "slm_active_size_mm": tuple(np.asarray(ctx["slm_active_size"]) / mm),
    }



# =============================================================================
# v2 device-faithful SLM and sampling diagnostics
# =============================================================================

SLM_DEVICE_DEFAULT = {
    "part_no": "HES 7020-16010-NIR-149",
    "display_type": "Reflective LCOS",
    "resolution": (1920, 1080),
    "wavelength_range_nm": (1000.0, 1100.0),
    "pixel_pitch": 8.0 * um,
    "fill_factor": 0.93,
    "active_size": (15.36 * mm, 8.64 * mm),
    "phase_bits": 8,
}

def attach_actual_slm_defaults(ctx, device: Optional[Dict[str, object]] = None):
    """Attach the actual SLM device specification to a context dict."""
    dev = dict(SLM_DEVICE_DEFAULT if device is None else device)
    ctx["slm_device"] = dev
    ctx["pixel_pitch_um_nominal"] = float(dev["pixel_pitch"] / um)
    ctx["slm_fill_factor"] = float(dev["fill_factor"])
    ctx["slm_phase_bits"] = int(dev["phase_bits"])
    ctx["slm_resolution"] = tuple(dev["resolution"])
    ctx["slm_active_size"] = tuple(dev["active_size"])
    return ctx


def make_integer_pitch_device_context(
    dx_um: float = 4.0,
    N: int = 2048,
    wavelength: float = 1030 * nm,
    w0: float = 2.0 * mm,
    ell: int = 3,
    gamma_deg: float = 1.0,
    quality_label: str = "paper",
):
    """
    Build a device-faithful context where the numerical sample pitch is an
    integer submultiple of the real 8 um SLM pixel pitch.

    Example:
        dx_um = 4 -> one real SLM pixel spans exactly 2 simulation samples.
    """
    dx = float(dx_um) * um
    L = 0.5 * int(N) * dx
    ctx = make_research_context(
        quality=quality_label,
        metric_quality=quality_label,
        showcase_quality=quality_label,
        use_showcase_quality=True,
        grid_half_width=L,
        wavelength=wavelength,
        w0=w0,
        ell=ell,
        gamma_deg=gamma_deg,
    )
    gs = GridSpec(N=int(N), L=float(L))
    ctx["gs"] = gs
    ctx["gs_metric"] = gs
    ctx["gs_showcase"] = gs
    ctx["gs_display"] = gs
    ctx["grid_half_width"] = float(L)
    ctx["sim_dx_um_target"] = float(dx_um)
    return attach_actual_slm_defaults(ctx)


def simulation_dx(gs: GridSpec) -> float:
    return float(2.0 * gs.L / gs.N)


def asm_critical_distance(gs: GridSpec, wavelength: float) -> float:
    dx = simulation_dx(gs)
    return float(gs.N * dx * dx / wavelength)


def axicon_cone_angle_rad(axicon: AxiconSpec, n_medium: float = 1.0) -> float:
    gamma = np.deg2rad(axicon.gamma_deg)
    return float((axicon.n_axicon - n_medium) * np.tan(gamma))


def bessel_sampling_limit_dx(beam: BeamSpec, axicon: AxiconSpec) -> Dict[str, float]:
    k = 2 * np.pi / beam.wavelength
    kr = compute_kr(k, axicon.n_axicon, beam.n_medium, np.deg2rad(axicon.gamma_deg), mode=axicon.kr_mode)
    fmax = float(kr / (2 * np.pi))
    dx_max = float(1.0 / (2.0 * fmax)) if fmax > 0 else float("inf")
    beta = axicon_cone_angle_rad(axicon, n_medium=beam.n_medium)
    return {
        "kr": float(kr),
        "fmax": float(fmax),
        "dx_max": float(dx_max),
        "beta_rad": float(beta),
        "beta_deg": float(np.rad2deg(beta)),
    }


def grid_vs_slm_report(ctx, gs: Optional[GridSpec] = None, beam: Optional[BeamSpec] = None, axicon: Optional[AxiconSpec] = None, pixel_pitch: Optional[float] = None) -> Dict[str, float]:
    """
    Compact numerical report tying together:
    - real SLM pitch / active area
    - current numerical pitch
    - ASM critical distance
    - Bessel / axicon Nyquist-style limit
    """
    gs = ctx["gs_metric"] if gs is None else gs
    beam = ctx["beam_base"] if beam is None else beam
    axicon = ctx["axicon_base"] if axicon is None else axicon
    attach_actual_slm_defaults(ctx)

    dx = simulation_dx(gs)
    zcrit = asm_critical_distance(gs, beam.wavelength)
    bessel = bessel_sampling_limit_dx(beam, axicon)
    pixel_pitch = ctx["slm_device"]["pixel_pitch"] if pixel_pitch is None else pixel_pitch

    ratio = float(pixel_pitch / dx) if pixel_pitch is not None and dx > 0 else float("inf")
    integer_multiple = bool(np.isfinite(ratio) and np.isclose(ratio, round(ratio), atol=1e-9))
    slm_resolved = bool(pixel_pitch is None or dx < pixel_pitch)
    slm_good = bool(pixel_pitch is None or dx <= 0.5 * pixel_pitch)
    bessel_safe = bool(dx <= bessel["dx_max"])

    if pixel_pitch is None:
        slm_label = "ideal"
    elif dx > pixel_pitch:
        slm_label = "unresolved"
    elif dx > 0.5 * pixel_pitch:
        slm_label = "marginal"
    else:
        slm_label = "resolved"

    if dx > bessel["dx_max"]:
        beam_label = "fail"
    elif dx > 0.75 * bessel["dx_max"]:
        beam_label = "marginal"
    else:
        beam_label = "pass"

    return {
        "sim_dx_um": float(dx / um),
        "sim_window_mm": float((2 * gs.L) / mm),
        "sim_N": int(gs.N),
        "asm_zcrit_mm": float(zcrit / mm),
        "beta_deg": float(bessel["beta_deg"]),
        "kr_per_m": float(bessel["kr"]),
        "fmax_lpmm": float(bessel["fmax"] / 1e3),
        "dx_max_bessel_um": float(bessel["dx_max"] / um),
        "pixel_pitch_um": None if pixel_pitch is None else float(pixel_pitch / um),
        "samples_per_slm_pixel": float(ratio),
        "slm_integer_multiple": integer_multiple,
        "slm_sampling_label": slm_label,
        "beam_sampling_label": beam_label,
        "slm_resolved": slm_resolved,
        "slm_good": slm_good,
        "bessel_safe": bessel_safe,
        "beam_radius_pixels_at_nominal_pitch": float(beam.w0 / ctx["slm_device"]["pixel_pitch"]),
        "beam_diameter_pixels_at_nominal_pitch": float(2.0 * beam.w0 / ctx["slm_device"]["pixel_pitch"]),
    }


def print_grid_vs_slm_report(report: Dict[str, float]) -> None:
    print("=== Grid / SLM / Bessel sampling report ===")
    print(f"simulation grid: N = {report['sim_N']}, dx = {report['sim_dx_um']:.3f} um, full window = {report['sim_window_mm']:.3f} mm")
    print(f"ASM critical distance ~ {report['asm_zcrit_mm']:.2f} mm")
    print(f"axicon cone beta ~ {report['beta_deg']:.4f} deg")
    print(f"Bessel sampling limit: dx <= {report['dx_max_bessel_um']:.3f} um  ({report['beam_sampling_label']})")
    if report["pixel_pitch_um"] is None:
        print("SLM pixel pitch: ideal / not applied")
    else:
        print(f"SLM pixel pitch = {report['pixel_pitch_um']:.3f} um, samples per SLM pixel = {report['samples_per_slm_pixel']:.3f}  ({report['slm_sampling_label']})")
        print(f"integer-multiple relationship p_SLM / dx = integer? {report['slm_integer_multiple']}")
    print(f"beam footprint at nominal 8 um pitch: radius ~ {report['beam_radius_pixels_at_nominal_pitch']:.1f} px, diameter ~ {report['beam_diameter_pixels_at_nominal_pitch']:.1f} px")


def _square_fill_mask(X: np.ndarray, Y: np.ndarray, pixel_pitch: float, fill_factor: float) -> np.ndarray:
    return slm_fill_factor_mask(X, Y, pixel_pitch, fill_factor)


def build_phase_realism_views(
    gs: GridSpec,
    beam: BeamSpec,
    axicon: AxiconSpec,
    mask: MaskSpec,
    pixel_pitch: Optional[float] = None,
    quantize_bits: Optional[int] = None,
    fill_factor: Optional[float] = None,
):
    """
    Build ideal / pixelated / quantized phase maps for direct inspection.
    """
    field = build_input_field(gs, beam)
    base_realism = SLMRealism(
        quantize_bits=None,
        pixel_pitch=None,
        carrier_lpmm=0.0,
        tilt_mrad=(0.0, 0.0),
        active_size=None,
    )
    slm_ideal = SLMPhase(beam=beam, axicon=axicon, mask=mask, realism=base_realism)
    phi_ideal = slm_ideal.phase_pattern(field)

    realism_pix = replace(base_realism, pixel_pitch=pixel_pitch, quantize_bits=None)
    phi_pixel = SLMPhase(beam=beam, axicon=axicon, mask=mask, realism=realism_pix).phase_pattern(field)

    realism_quant = replace(base_realism, pixel_pitch=pixel_pitch, quantize_bits=quantize_bits)
    phi_quant = SLMPhase(beam=beam, axicon=axicon, mask=mask, realism=realism_quant).phase_pattern(field)

    ff_mask = _square_fill_mask(field.grid.X, field.grid.Y, pixel_pitch, fill_factor) if pixel_pitch is not None else np.ones_like(phi_ideal, dtype=float)
    phi_quant_fill = np.where(ff_mask > 0.5, phi_quant, np.nan)

    def wrap_err(a, b):
        return np.angle(np.exp(1j * (a - b)))

    return {
        "x": field.grid.x,
        "grid": field.grid,
        "phi_ideal": phi_ideal,
        "phi_pixel": phi_pixel,
        "phi_quant": phi_quant,
        "phi_quant_fill": phi_quant_fill,
        "fill_mask": ff_mask,
        "phase_error_pixel": wrap_err(phi_pixel, phi_ideal),
        "phase_error_quant": wrap_err(phi_quant, phi_ideal),
    }


def _phase_display(phi: np.ndarray) -> np.ndarray:
    arr = np.asarray(phi, float).copy()
    finite = np.isfinite(arr)
    if np.any(finite):
        arr[finite] = np.mod(arr[finite], 2 * np.pi)
    return arr


def plot_phase_map_panel(
    views: Dict[str, np.ndarray],
    lim_mm: float = 0.8,
    title: str = "SLM phase realism",
) -> None:
    items = [
        ("phi_ideal", "Ideal wrapped phase", "phase"),
        ("phi_pixel", "Pixelated phase", "phase"),
        ("phi_quant", "Pixelated + quantized", "phase"),
        ("fill_mask", "Fill-factor mask", "mask"),
    ]

    fig, axes = plt.subplots(1, len(items), figsize=(4.0 * len(items), 4.2), constrained_layout=True)
    if len(items) == 1:
        axes = [axes]

    for ax, (key, ttl, kind) in zip(axes, items):
        if kind == "phase":
            V = _phase_display(views[key])
            im = ax.imshow(
                V,
                extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
                origin="lower",
                cmap="twilight",
                vmin=0.0,
                vmax=2 * np.pi,
                interpolation="nearest",
            )
        else:
            V = np.asarray(views[key], float)
            im = ax.imshow(
                V,
                extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
        ax.set_xlim(-lim_mm, lim_mm)
        ax.set_ylim(-lim_mm, lim_mm)
        ax.set_title(ttl)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.show()


def plot_fill_factor_zoom(
    views: Dict[str, np.ndarray],
    pixel_pitch: float,
    n_pixels: int = 12,
    title: str = "Fill-factor zoom",
) -> None:
    """
    Pixel-scale zoom on the quantized phase and anti-aliased fill mask.
    This is a diagnostic view; it should not be confused with the full-view panel.
    """
    lim_mm = 0.5 * n_pixels * pixel_pitch / mm

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), constrained_layout=True)

    Vphi = _phase_display(views["phi_quant"])
    im0 = axes[0].imshow(
        Vphi,
        extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
        origin="lower",
        cmap="twilight",
        vmin=0.0,
        vmax=2 * np.pi,
        interpolation="nearest",
    )
    axes[0].set_xlim(-lim_mm, lim_mm)
    axes[0].set_ylim(-lim_mm, lim_mm)
    axes[0].set_title("Quantized phase (zoom)")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    Vmask = np.asarray(views["fill_mask"], float)
    im1 = axes[1].imshow(
        Vmask,
        extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
        origin="lower",
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axes[1].set_xlim(-lim_mm, lim_mm)
    axes[1].set_ylim(-lim_mm, lim_mm)
    axes[1].set_title("Fill mask (zoom)")
    axes[1].set_xlabel("x [mm]")
    axes[1].set_ylabel("y [mm]")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.show()


def plot_phase_error_panel(
    views: Dict[str, np.ndarray],
    lim_mm: float = 0.8,
    title: str = "Phase-map error relative to ideal",
) -> None:
    items = [
        ("phase_error_pixel", "pixelated - ideal"),
        ("phase_error_quant", "quantized - ideal"),
        ("fill_mask", "fill-factor mask"),
    ]
    fig, axes = plt.subplots(1, len(items), figsize=(4.2 * len(items), 4.2), constrained_layout=True)
    if len(items) == 1:
        axes = [axes]
    for ax, (key, ttl) in zip(axes, items):
        V = np.asarray(views[key], float)
        if key == "fill_mask":
            im = ax.imshow(
                V,
                extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
                origin="lower",
                cmap="gray",
                interpolation="nearest",
                vmin=0.0,
                vmax=1.0,
            )
        else:
            im = ax.imshow(
                V,
                extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
                origin="lower",
                cmap="coolwarm",
                interpolation="nearest",
                vmin=-np.pi,
                vmax=np.pi,
            )
        ax.set_xlim(-lim_mm, lim_mm)
        ax.set_ylim(-lim_mm, lim_mm)
        ax.set_title(ttl)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.show()


def xy_display_view(I: np.ndarray, normalize: bool = True, gamma: float = 0.55) -> np.ndarray:
    """Display helper for beam-intensity plots."""
    view = np.maximum(np.asarray(I, float), 0.0)
    if normalize:
        view = view / (np.max(view) + 1e-15)
    if gamma != 1.0:
        view = np.power(view, gamma)
    return view


def xz_display_view(
    xs: np.ndarray,
    normalize: bool = True,
    gamma: float = 0.60,
    upscale_z: float = 3.0,
    upscale_x: float = 1.5,
    order: int = 3,
) -> np.ndarray:
    """
    Display-only view of an x-z intensity map.

    The stored scalar data remain untouched.  Any interpolation here is purely
    for presentation so the Bessel-region panels do not look artificially
    blocky when the z scan is intentionally kept modest for metric speed.
    """
    view = np.maximum(np.asarray(xs, float), 0.0)
    if (float(upscale_x) != 1.0) or (float(upscale_z) != 1.0):
        view = zoom(
            view,
            zoom=(max(float(upscale_x), 1.0), max(float(upscale_z), 1.0)),
            order=int(order),
            mode="nearest",
            prefilter=bool(order > 1),
        )
        view = np.maximum(view, 0.0)
    if normalize:
        view = view / (np.nanmax(view) + 1e-15)
    if gamma != 1.0:
        view = np.power(view, gamma)
    return np.nan_to_num(view, nan=0.0, posinf=1.0, neginf=0.0)


def common_xy_limit_mm(
    rows: List[Dict[str, object]],
    energy_frac: float = 0.997,
    pad: float = 1.12,
    min_mm: float = 0.35,
    max_mm: float = 1.30,
) -> float:
    """Common XY crop across a sweep, expressed in physical millimetres."""
    if not rows:
        return float(min_mm)
    images = [np.asarray(row["I_eval"], float) for row in rows if "I_eval" in row]
    if not images:
        return float(min_mm)
    return auto_crop_mm_from_images(
        images,
        rows[0]["x"],
        energy_frac=energy_frac,
        pad=pad,
        min_mm=min_mm,
        max_mm=max_mm,
    )


def common_xz_limit_mm(
    rows: List[Dict[str, object]],
    pad: float = 1.18,
    min_mm: float = 0.35,
    max_mm: float = 1.30,
) -> float:
    """
    Common transverse x-limit for x-z maps.

    The limit is based on the larger of an image-energy crop and a ring-metric
    estimate so the first ring stays visible with honest physical axes.
    """
    if not rows:
        return float(min_mm)

    lim_from_xy = common_xy_limit_mm(rows, pad=pad, min_mm=min_mm, max_mm=max_mm)
    ring_based = []
    for row in rows:
        diameter = float(row.get("ring_diameter", np.nan))
        width = float(row.get("ring_width", np.nan))
        if not np.isfinite(diameter):
            diameter = float(row.get("ring_diameter_mm", np.nan)) * mm
        if not np.isfinite(width):
            width = float(row.get("ring_width_mm", np.nan)) * mm
        if np.isfinite(diameter) and diameter > 0.0:
            radius = 0.5 * diameter
            extra = 2.5 * width if np.isfinite(width) and width > 0.0 else 0.18 * radius
            ring_based.append((radius + extra) / mm)

    if ring_based:
        lim_from_ring = float(np.clip(pad * max(ring_based), min_mm, max_mm))
        return max(lim_from_xy, lim_from_ring)
    return lim_from_xy


def plot_phase_and_output_panel(
    views: Dict[str, np.ndarray],
    output_rows: List[Dict[str, object]],
    phase_titles: List[str],
    beam_titles: List[str],
    lim_mm: float = 0.8,
    z_eval_mm: Optional[float] = None,
    title: str = "Phase masks and resulting beam outputs",
) -> None:
    """Compact meeting-friendly top/bottom comparison panel."""
    default_phase_keys = ["phi_ideal", "phi_pixel", "phi_quant", "fill_mask"]
    phase_keys = default_phase_keys[:len(phase_titles)]
    n = max(len(phase_keys), len(output_rows), 1)

    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 7.0), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(2, 1)

    for j in range(n):
        ax = axes[0, j]
        if j < len(phase_keys):
            key = phase_keys[j]
            if key == "fill_mask":
                V = np.asarray(views[key], float)
                ax.imshow(
                    V,
                    extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
                    origin="lower",
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
            else:
                V = _phase_display(views[key])
                ax.imshow(
                    V,
                    extent=[views["x"][0] / mm, views["x"][-1] / mm, views["x"][0] / mm, views["x"][-1] / mm],
                    origin="lower",
                    cmap="twilight",
                    vmin=0.0,
                    vmax=2 * np.pi,
                    interpolation="nearest",
                )
            ax.set_title(phase_titles[j])
            ax.set_xlim(-lim_mm, lim_mm)
            ax.set_ylim(-lim_mm, lim_mm)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
        else:
            ax.axis("off")

    for j in range(n):
        ax = axes[1, j]
        if j < len(output_rows):
            row = output_rows[j]
            V = xy_display_view(row["I_eval"], normalize=True)
            ax.imshow(
                V,
                extent=[row["x"][0] / mm, row["x"][-1] / mm, row["x"][0] / mm, row["x"][-1] / mm],
                origin="lower",
                cmap="hot",
                vmin=0.0,
                vmax=1.0,
                interpolation="spline36",
            )
            if j < len(beam_titles):
                ax.set_title(beam_titles[j])
            ax.set_xlim(-lim_mm, lim_mm)
            ax.set_ylim(-lim_mm, lim_mm)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
        else:
            ax.axis("off")

    fig.suptitle(title if z_eval_mm is None else f"{title} at z = {z_eval_mm:.1f} mm")
    plt.show()

def sampling_validity_label(report: Dict[str, float], phase_report: Optional[Dict[str, float]] = None) -> str:
    labels = [report.get('beam_sampling_label', 'pass'), report.get('slm_sampling_label', 'resolved')]
    if phase_report is not None:
        labels.extend([phase_report.get('sim_phase_label', 'pass'), phase_report.get('slm_phase_label', 'pass')])
    if any(lbl in ('fail', 'unresolved') for lbl in labels):
        return 'fail'
    if any(lbl in ('marginal',) for lbl in labels):
        return 'marginal'
    return 'pass'



def make_actual_device_realism(ctx, pixel_pitch: Optional[float] = None, quantize_bits: Optional[int] = None, include_fill_factor: bool = True, include_active_area: bool = True) -> SLMRealism:
    """Return an SLMRealism object matching the actual device by default."""
    attach_actual_slm_defaults(ctx)
    dev = ctx["slm_device"]
    return replace(
        ctx["realism"],
        pixel_pitch=dev["pixel_pitch"] if pixel_pitch is None else pixel_pitch,
        quantize_bits=dev["phase_bits"] if quantize_bits is None else quantize_bits,
        active_size=dev["active_size"] if include_active_area else None,
        fill_factor=(dev["fill_factor"] if include_fill_factor else None),
    )





def device_phase_gradient_map(phi: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """Estimate local spatial frequency from the wrapped phase gradient.

    Returns cycles/m along x and y and the total magnitude.
    """
    phi = np.asarray(phi, float)
    dphix = np.angle(np.exp(1j * (np.roll(phi, -1, axis=1) - phi))) / float(dx)
    dphiy = np.angle(np.exp(1j * (np.roll(phi, -1, axis=0) - phi))) / float(dx)
    fx = np.abs(dphix) / (2 * np.pi)
    fy = np.abs(dphiy) / (2 * np.pi)
    fmag = np.sqrt(fx ** 2 + fy ** 2)
    return {"fx": fx, "fy": fy, "fmag": fmag}





def phase_sampling_report(views: Dict[str, np.ndarray], dx: float, pixel_pitch: Optional[float] = None, active_size: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """Report how steep the phase map is relative to simulation and SLM Nyquist limits."""
    grad = device_phase_gradient_map(views["phi_ideal"], dx)
    fmag = grad["fmag"]
    if active_size is not None:
        X = views["grid"].X
        Y = views["grid"].Y
        sx, sy = active_size
        active_mask = (np.abs(X) <= 0.5 * sx) & (np.abs(Y) <= 0.5 * sy)
    else:
        active_mask = np.ones_like(fmag, dtype=bool)
    vals = fmag[active_mask]
    if vals.size == 0:
        vals = fmag.ravel()
    fmax_local = float(np.nanmax(vals))
    f95_local = float(np.nanpercentile(vals, 95))
    sim_nyquist = float(1.0 / (2.0 * dx))
    slm_nyquist = None if pixel_pitch is None else float(1.0 / (2.0 * pixel_pitch))
    frac_sim_exceed = float(np.mean(vals > sim_nyquist))
    frac_slm_exceed = float(np.mean(vals > slm_nyquist)) if slm_nyquist is not None else 0.0
    if frac_sim_exceed > 0.02:
        sim_label = "fail"
    elif frac_sim_exceed > 0.0 or f95_local > 0.9 * sim_nyquist:
        sim_label = "marginal"
    else:
        sim_label = "pass"
    if slm_nyquist is None:
        slm_label = "ideal"
    elif frac_slm_exceed > 0.02:
        slm_label = "fail"
    elif frac_slm_exceed > 0.0 or f95_local > 0.9 * slm_nyquist:
        slm_label = "marginal"
    else:
        slm_label = "pass"
    return {
        "fmax_local_lpmm": fmax_local / 1e3,
        "f95_local_lpmm": f95_local / 1e3,
        "sim_nyquist_lpmm": sim_nyquist / 1e3,
        "slm_nyquist_lpmm": None if slm_nyquist is None else slm_nyquist / 1e3,
        "frac_sim_exceed": frac_sim_exceed,
        "frac_slm_exceed": frac_slm_exceed,
        "sim_phase_label": sim_label,
        "slm_phase_label": slm_label,
    }





def print_phase_sampling_report(report: Dict[str, float]) -> None:
    print("=== Phase-map sampling report ===")
    print(f"local phase-gradient frequency: max = {report['fmax_local_lpmm']:.2f} lp/mm, p95 = {report['f95_local_lpmm']:.2f} lp/mm")
    print(f"simulation Nyquist = {report['sim_nyquist_lpmm']:.2f} lp/mm  ({report['sim_phase_label']})")
    if report['slm_nyquist_lpmm'] is None:
        print("SLM Nyquist: ideal / not applied")
    else:
        print(f"SLM Nyquist = {report['slm_nyquist_lpmm']:.2f} lp/mm  ({report['slm_phase_label']})")
        print(f"fraction above SLM Nyquist = {100.0 * report['frac_slm_exceed']:.3f}%")





def plot_phase_gradient_panel(views: Dict[str, np.ndarray], pixel_pitch: Optional[float], lim_mm: float = 0.8, title: str = "Phase-gradient sampling map") -> None:
    grad = device_phase_gradient_map(views['phi_ideal'], views['grid'].dx)
    sim_ny = 1.0 / (2.0 * views['grid'].dx)
    slm_ny = None if pixel_pitch is None else 1.0 / (2.0 * pixel_pitch)
    maps = [
        (grad['fmag'] / sim_ny, r"$f_{local}/f_{Nyq,sim}$"),
        (np.where(np.isfinite(views['fill_mask']), views['fill_mask'], np.nan), "fill mask"),
    ]
    if slm_ny is not None:
        maps.insert(1, (grad['fmag'] / slm_ny, r"$f_{local}/f_{Nyq,SLM}$"))
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 4.1), constrained_layout=True)
    if len(maps) == 1:
        axes = [axes]
    for ax, (M, ttl) in zip(axes, maps):
        if 'fill mask' in ttl:
            im = ax.imshow(M, extent=[views['x'][0]/mm, views['x'][-1]/mm, views['x'][0]/mm, views['x'][-1]/mm], origin='lower', cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        else:
            im = ax.imshow(M, extent=[views['x'][0]/mm, views['x'][-1]/mm, views['x'][0]/mm, views['x'][-1]/mm], origin='lower', cmap='magma', vmin=0, vmax=max(1.0, float(np.nanpercentile(M, 99))), interpolation='nearest')
            ax.contour(views['x']/mm, views['x']/mm, M, levels=[1.0], colors='cyan', linewidths=1.0)
        ax.set_xlim(-lim_mm, lim_mm)
        ax.set_ylim(-lim_mm, lim_mm)
        ax.set_title(ttl)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    plt.show()






def run_actual_device_case(
    ctx,
    beam=None,
    axicon=None,
    gs=None,
    ns_points=None,
    include_fill_factor: bool = True,
    include_active_area: bool = True,
    quantize_bits: Optional[int] = None,
    show_debug: bool = False,
):
    """Run one case using the actual 8 um / 93% / 8-bit device as baseline."""
    beam = ctx['beam_base'] if beam is None else beam
    axicon = ctx['axicon_base'] if axicon is None else axicon
    gs = ctx['gs_metric'] if gs is None else gs
    realism = make_actual_device_realism(
        ctx,
        quantize_bits=quantize_bits,
        include_fill_factor=include_fill_factor,
        include_active_area=include_active_area,
    )
    row = run_case(ctx, beam=beam, axicon=axicon, realism=realism, gs=gs, ns_points=ns_points, show_debug=show_debug)
    row['device_mode'] = 'actual'
    row['pixel_pitch_scale'] = 1.0
    row['effective_pixel_pitch'] = realism.pixel_pitch
    row['effective_pixel_pitch_um'] = float(realism.pixel_pitch / um)
    row['fill_factor'] = realism.fill_factor
    row['quantize_bits'] = realism.quantize_bits
    row['active_size'] = realism.active_size
    return row



def run_actual_device_sweep(
    ctx,
    vary: str,
    values,
    gs=None,
    ns_points=None,
    include_fill_factor: bool = True,
    include_active_area: bool = True,
    quantize_bits: Optional[int] = None,
):
    rows = []
    for value in values:
        beam = ctx['beam_base']
        axicon = ctx['axicon_base']
        if vary in ('ell', 'l'):
            beam = replace(beam, ell=int(value))
        elif vary == 'gamma_deg':
            axicon = replace(axicon, gamma_deg=float(value))
        elif vary == 'w0':
            beam = replace(beam, w0=float(value))
        else:
            raise ValueError("vary must be 'ell'/'l', 'gamma_deg', or 'w0'")
        rows.append(
            run_actual_device_case(
                ctx,
                beam=beam,
                axicon=axicon,
                gs=gs,
                ns_points=ns_points,
                include_fill_factor=include_fill_factor,
                include_active_area=include_active_area,
                quantize_bits=quantize_bits,
            )
        )
    return rows


def axial_onaxis_intensity_from_case(row: Dict[str, object]) -> Dict[str, object]:
    """
    Return the on-axis intensity trace for one propagated case.

    This uses the stored x-z slice when available, so it is consistent with the
    scan already used for Bessel-region metrics and does not require a second
    propagation pass.
    """
    z_values = np.asarray(row["z_values"], float)
    if "xs" in row:
        xs = np.asarray(row["xs"], float)
        x_mid = int(xs.shape[0] // 2)
        onaxis = xs[x_mid, :].astype(float, copy=False)
    else:
        propagator = row["propagator"]
        grid = row["grid"]
        c = int(grid.N // 2)
        onaxis = np.zeros_like(z_values, dtype=float)
        for i, zi in enumerate(z_values):
            Uz = propagator(float(zi)).U
            onaxis[i] = float(np.abs(Uz[c, c]) ** 2)

    ref_idx = int(np.argmin(np.abs(z_values - float(row["z_eval"]))))
    onaxis_eval = float(onaxis[ref_idx])
    onaxis_max = float(np.nanmax(onaxis)) if onaxis.size else float("nan")
    return {
        "z_values": z_values,
        "z_mm": z_values / mm,
        "z_ref": float(row["z_eval"]),
        "ref_index": ref_idx,
        "intensity": onaxis,
        "normalized": onaxis / (onaxis_eval + 1e-30),
        "normalized_to_peak": onaxis / (onaxis_max + 1e-30),
        "eval_intensity": onaxis_eval,
        "max_intensity": onaxis_max,
    }


def run_actual_device_capability_sweep(
    ctx,
    ell_values: Optional[List[int]] = None,
    gs=None,
    ns_points: Optional[int] = None,
    include_fill_factor: bool = True,
    include_active_area: bool = True,
    quantize_bits: Optional[int] = None,
    use_fixed_annulus_from_eval: bool = True,
    include_tracked_annulus: bool = True,
) -> List[Dict[str, object]]:
    """
    Sweep topological charge for the real SLM device using the current scalar engine.

    Fixed-annulus Richard observables remain the main reported metrics.
    Tracked-annulus observables are attached separately for diagnostic
    comparison so metric-window fragility can be separated from genuine beam
    degradation.
    """
    attach_actual_slm_defaults(ctx)
    ell_values = list(range(26)) if ell_values is None else [int(v) for v in ell_values]
    gs = ctx["gs_metric"] if gs is None else gs
    ns_points = ctx["metric_ns_points"] if ns_points is None else int(ns_points)

    rows = []
    for ell in ell_values:
        beam = replace(ctx["beam_base"], ell=int(ell))
        row = run_actual_device_case(
            ctx,
            beam=beam,
            gs=gs,
            ns_points=ns_points,
            include_fill_factor=include_fill_factor,
            include_active_area=include_active_area,
            quantize_bits=quantize_bits,
        )
        row["device_mode"] = "actual_capability"
        row["sampling_report"] = grid_vs_slm_report(
            ctx,
            gs=gs,
            beam=beam,
            axicon=row["axicon"],
            pixel_pitch=row.get("effective_pixel_pitch"),
        )

        axial_fixed, stability_fixed = enrich_with_axial_observables(
            row,
            ctx,
            use_fixed_annulus_from_eval=use_fixed_annulus_from_eval,
            annulus_mode="fixed",
        )
        row["axial_fixed"] = axial_fixed
        row["axial_stability_fixed"] = dict(stability_fixed)

        if include_tracked_annulus:
            axial_tracked, stability_tracked = enrich_with_axial_observables(
                row,
                ctx,
                use_fixed_annulus_from_eval=use_fixed_annulus_from_eval,
                annulus_mode="tracked",
            )
            row["axial_tracked"] = axial_tracked
            row["axial_stability_tracked"] = dict(stability_tracked)

        onaxis = axial_onaxis_intensity_from_case(row)
        row["onaxis"] = onaxis
        row["sampling_feasibility"] = sampling_feasibility_metric(
            axial_fixed["r_peak"],
            row["beam"].ell,
            row.get("effective_pixel_pitch"),
        )
        ref_idx = int(onaxis["ref_index"])
        row["onaxis_eval_norm_to_peak"] = float(onaxis["normalized_to_peak"][ref_idx])
        row["onaxis_peak_norm_max"] = 1.0 if onaxis["intensity"].size else float("nan")
        center_eval = float(np.asarray(row["I_eval"], float)[row["I_eval"].shape[0] // 2, row["I_eval"].shape[1] // 2])
        row["center_to_ring_peak_ratio"] = center_eval / (float(axial_fixed["I1_peak"][ref_idx]) + 1e-30)
        row["usable_bessel_collapsed"] = not bool(row.get("axial_interval_valid", False))
        rows.append(row)

    return rows
