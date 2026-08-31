"""Physics-specific observables for fitting the Vortex-Bessel digital twin.

A multi-plane camera stack contains several qualitatively different pieces of
information.  Treating every optical parameter with one generic pixelwise loss
creates avoidable degeneracies: an unknown higher-order phase can otherwise be
mislabelled as a translation, and independent plane normalization removes the
longitudinal illumination information needed to estimate beam size.

This module provides reusable observables that preserve the symmetry and
propagation information associated with different physical parameters:

``longitudinal_brightness_envelope``
    Relative Bessel-region brightness versus z, with one global normalization.
    Useful for quantities such as Gaussian illumination radius that control how
    successive stationary-phase annuli are weighted.  Experimental use requires
    fixed camera gain/exposure across the scan.

``axisymmetric_radial_morphology``
    Azimuthally averaged transverse intensity after removal of translation and
    independent plane normalization.  Useful for aperture/ring geometry while
    marginalizing non-axisymmetric phase.

``centroid_trajectory``
    First spatial moment versus z.  Useful for lateral alignment/steering terms
    and intentionally insensitive to even/higher-order angular distortions.

The functions return deterministic scalar losses compatible with
``hierarchical_physical_fit``.  They are observables, not statistical confidence
models; uncertainty/identifiability must still be assessed from parameter and
family separation or dedicated inference runs.
"""
from __future__ import annotations

import numpy as np

EPS = np.finfo(float).tiny


def threshold_weights(image: np.ndarray, *, floor_fraction: float = 0.01,
                      exponent: float = 1.35) -> np.ndarray:
    a=np.maximum(np.asarray(image,float),0.0)
    a=a/max(float(np.max(a)),EPS)
    floor=float(np.clip(floor_fraction,0.0,0.95))
    w=np.clip((a-floor)/max(1.0-floor,EPS),0.0,1.0)
    return w**float(exponent)


def intensity_centroid(image: np.ndarray) -> tuple[float,float]:
    w=threshold_weights(image)
    yy,xx=np.indices(w.shape,dtype=float)
    s=max(float(np.sum(w)),EPS)
    return float(np.sum(yy*w)/s),float(np.sum(xx*w)/s)


def radial_profile_about_centroid(image: np.ndarray, *, bins: int=72) -> np.ndarray:
    a=np.maximum(np.asarray(image,float),0.0)
    a=a/max(float(np.max(a)),EPS)
    cy,cx=intensity_centroid(a)
    yy,xx=np.indices(a.shape,dtype=float)
    rr=np.hypot(yy-cy,xx-cx)
    rmax=0.47*min(a.shape)
    edges=np.linspace(0.0,rmax,int(bins)+1)
    ids=np.clip(np.digitize(rr.ravel(),edges)-1,0,int(bins)-1)
    good=rr.ravel()<=rmax
    sums=np.bincount(ids[good],weights=a.ravel()[good],minlength=int(bins))
    nums=np.bincount(ids[good],minlength=int(bins))
    p=sums/np.maximum(nums,1)
    return p/max(float(np.max(p)),EPS)


def axisymmetric_radial_morphology(model: np.ndarray, data: np.ndarray,
                                    *, bins: int=72) -> float:
    """RMSE of centered azimuthal radial profiles across the z stack."""
    m=np.asarray(model,float); d=np.asarray(data,float)
    if m.shape!=d.shape or m.ndim!=3:
        raise ValueError('model and data must have matching (z,y,x) shape')
    mp=np.stack([radial_profile_about_centroid(p,bins=bins) for p in m])
    dp=np.stack([radial_profile_about_centroid(p,bins=bins) for p in d])
    return float(np.sqrt(np.mean((mp-dp)**2)))


def centroid_trajectory(model: np.ndarray, data: np.ndarray,
                        *, derivative_weight: float=0.35,
                        position_scale_fraction: float=0.15) -> float:
    """Compare absolute centroid position and its change along the z scan."""
    m=np.asarray(model,float); d=np.asarray(data,float)
    if m.shape!=d.shape or m.ndim!=3:
        raise ValueError('model and data must have matching (z,y,x) shape')
    mc=np.asarray([intensity_centroid(p) for p in m],float)
    dc=np.asarray([intensity_centroid(p) for p in d],float)
    scale=max(float(position_scale_fraction)*min(m.shape[-2:]),EPS)
    pos=float(np.sqrt(np.mean(((mc-dc)/scale)**2)))
    slope=0.0
    if len(mc)>2:
        slope=float(np.sqrt(np.mean(((np.diff(mc,axis=0)-np.diff(dc,axis=0))/scale)**2)))
    return float(pos+float(derivative_weight)*slope)


def longitudinal_brightness_envelope(stack: np.ndarray, *, top_fraction: float=0.012) -> np.ndarray:
    """Robust bright-field envelope versus z with one normalization for the scan.

    The mean of the brightest small pixel fraction tracks the q-vortex Bessel
    ring brightness more robustly than one peak pixel.  No plane-wise
    normalization is performed before the envelope is calculated.
    """
    a=np.maximum(np.asarray(stack,float),0.0)
    if a.ndim!=3:
        raise ValueError('stack must have shape (z,y,x)')
    frac=float(top_fraction)
    if not (0.0<frac<0.5): raise ValueError('top_fraction must lie between 0 and 0.5')
    k=max(8,int(round(frac*a.shape[-1]*a.shape[-2])))
    env=[]
    for plane in a:
        f=plane.ravel(); hi=np.partition(f,f.size-k)[-k:]
        env.append(float(np.mean(hi)))
    env=np.asarray(env,float)
    return env/max(float(np.max(env)),EPS)


def longitudinal_envelope_rmse(model: np.ndarray, data: np.ndarray,
                               *, top_fraction: float=0.012) -> float:
    """RMSE between globally normalized Bessel brightness envelopes versus z."""
    m=longitudinal_brightness_envelope(model,top_fraction=top_fraction)
    d=longitudinal_brightness_envelope(data,top_fraction=top_fraction)
    if m.shape!=d.shape: raise ValueError('model/data z dimensions differ')
    return float(np.sqrt(np.mean((m-d)**2)))
