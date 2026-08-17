from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage, optimize, special

EPS=1e-12

@dataclass
class PlaneFit:
    z_index:int
    q:int
    kr_m_inv:float
    center_y_px:float
    center_x_px:float
    ring_radius_px:float
    core_score:float
    m_values:np.ndarray
    coeffs:np.ndarray
    loss:float
    corr:float
    nrmse:float
    phase_only_corr:float
    phase_only_nrmse:float
    dark_core_before:float
    dark_core_after:float
    azimuth_cv_before:float
    azimuth_cv_after:float


def read_bmg(path:Path, shape=(2048,2048)) -> np.ndarray:
    raw=path.read_bytes(); n=int(np.prod(shape)); payload=n*4; off=len(raw)-payload
    if off<=0: raise ValueError(f'{path}: invalid size')
    return np.frombuffer(raw,dtype='<i4',count=n,offset=off).reshape(shape).copy()


def preprocess(img:np.ndarray, background_percentile=35.0) -> np.ndarray:
    bg=float(np.percentile(img,background_percentile)); out=img.astype(np.float32)-bg; out[out<0]=0
    return out


def find_dark_core_center(img:np.ndarray, search_half=30) -> tuple[float,float,float]:
    """Find a local annular centre near the brightest compact feature.

    Score favours a dark central disk surrounded by a bright small annulus. It is deliberately
    local so outer axicon/Fresnel rings do not pull the centre estimate.
    """
    sm=ndimage.gaussian_filter(img,1.2)
    py,px=np.unravel_index(np.argmax(sm),sm.shape)
    y0=max(0,py-45); y1=min(img.shape[0],py+46); x0=max(0,px-45); x1=min(img.shape[1],px+46)
    p=sm[y0:y1,x0:x1]; yy,xx=np.indices(p.shape)
    best=(-np.inf,(float(py),float(px)))
    cy0=py-y0; cx0=px-x0
    for cy in range(max(20,cy0-search_half), min(p.shape[0]-20,cy0+search_half+1)):
        for cx in range(max(20,cx0-search_half), min(p.shape[1]-20,cx0+search_half+1)):
            rr=np.hypot(xx-cx,yy-cy)
            centre=float(p[rr<4].mean())
            ann=float(p[(rr>=7)&(rr<=20)].mean())
            score=(ann-centre)/max(ann,EPS)
            if score>best[0]: best=(score,(float(y0+cy),float(x0+cx)))
    return best[1][0],best[1][1],float(best[0])


def radial_profile(img,cy,cx,rmax_px=80.0,dr=.5):
    h=int(np.ceil(rmax_px+3)); yi=int(round(cy)); xi=int(round(cx))
    y0=max(0,yi-h); y1=min(img.shape[0],yi+h+1); x0=max(0,xi-h); x1=min(img.shape[1],xi+h+1)
    p=img[y0:y1,x0:x1]; yy,xx=np.indices(p.shape); rr=np.hypot(xx-(cx-x0),yy-(cy-y0))
    bins=np.arange(0,rmax_px+dr,dr); idx=np.digitize(rr.ravel(),bins)
    sums=np.bincount(idx,weights=p.ravel(),minlength=len(bins)+1); nums=np.bincount(idx,minlength=len(bins)+1)
    prof=sums[1:len(bins)]/np.maximum(nums[1:len(bins)],1); r=.5*(bins[:-1]+bins[1:])
    return r,prof


def principal_inner_ring(img,cy,cx,min_px=5,max_px=30):
    r,p=radial_profile(img,cy,cx,rmax_px=max_px+5,dr=.5); mask=(r>=min_px)&(r<=max_px)
    rp=float(r[mask][np.argmax(p[mask])]); return rp,r,p


def sample_polar(img,cy,cx,radii_px,theta):
    rr,tt=np.meshgrid(np.asarray(radii_px,float),np.asarray(theta,float),indexing='ij')
    ys=cy+rr*np.sin(tt); xs=cx+rr*np.cos(tt)
    return ndimage.map_coordinates(img,[ys,xs],order=1,mode='constant',cval=0.0)


def estimate_global_kr(images,pixel_m,q=20,quality_threshold=.55):
    rows=[]
    jprime=float(special.jnp_zeros(abs(q),1)[0])
    for zi,img in enumerate(images):
        cy,cx,score=find_dark_core_center(img); rp,_,_=principal_inner_ring(img,cy,cx)
        kr=jprime/(rp*pixel_m)
        rows.append(dict(z_index=zi,cy=cy,cx=cx,core_score=score,ring_radius_px=rp,kr_m_inv=kr))
    df=pd.DataFrame(rows)
    good=df.core_score>=quality_threshold
    if good.sum()<2: good=np.ones(len(df),dtype=bool)
    # robust median; a fixed cone should yield a near-constant principal Jq ring radius
    kr=float(np.median(df.loc[good,'kr_m_inv']))
    return kr,df


def modal_basis(q:int, m_values:np.ndarray, kr_m_inv:float, r_m:np.ndarray, phi:np.ndarray):
    """Eq. (3) basis from Miao et al., indexed by aberration harmonic m=n+q.

    Ideal q-th order Bessel corresponds to m=0 -> n=-q.
    """
    n_values=np.asarray(m_values,int)-int(q)
    cols=[]
    for n in n_values:
        cols.append(((-1j)**n)*special.jv(n,kr_m_inv*r_m)*np.exp(-1j*n*phi))
    return np.column_stack(cols),n_values


def _pack_coeffs(c,m_values):
    i0=int(np.where(m_values==0)[0][0]); out=[np.log(max(float(np.real(c[i0])),1e-6))]
    for i in range(len(c)):
        if i==i0: continue
        out.extend([float(np.real(c[i])),float(np.imag(c[i]))])
    return np.asarray(out)


def _unpack_coeffs(x,m_values):
    i0=int(np.where(m_values==0)[0][0]); c=np.zeros(len(m_values),complex); c[i0]=np.exp(x[0]); k=1
    for i in range(len(c)):
        if i==i0: continue
        c[i]=x[k]+1j*x[k+1]; k+=2
    return c


def fit_coefficients(B,y,w,m_values,maxiter=90,reg=2e-4):
    """Fit complex modal coefficients to intensity with an analytic gradient.

    The m=0 (ideal q-mode) coefficient is constrained real-positive to remove the
    unobservable global-phase degree of freedom.
    """
    y=np.asarray(y,float); w=np.asarray(w,float); y=y/max(np.max(y),EPS)
    den=max(float(np.sum(w*y*y)),EPS)
    i0=int(np.where(m_values==0)[0][0])
    # amplitude initialization from weighted least-squares against ideal mode intensity
    p0=np.abs(B[:,i0])**2
    a2=float(np.sum(w*p0*y)/max(np.sum(w*p0*p0),EPS))
    c0=np.zeros(B.shape[1],complex); c0[i0]=np.sqrt(max(a2,1e-8))
    x0=_pack_coeffs(c0,m_values)

    def fg(x):
        c=_unpack_coeffs(x,m_values)
        u=B@c; pred=np.abs(u)**2; r=pred-y
        loss=float(np.sum(w*r*r)/den)
        z=(w*r)*np.conj(u)
        # d loss / d Re(c_j), d Im(c_j)
        ga=4.0*np.real(z @ B)/den
        gb=-4.0*np.imag(z @ B)/den

        idx=np.arange(len(c))!=i0
        c0r=max(float(np.real(c[i0])),1e-12)
        S=float(np.sum(np.abs(c[idx])**2))
        R=reg*S/(c0r*c0r)
        loss+=R
        ga[idx]+=2*reg*np.real(c[idx])/(c0r*c0r)
        gb[idx]+=2*reg*np.imag(c[idx])/(c0r*c0r)

        grad=np.empty_like(x)
        # x[0]=log(c0), chain rule; reg contributes -2R wrt log(c0)
        grad[0]=ga[i0]*c0r - 2*R
        k=1
        for j in range(len(c)):
            if j==i0: continue
            grad[k]=ga[j]; grad[k+1]=gb[j]; k+=2
        return loss,grad

    res=optimize.minimize(lambda x: fg(x)[0],x0,jac=lambda x: fg(x)[1],method='L-BFGS-B',
                          options={'maxiter':maxiter,'ftol':1e-12,'gtol':1e-7,'maxls':40})
    c=_unpack_coeffs(res.x,m_values)
    return c,float(res.fun),res

def aberration_field_theta(coeffs,m_values,theta):
    # c_{m-q} = integral g(theta) exp(i m theta) dtheta, so inverse series g ~ sum c_m exp(-im theta)
    g=np.zeros_like(theta,dtype=complex)
    for c,m in zip(coeffs,m_values): g+=c*np.exp(-1j*m*theta)
    return g


def corrected_coefficients_phase_only(coeffs,m_values,n_theta=2048):
    th=np.linspace(0,2*np.pi,n_theta,endpoint=False); g=aberration_field_theta(coeffs,m_values,th)
    gc=np.abs(g) # ideal phase-only precompensation removes arg(g), amplitude nonuniformity remains
    # forward coefficients c_m = integral g exp(i m theta) dtheta (common scalar factor irrelevant)
    return np.asarray([np.mean(gc*np.exp(1j*m*th)) for m in m_values],complex),th,g


def image_metrics(I,r_m,phi,ring_r_m):
    I=np.asarray(I,float); ring=np.abs(r_m-ring_r_m)<=max(0.15*ring_r_m,5e-6)
    vals=I[ring]; azcv=float(np.std(vals)/max(np.mean(vals),EPS)) if vals.size else np.nan
    core=I[r_m<0.35*ring_r_m]; ringmean=np.mean(vals) if vals.size else np.nan
    dark=float(np.mean(core)/max(ringmean,EPS)) if core.size else np.nan
    return dark,azcv


def fit_plane(img,z_index,pixel_m,q,kr_m_inv,m_max=8,rmax_um=260,n_r=70,n_theta=144,center=None):
    if center is None: cy,cx,score=find_dark_core_center(img)
    else: cy,cx=center; score=np.nan
    rp,_,_=principal_inner_ring(img,cy,cx); ring_r_m=rp*pixel_m
    radii_px=np.linspace(1.5,min(rmax_um*1e-6/pixel_m,65),n_r); theta=np.linspace(0,2*np.pi,n_theta,endpoint=False)
    pol=sample_polar(img,cy,cx,radii_px,theta); y=pol.ravel(); rr,tt=np.meshgrid(radii_px*pixel_m,theta,indexing='ij'); rflat=rr.ravel(); pflat=tt.ravel()
    # radial integration measure r dr dphi; suppress very low-SNR pixels only softly
    w=(rflat/max(rflat.max(),EPS)); w*=0.25+0.75*np.sqrt(np.clip(y/max(y.max(),EPS),0,1))
    m_values=np.arange(-m_max,m_max+1,dtype=int)
    B,_=modal_basis(q,m_values,kr_m_inv,rflat,pflat)
    c,loss,res=fit_coefficients(B,y,w,m_values)
    pred=np.abs(B@c)**2; yn=y/max(y.max(),EPS); pn=pred/max(pred.max(),EPS)
    corr=float(np.corrcoef(yn,pn)[0,1]); nrmse=float(np.sqrt(np.mean((yn-pn)**2))/max(np.sqrt(np.mean(yn**2)),EPS))
    cc,th,g=corrected_coefficients_phase_only(c,m_values); predc=np.abs(B@cc)**2; pcn=predc/max(predc.max(),EPS)
    corr_c=float(np.corrcoef(yn,pcn)[0,1]); nrmse_c=float(np.sqrt(np.mean((yn-pcn)**2))/max(np.sqrt(np.mean(yn**2)),EPS))
    dark0,az0=image_metrics(pred,rflat,pflat,ring_r_m); dark1,az1=image_metrics(predc,rflat,pflat,ring_r_m)
    return PlaneFit(z_index,q,kr_m_inv,cy,cx,rp,score,m_values,c,loss,corr,nrmse,corr_c,nrmse_c,dark0,dark1,az0,az1), dict(radii_px=radii_px,theta=theta,measured=pol,pred=pred.reshape(n_r,n_theta),pred_corr=predc.reshape(n_r,n_theta),g_theta=g,theta_dense=th,coeffs_corr=cc)


def load_first_scan(folder:Path,roi_size=768):
    images=[]
    for z in sorted({int(p.stem.split('_')[0][1:]) for p in folder.glob('z*_*.bmg')}):
        frames=[]
        for p in sorted(folder.glob(f'z{z}_*.bmg')):
            a=preprocess(read_bmg(p)); cy,cx,_=find_dark_core_center(a); h=roi_size//2; y0=max(0,min(a.shape[0]-roi_size,int(round(cy))-h)); x0=max(0,min(a.shape[1]-roi_size,int(round(cx))-h)); frames.append(a[y0:y0+roi_size,x0:x0+roi_size])
        # align on dark-core centres, not global intensity centroid
        centers=[find_dark_core_center(f)[:2] for f in frames]; target=np.median(np.asarray(centers),axis=0); regs=[ndimage.shift(f,(target[0]-c[0],target[1]-c[1]),order=1,mode='constant',cval=0) for f,c in zip(frames,centers)]
        images.append(np.mean(regs,axis=0))
    return images

def fit_stack_torch(images,pixel_m,q,kr_m_inv,m_max=8,rmax_um=220,n_r=48,n_theta=96,steps=1800,lr=0.035):
    """Fast batched modal intensity fit using PyTorch autograd.

    Each z plane has its own complex aberration coefficients, while q and k_perp are shared.
    This mirrors the Miao annular retrieval: each z samples a different input annulus.
    """
    import torch
    torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    m_values=np.arange(-m_max,m_max+1,dtype=int)
    pols=[]; centers=[]; rings=[]; scores=[]
    radii_px=np.linspace(1.5,min(rmax_um*1e-6/pixel_m,70),n_r)
    theta=np.linspace(0,2*np.pi,n_theta,endpoint=False)
    rr,tt=np.meshgrid(radii_px*pixel_m,theta,indexing='ij')
    B,_=modal_basis(q,m_values,kr_m_inv,rr.ravel(),tt.ravel())
    rflat=rr.ravel(); base_w=rflat/max(rflat.max(),EPS)
    for img in images:
        cy,cx,score=find_dark_core_center(img); rp,_,_=principal_inner_ring(img,cy,cx)
        pol=sample_polar(img,cy,cx,radii_px,theta); pols.append(pol); centers.append((cy,cx)); rings.append(rp); scores.append(score)
    Y=np.stack([p.ravel()/max(p.max(),EPS) for p in pols]).astype(np.float32)
    W=np.stack([(base_w*(0.25+0.75*np.sqrt(np.clip(y,0,1)))) for y in Y]).astype(np.float32)
    Bt=torch.tensor(B.astype(np.complex64))
    Yt=torch.tensor(Y); Wt=torch.tensor(W)
    P=len(images); K=len(m_values); i0=int(np.where(m_values==0)[0][0])
    cr=torch.zeros((P,K),dtype=torch.float32,requires_grad=True); ci=torch.zeros((P,K),dtype=torch.float32,requires_grad=True)
    with torch.no_grad(): cr[:,i0]=1.0
    opt=torch.optim.Adam([cr,ci],lr=lr)
    for it in range(steps):
        opt.zero_grad()
        C=torch.complex(cr,ci)
        U=torch.einsum('nk,pk->pn',Bt,C)
        pred=U.abs().square()
        # Per-plane optimal intensity scale is folded into the coefficient norm naturally, but
        # normalizing predicted peak stabilizes the inverse problem against the arbitrary A(z).
        pred=pred/(pred.amax(dim=1,keepdim=True)+1e-8)
        den=(Wt*Yt.square()).sum(dim=1)+1e-8
        data=((Wt*(pred-Yt).square()).sum(dim=1)/den).mean()
        nonideal=(cr.square()+ci.square()); nonideal[:,i0]=0
        reg=2e-4*(nonideal.sum(dim=1)/(cr[:,i0].square()+ci[:,i0].square()+1e-6)).mean()
        phasefix=1e-4*ci[:,i0].square().mean()
        loss=data+reg+phasefix
        loss.backward(); opt.step()
        if it in (0,steps//2,steps-1): pass
    C=(cr.detach().numpy()+1j*ci.detach().numpy())
    # rotate each plane global phase so ideal coefficient is real-positive
    C=C*np.exp(-1j*np.angle(C[:,i0]))[:,None]
    outputs=[]
    for p in range(P):
        u=B@C[p]; pred=np.abs(u)**2; yn=Y[p]; pn=pred/max(pred.max(),EPS)
        corr=float(np.corrcoef(yn,pn)[0,1]); nrmse=float(np.sqrt(np.mean((yn-pn)**2))/max(np.sqrt(np.mean(yn**2)),EPS))
        cc,thd,g=corrected_coefficients_phase_only(C[p],m_values); predc=np.abs(B@cc)**2; pcn=predc/max(predc.max(),EPS)
        corr_c=float(np.corrcoef(yn,pcn)[0,1]); nrmse_c=float(np.sqrt(np.mean((yn-pcn)**2))/max(np.sqrt(np.mean(yn**2)),EPS))
        ring_r_m=rings[p]*pixel_m; dark0,az0=image_metrics(pred,rflat,tt.ravel(),ring_r_m); dark1,az1=image_metrics(predc,rflat,tt.ravel(),ring_r_m)
        fit=PlaneFit(p,q,kr_m_inv,centers[p][0],centers[p][1],rings[p],scores[p],m_values,C[p],float(np.mean((pn-yn)**2)),corr,nrmse,corr_c,nrmse_c,dark0,dark1,az0,az1)
        outputs.append((fit,dict(radii_px=radii_px,theta=theta,measured=pols[p],pred=pred.reshape(n_r,n_theta),pred_corr=predc.reshape(n_r,n_theta),g_theta=g,theta_dense=thd,coeffs_corr=cc)))
    return outputs
