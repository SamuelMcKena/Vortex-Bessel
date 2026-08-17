# Phase 2H — common-eikonal vector two-surface refractive axicon mathematics

## Scope and model class

Phase 2H constructs the electromagnetic boundary field produced by a rigidly
misaligned macroscopic plano-conical refractive axicon.  The optic is represented
by two real dielectric surfaces and the transmitted complex vector field is
placed on one fixed laboratory plane immediately downstream.  Subsequent
propagation is performed by the repository vector angular-spectrum / Debye
machinery.

The model is a **vector geometrical-optics / eikonal boundary-field model**, not a
full-volume FDTD/FEM solution of the glass.  Its high-frequency ansatz is

\[
\mathbf E(\mathbf r)=\mathbf a(\mathbf r)e^{i\Phi(\mathbf r)},
\qquad
\mathbf k=\nabla\Phi,
\]

with a common eikonal shared by the coherent polarization components.  The code
therefore contains explicit validity gates that reject fields for which a single
local phase normal is not a good description.

The primary external formulation/validation references recorded for this stage
are Yun, Crabtree & Chipman (Applied Optics 50, 2855–2865, 2011), Kim et al.
(JOSA A 35, 526–535, 2018), Zhao Bin & Li Zhu (Applied Optics 37, 2563–2568,
1998), and Thaning, Jaroszewicz & Friberg (Applied Optics 42, 9–17, 2003).

## 1. Coordinate systems and physical pose

Let \((x',y',z')\) be fixed to the axicon and \((x,y,z)\) be the laboratory
frame.  The repository rigid-rotation convention is

\[
R=R_y(\theta_y)R_x(\theta_x),
\qquad
\mathbf r_{\rm lab}=R\mathbf r_{\rm ax}.
\]

Field components remain in the fixed laboratory Cartesian basis during
rotated-plane sampling.  At a physical interface they are transformed to the
axicon basis by

\[
\mathbf E_{\rm ax}=R^T\mathbf E_{\rm lab}.
\]

A lateral axicon decentre is a tangent-plane displacement and remains separate
from rigid rotation.  The currently implemented physical surface order is
**flat entrance -> conical exit**.  Cone-first orientation is deliberately not
reinterpreted as the same geometry.

## 2. Physical plano-conical surfaces

For the current flat-first model,

\[
z'=0
\]

is the external-to-glass entrance surface and

\[
z'=t_c-r'\tan\gamma,
\qquad
r'=\sqrt{x'^2+y'^2}
\]

is the conical glass-to-external exit surface.  A declared clear radius \(R_c\)
requires

\[
t_e=t_c-R_c\tan\gamma>0.
\]

Impossible combinations of base angle, radius and centre thickness are rejected.
The conical outward normal is

\[
\hat{\mathbf n}_2=
\begin{bmatrix}
\sin\gamma\cos\phi\\
\sin\gamma\sin\phi\\
\cos\gamma
\end{bmatrix}.
\]

The calibration wrapper only accepts the angle convention
`base_angle_from_flat_face`; vendor apex/deviation conventions must be converted
explicitly before they enter this geometry.

## 3. Carrier-tracked vector sampling on the tilted entrance plane

The incoming vector field is first projected onto the Maxwell transverse
subspace.  A single spectral carrier centre \((f_{c,x},f_{c,y})\) is estimated
from the combined Ex/Ey/Ez spectrum and removed before rotated-plane
interpolation.  Each Cartesian component is then rotated with the same
carrier-aware angular-spectrum mapping.

The destination carrier on the physical tilted surface is retained **analytically**.
It is not sampled as

\[
e^{i2\pi(f'_{c,x}x'+f'_{c,y}y')}
\]

on the entrance grid.  This is essential because a several-degree physical plane
rotation can produce a local carrier above that grid's Nyquist limit even when
the final laboratory output field is perfectly representable.

The entrance data therefore consist of

\[
\mathbf E_{\rm env}(x',y')
\]

plus an analytic carrier vector.  Maxwell-consistent \(\mathbf H\) is reconstructed
spectrally from

\[
Z_0\mathbf H=n\,\hat{\mathbf k}\times\mathbf E,
\]

which also supplies an independent Poynting-flux diagnostic.

## 4. Why Snell law uses the eikonal, not total structured-field Poynting

An early Phase-2H prototype used the local total time-averaged Poynting direction
as the ray direction.  The independent Fermat/eikonal gradient gate rejected
that formulation.  For a single local plane wave, phase normal and Poynting
vector are parallel.  For a structured vector superposition, however, spin and
interference currents can make the total energy-flow direction differ from the
local canonical wavevector.

The accepted model therefore enforces

\[
\boxed{\text{Snell direction}=\hat{\mathbf k}=\nabla\Phi/|\nabla\Phi|}
\]

and uses Poynting only for energy-flow validation.

For a vector envelope with fixed Cartesian components \(E_a\), the local
canonical transverse phase gradients are estimated in a basis-invariant form:

\[
q_j^{\rm env}=
\frac{\operatorname{Im}\left[\sum_a E_a^*\,\partial_jE_a\right]}
{\sum_a|E_a|^2},
\qquad j\in\{x',y'\}.
\]

The analytic carrier is then added exactly:

\[
q_x=q_x^{\rm env}+2\pi f'_{c,x},
\qquad
q_y=q_y^{\rm env}+2\pi f'_{c,y}.
\]

For medium wavenumber

\[
k=n\frac{2\pi}{\lambda_0},
\]

the forward local longitudinal component is

\[
q_z=\sqrt{k^2-q_x^2-q_y^2},
\]

and

\[
\hat{\mathbf k}_{\rm in}=\frac{1}{k}(q_x,q_y,q_z).
\]

A non-positive radicand is a non-propagating local phase gradient and is rejected.

## 5. Common-eikonal validity and Southwell integration

The vector GO model is only valid when the energetic components genuinely share
one local eikonal.  For each sufficiently bright Cartesian component, its own
local canonical phase gradient is compared with the vector-weighted gradient.
The intensity-weighted disagreement is recorded as a fraction of \(|k|\).  The
production default rejects the field when the 95th percentile exceeds 2%.

The two measured transverse gradients are then integrated to one scalar eikonal
with the repository sparse Southwell/trapezoidal least-squares reconstruction.
Writing optical-path slopes as

\[
\frac{\partial W}{\partial x'}=\frac{q_x}{k_0},
\qquad
\frac{\partial W}{\partial y'}=\frac{q_y}{k_0},
\]

with \(k_0=2\pi/\lambda_0\), the reconstructed phase is

\[
\Phi_{\rm in}=k_0W.
\]

The reconstructed gradients are compared back to \((q_x,q_y)\).  The default
95th-percentile reconstruction-error gate is 1% of \(|k|\).  Thus a genuinely
multimode/non-integrable vector superposition is refused instead of being forced
through a one-ray-per-point model.

The angle between the reconstructed wavevector and the separately calculated
Poynting direction is stored as a diagnostic; disagreement here is not used to
alter Snell's law.

## 6. Exact vector Snell refraction

At each interface, let \(\hat{\mathbf n}\) point from medium 1 to medium 2.  Split

\[
\hat{\mathbf k}_1=
(\hat{\mathbf k}_1\cdot\hat{\mathbf n})\hat{\mathbf n}
+\hat{\mathbf k}_{1t}.
\]

Tangential-wavevector continuity gives

\[
n_1\hat{\mathbf k}_{1t}=n_2\hat{\mathbf k}_{2t},
\]

so

\[
\hat{\mathbf k}_{2t}=\frac{n_1}{n_2}\hat{\mathbf k}_{1t}
\]

and the forward transmitted direction is

\[
\boxed{
\hat{\mathbf k}_2=
\frac{n_1}{n_2}\hat{\mathbf k}_{1t}
+\sqrt{1-\left\|\frac{n_1}{n_2}\hat{\mathbf k}_{1t}\right\|^2}\,
\hat{\mathbf n}
}.
\]

A negative radicand is total internal reflection and is not numerically repaired.
The same law is applied at the flat entrance and conical exit.

## 7. Exact ray/cone intersection and finite glass path

After the entrance refraction, an internal ray is

\[
\mathbf r'(s)=\mathbf r'_0+s\hat{\mathbf k}_g.
\]

Substitution into the cone equation produces a quadratic in \(s\).  The first
positive physical intersection is selected.  This supplies the actual exit point,
local cone normal and glass path length for each valid entrance sample.  The
solver therefore does not replace rigid tilt with a rotated thin conical phase.

## 8. Three-dimensional local s/p Fresnel transport

Before interface transport, the electric vector is projected transverse to its
incident wavevector:

\[
\mathbf E_\perp=
\left(I-\hat{\mathbf k}_1\hat{\mathbf k}_1^T\right)\mathbf E.
\]

For non-normal incidence,

\[
\hat{\mathbf s}=\frac{\hat{\mathbf n}\times\hat{\mathbf k}_1}
{\|\hat{\mathbf n}\times\hat{\mathbf k}_1\|},
\]

\[
\hat{\mathbf p}_1=\hat{\mathbf k}_1\times\hat{\mathbf s},
\qquad
\hat{\mathbf p}_2=\hat{\mathbf k}_2\times\hat{\mathbf s}.
\]

The local amplitudes are

\[
E_s=\mathbf E_\perp\cdot\hat{\mathbf s},
\qquad
E_p=\mathbf E_\perp\cdot\hat{\mathbf p}_1.
\]

For isotropic nonmagnetic media,

\[
t_s=\frac{2n_1\cos\theta_i}
{n_1\cos\theta_i+n_2\cos\theta_t},
\]

\[
t_p=\frac{2n_1\cos\theta_i}
{n_2\cos\theta_i+n_1\cos\theta_t}.
\]

The transmitted complex vector is

\[
\boxed{
\mathbf E_2=t_sE_s\hat{\mathbf s}+t_pE_p\hat{\mathbf p}_2.
}
\]

At exact normal incidence the s/p plane is degenerate and the code uses the
polarization-independent normal-incidence amplitude.  The operation is repeated
at the second surface.  Because the conical normal and oblique incidence vary
spatially, the model uses no single global 2x2 Jones matrix for the tilted axicon.

## 9. Interface energy identity

For the actual mixed polarization state,

\[
T=\frac{n_2\cos\theta_t|\mathbf E_2|^2}
{n_1\cos\theta_i|\mathbf E_1|^2}.
\]

Reflection is calculated from \(r_s,r_p\).  With real lossless indices each
surface must satisfy

\[
\boxed{R+T=1}
\]

to numerical tolerance.  These are independent surface gates, not an inferred
end-to-end power check.

## 10. Finite optical path and total transmitted eikonal

For internal glass distance \(L_g\) and downstream external distance \(L_e\),

\[
\mathrm{OPL}=n_gL_g+n_eL_e.
\]

The phase arriving at the reference plane is

\[
\Phi_{\rm ref}(x',y')=
\Phi_{\rm in}(x',y')+k_0\,\mathrm{OPL}(x',y'),
\]

up to one irrelevant global piston.  This is the phase used by the Fermat gate:
a coordinate-transformed gradient on the output plane must recover the independently
traced outgoing transverse wavevector.

## 11. Fixed laboratory reference plane

The output boundary field is constructed on one physical laboratory plane

\[
z=z_{\rm ref}
\]

placed immediately downstream of all valid exit points.  For each outgoing ray,

\[
L_e=\frac{z_{\rm ref}-z_{\rm exit}}{k_{z,\rm out}/|\mathbf k|},
\]

\[
x_{\rm ref}=x_{\rm exit}+L_e\hat k_{x,\rm out},
\qquad
y_{\rm ref}=y_{\rm exit}+L_e\hat k_{y,\rm out}.
\]

No z-dependent recentering or beam-following coordinate warp is allowed.  Real
steering/decentre therefore survives into subsequent propagation and objective
mapping.

## 12. Ray-tube Jacobian and normal-flux amplitude

The entrance-to-reference-plane map has Jacobian

\[
J=\det
\begin{bmatrix}
\partial x_{\rm ref}/\partial x' & \partial x_{\rm ref}/\partial y'\\
\partial y_{\rm ref}/\partial x' & \partial y_{\rm ref}/\partial y'
\end{bmatrix}.
\]

A mixed sign over the physical footprint indicates a fold/caustic before the
chosen boundary plane and is rejected rather than hidden by interpolation.

Within the common-eikonal local-plane-wave approximation, normal flux conservation
sets the output magnitude:

\[
n_e|\mathbf E_{\rm out}|^2\hat k_{z,\rm out}|J|
=
n_e|\mathbf E_{\rm in,\perp}|^2\hat k_{z,\rm in}T_1T_2.
\]

The independently reconstructed electromagnetic Poynting flux is also compared
with the local-plane-wave normal-flux model.  Excessive spatial disagreement is
a model-validity failure; Poynting is not used to redefine the ray direction.

## 13. Phase-safe inverse remapping

The traced output coordinates are irregular.  The implementation first fits a
sparse affine inverse seed, then Newton-solves

\[
(x',y')\mapsto(x_{\rm ref},y_{\rm ref})
\]

for every regular output-grid point.  Invalid rays, points outside the mapped
footprint and large inverse residuals are rejected.

Crucially, two quantities are remapped **separately**:

1. the slowly sampled complex vector/polarization envelope;
2. the continuous unwrapped common-eikonal + OPL phase.

Only on the final laboratory grid is

\[
\mathbf E_{\rm ref}=\mathbf A_{\rm ref}e^{i\Phi_{\rm ref}}
\]

formed.  This prevents an unrepresentable entrance-plane carrier from aliasing
before physical refraction.

## 14. Maxwell projection and spectral normal-flux closure

Interpolation can create a small longitudinal inconsistency, so the final regular
field is projected in angular-spectrum space with

\[
P(\mathbf k)=I-\frac{\mathbf k\mathbf k^T}{|\mathbf k|^2}.
\]

For a projected field in a lossless dielectric, the plane-integrated +z flux is
proportional to

\[
\iint n\frac{k_z}{k}
\left(|\widetilde E_x|^2+|\widetilde E_y|^2+|\widetilde E_z|^2\right)
\,dk_xdk_y.
\]

The local morphology has already been fixed by the physical ray-tube Jacobian.
One final **global** amplitude factor is allowed only to close small interpolation
and transverse-projection loss.  The default implementation rejects a case if
this power correction differs from unity by more than 10%.

## 15. Sampling / alias rejection

Outgoing rays require transverse spatial frequencies

\[
f_x=\frac{n_e}{\lambda_0}\hat k_x,
\qquad
f_y=\frac{n_e}{\lambda_0}\hat k_y.
\]

For output spacing \(\Delta x\),

\[
f_{\rm Nyq}=\frac{1}{2\Delta x}.
\]

The default production gate requires the largest traced transverse frequency to
remain below 90% of Nyquist.  A physically valid high-cone-angle optic can
therefore be rejected on a coarse/large computational window instead of producing
an aliased but visually plausible result.

## 16. Calibration boundary

The calibrated segmented-vector tilt wrapper reuses the already-validated
SLM1 -> relay -> HWP -> SLM2 -> QWP -> physical 4F selected-order chain and
replaces only its normal-incidence axicon stage with Phase 2H.  Absolute tilt is
enabled only when the bundle contains calibrated provenance for:

- base angle;
- explicit angle convention `base_angle_from_flat_face`;
- clear radius;
- centre thickness;
- refractive index;
- verified flat-face-upstream orientation.

The current wrapper refuses cone-first orientation and refuses to infer a base
angle from an apex/deviation/vendor label.

## 17. Mandatory validation hierarchy

Before Phase-2H figures can be report-authorised, all of the following must pass:

1. vector Fresnel \(R+T=1\) at each lossless surface;
2. zero-tilt plane-wave recovery of the exact Snell conical direction;
3. common-component eikonal agreement;
4. common-phase integrability / Southwell reconstruction agreement;
5. Fermat consistency: propagated phase gradient equals the independently traced
   outgoing transverse wavevector;
6. x/y rotational covariance for an axisymmetric optic and circular input;
7. positive/negative tilt mirror consistency in polarization-independent ray
   metrics;
8. ray-tube normal-flux closure;
9. regular-grid spectral normal-flux closure;
10. Maxwell transversality after remapping/projection;
11. hard rejection of under-sampled outgoing wavevectors;
12. direct 0/5/10-degree plane-wave geometry agreement with the pre-existing,
    independently validated scalar two-surface Snell tracer;
13. end-to-end execution of the calibrated six-sector vector SLM/4F field through
    Phase 2H and into the spatial-vector objective/sample solver;
14. preservation of all existing Phase-2G vector/analyzer and Phase-2C
    objective/interface numerical regressions.

The old total-Poynting-directed prototype remains non-authoritative provenance.
The accepted Phase-2H route is the **common-eikonal / phase-normal** formulation.
