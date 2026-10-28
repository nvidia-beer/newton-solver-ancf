# Method

ANCF (absolute nodal coordinate formulation) represents the position field by
$\mathbf r(\mathbf X)=\mathbf S(\mathbf X)\mathbf q$, with global positions and
spatial derivatives as coordinates. The package supplies four element families:

| Element | Coordinates | Elastic model |
| --- | --- | --- |
| Cable | 2 nodes, 12 DOFs | Cubic Hermite centerline; axial Green strain and curvature bending; no torsion |
| Beam | 3243: 2 nodes, 24 DOFs; 3333: 3 nodes, 27 DOFs | Shear-deformable continuum beam; selective integration |
| Shell | 3423: 4 nodes, 24 DOFs | Bilinear position/director interpolation; assumed natural strains (ANS) |
| Tetrahedron | 4 nodes, 48 DOFs | Cubic FN interpolation; straight reference cells; 46-point degree-eight quadrature |

For continuum elements, $\mathbf F=\partial\mathbf r/\partial\mathbf X$,
$\mathbf E=(\mathbf F^T\mathbf F-\mathbf I)/2$, and

$$
U(\mathbf q)=\frac12\int_{\Omega_0}\mathbf e^T\mathbf C\mathbf e\,dV,
\qquad \mathbf M=\int_{\Omega_0}\rho\mathbf S^T\mathbf S\,dV,
\qquad \mathbf f_{\rm int}=\nabla U,\quad \mathbf K=\nabla^2U.
$$

Here $\mathbf e$ contains Green strain with engineering shear components and
$\mathbf C$ is the constitutive matrix. Generalized elements use float64,
consistent mass, and exact local energy Hessians assembled into sparse CSR.
The cable uses axial and isotropic curvature energy; its intrinsic curvature
norm is regularized for initially curved elements. It has no material frame.

## Shared HHT integration

Both generalized elements and optimized tires use the same HHT formulas, with
$\alpha=-0.2$, $\beta=(1-\alpha)^2/4=0.36$, $\gamma=1/2-\alpha=0.7$:

$$
\begin{aligned}
\widehat{\mathbf q}&=\mathbf q_n+h\mathbf v_n+h^2(1/2-\beta)\mathbf a_n,&
\widehat{\mathbf v}&=\mathbf v_n+h(1-\gamma)\mathbf a_n,\\
\mathbf a&=(\mathbf q-\widehat{\mathbf q})/(\beta h^2),&
\mathbf v&=\widehat{\mathbf v}+\gamma h\mathbf a,\\
\mathbf R&=\mathbf M\mathbf a+(1+\alpha)(\mathbf I-\mathbf f_{\rm ext})
-\alpha(\mathbf I_n-\mathbf f_{{\rm ext},n})=\mathbf0,&
\mathbf I&=\nabla U+\mathbf C_d\mathbf v.
\end{aligned}
$$

Generalized elements use $\mathbf C_d=c\mathbf M$ and a regularized Newton
direction with Armijo search; acceptance checks the unregularized residual.
Sparse PCG checks curvature and the true residual. Position, velocity,
acceleration, and both force histories travel together between steps.
Initial states have zero acceleration/history; `initialize_history()` fills
forces but does not solve for a dynamically consistent initial acceleration.

The optimized tire path uses float32, lumped mass, sparse six-DOF node blocks,
fixed Newton/PCG budgets, pressure and friction. Single-tire kernels update
enhanced assumed strains (EAS); batched kernels keep EAS variables zero.
Shared HHT does not make its approximate tangent, convergence policy or state
layout identical to the generalized path. Rigid coupling is separate in
`_src/solvers/coupled/`; the MuJoCo adapter uses the installed public API.

## Derivatives and contact

A generalized recorded step differentiates its converged residual on free DOFs:

$$
\mathbf R_{\mathbf q}^{T}\boldsymbol\lambda=L_{\mathbf q},
\qquad \frac{dL}{d\theta}=L_\theta-\mathbf R_\theta^T\boldsymbol\lambda.
$$

The adjoint uses the unshifted sparse tangent with a checked GMRES fallback.
Supported inputs include all state histories, loads, prescribed coordinates,
and exposed stiffness parameters. Geometry, topology, mass/density, timestep,
gravity and contact coefficients stay fixed. Input arrays must remain unchanged
until backward; derivatives assume a locally nonsingular converged branch.
The generalized frictionless floor uses $d=z_{\rm floor}-z$,
$p=(d+\sqrt{d^2+s^2})/2$, and $V_{\rm floor}=kp^2/2$ at position nodes.

`ANCFTireEquilibrium` differentiates static shell/rim equilibrium for pressure
and stiffness fitting. Its tangent includes EAS condensation, nonsymmetric
follower pressure, and a sealed-cavity rank-one term. Sparse LU on
$\left[\begin{smallmatrix}K&u\\-v^T&1\end{smallmatrix}\right]$
applies $K+uv^T$ without forming a dense outer product. These derivatives hold
on a fixed contact/EAS-clipping branch. The dynamic tire and MuJoCo trajectories
do not have this adjoint; `diffsim_ancf_tire_lift` validates the static fit in a
live rig without differentiating that trajectory.

## Papers

The following describe the formulation foundations; the numerical choices and
derivative scope above describe this implementation.

- Hilber, Hughes and Taylor (1977), [Improved numerical dissipation for time integration algorithms in structural dynamics](https://doi.org/10.1002/eqe.4290050306): HHT integration.
- Gerstmayr and Shabana (2006), [Analysis of Thin Beams and Cables Using the Absolute Nodal Co-ordinate Formulation](https://doi.org/10.1007/s11071-006-1856-1): cable kinematics and strain energy.
- Nachbagauer, Gruber and Gerstmayr (2013), [Structural and Continuum Mechanics Approaches for a 3D Shear Deformable ANCF Beam Finite Element](https://doi.org/10.1115/1.4006787): shear-deformable beam formulation.
- Yamashita, Valkeapää, Jayakumar and Sugiyama (2015), [Continuum Mechanics Based Bilinear Shear Deformable Shell Element Using Absolute Nodal Coordinate Formulation](https://doi.org/10.1115/1.4028657): ANCF shell formulation.
- Pappalardo, Wang and Shabana (2017), [Development of ANCF tetrahedral finite elements for the nonlinear dynamics of flexible structures](https://doi.org/10.1007/s11071-017-3635-6): cubic FN tetrahedron.
