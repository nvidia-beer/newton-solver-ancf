# Paper-to-code guide

The links below map the papers to the implemented equations and methods.
Partial implementations and differences are identified alongside the code.

## Cubic ANCF tetrahedra

[Pappalardo, Wang and Shabana (2017)](https://doi.org/10.1007/s11071-017-3635-6),
*Development of ANCF tetrahedral finite elements for the nonlinear dynamics of
flexible structures*, describes both FN and ENMC elements. This package implements
the four-node FN branch with 48 coordinates, restricted to straight reference cells.

| Paper location | Implementation |
| --- | --- |
| §5, Eq. (33), Appendix A Eq. (A.4): cubic FN interpolation and Cartesian gradients | [`tetrahedron.py`](../src/newton_solver_ancf/_src/solvers/ancf/tetrahedron.py): `_shape_functions_batch`, `SolverANCFTetrahedron.shape_functions`, and device `tetrahedral_shape_values` |
| §7.1, Eq. (37): spatial derivatives | [`tetrahedron.py`](../src/newton_solver_ancf/_src/solvers/ancf/tetrahedron.py): `_reference_coordinates` and the derivative branch of `shape_functions` |
| §7.2, Eq. (38): constant mass | [`tetrahedron.py`](../src/newton_solver_ancf/_src/solvers/ancf/tetrahedron.py): `SolverANCFTetrahedron.create_model` integrates local mass and creates the sparse model |
| §7.4: elastic generalized forces | [`tetrahedron.py`](../src/newton_solver_ancf/_src/solvers/ancf/tetrahedron.py): `_kinematics`, `_energy`, `_gradient`, `_hessian` |

The paper derives equations rather than numbered executable pseudocode. The
following summarizes this implementation's sequence:

```text
create_model: validate tetrahedra; compute FN shape values and derivatives;
              integrate consistent mass and cache reference quadrature
per trial q: compute F and Green strain at quadrature points;
             reject nonpositive det(F); integrate energy, force and tangent
assemble local contributions into CSR; solve the HHT step on free coordinates
```

The final solve is `SolverImplicit.step` in
[`implicit/solver.py`](../src/newton_solver_ancf/_src/solvers/implicit/solver.py);
[`implicit/sparse.py`](../src/newton_solver_ancf/_src/solvers/implicit/sparse.py)
provides CSR assembly and linear solves.

The positive 46-point rule, guarded sparse solve, HHT integrator and custom
implicit adjoint are implementation choices. The ENMC branch and the paper's
full benchmark suite are not supplied by this package.

## Cable and shell ANCF

[Recuero and Negrut (2016), *The Absolute Nodal Coordinate Formulation — ANCF*](https://www.projectchrono.org/assets/white_papers/FEA/WP_ANCF.pdf)
provides the following equation-level map. Its beam in §2 is the two-node cable,
not this package's shear-deformable 3243/3333 beam family.

| Paper location | Implementation |
| --- | --- |
| §2, Eqs. (2)–(6): centerline, axial/bending energy and mass | [`cable.py`](../src/newton_solver_ancf/_src/solvers/ancf/cable.py): `cable_shape_functions`, `evaluate_element_energy`, `evaluate_elements`, `SolverANCFCable.create_model` |
| §3, Eqs. (7)–(12): bilinear shell interpolation and internal force | [`kernels_element.py`](../src/newton_solver_ancf/_src/solvers/ancf/kernels_element.py): `_shape`, `_jacobian`, `_natural_strain`; [`kernels_stiffness.py`](../src/newton_solver_ancf/_src/solvers/ancf/kernels_stiffness.py): `compute_element_forces_stiffness` |
| §3.1, Eqs. (13)–(19): reference/fiber transformation | [`kernels_element.py`](../src/newton_solver_ancf/_src/solvers/ancf/kernels_element.py): `_compute_beta`, `_beta_transform_diag`, `_beta_transform_shear` |
| §3.2.2 and §3.2.4, Eqs. (31)–(32): EAS update and condensation | [`kernels_stiffness.py`](../src/newton_solver_ancf/_src/solvers/ancf/kernels_stiffness.py): `compute_rest_jacobians`, `compute_element_forces_stiffness`; static condensation in [`differentiation.py`](../src/newton_solver_ancf/_src/solvers/ancf/differentiation.py): `ANCFTireEquilibrium._elastic` |
| §3.2.3, Eqs. (26)–(28): ANS tying | [`kernels_element.py`](../src/newton_solver_ancf/_src/solvers/ancf/kernels_element.py): `_e13_at`, `_e23_at`, `_e33_at`, and matching strain-derivative helpers |

There is no numbered algorithm in this white paper. The implemented tire loop is:

```text
precompute reference Jacobians, fiber transforms, EAS basis and lumped mass
predict coordinates with HHT
for each fixed-budget Newton correction:
    evaluate ANS/EAS shell force and approximate tangent
    add pressure/contact; impose bead coordinates; solve sparse block system
    correct coordinates, velocities and EAS variables
retain acceleration and old-force histories for the next step
```

This loop is in [`tire.py`](../src/newton_solver_ancf/_src/solvers/ancf/tire.py):
`_precompute_mass`, `_precompute_rest_jacobians`, `_step_single` and `_step_batched`;
`_evaluate_forces_single` and `_evaluate_forces_batched` launch its element kernels.

The dynamic tire uses lumped mass and bounded EAS updates in single-tire kernels;
batched kernels keep EAS variables zero. Eq. (29) states a consistent mass integral. The generalized [`shell.py`](../src/newton_solver_ancf/_src/solvers/ancf/shell.py)
uses conservative ANS without EAS. Cable curvature regularization, exact local
cable Hessians, pressure/contact laws, sparse PCG and implicit gradients are
additional implementation details, not claims made by that white paper.

## HHT time integration

[Hilber, Hughes and Taylor (1977)](https://doi.org/10.1002/eqe.4290050306) maps to
[`implicit/hht.py`](../src/newton_solver_ancf/_src/solvers/implicit/hht.py):
`predict_position`, `predict_velocity`, `weighted_residual`, `mass_scale` and
`stiffness_scale`. The [method](method.md) states the implemented equations and
adjoint assumptions.
