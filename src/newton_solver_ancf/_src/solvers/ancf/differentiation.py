# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit pressure and stiffness sensitivities for a stationary ANCF tire on a vertical rim.

This opt-in module evaluates the *static* equations in float64; it does not
differentiate a truncated HHT/PCG iteration or modify its state. In reduced
coordinates z = (free shell coordinates, rim height), the equations are

    R(z, p) = A.T (f_elastic - f_pressure - f_ground - f_gravity) + f_rim = 0
    J dz/dp = -R_p,       dL/dp = (h - h_target) dh/dp.

For a uniform elastic-modulus multiplier s, J dz/ds = -R_s uses the
unscaled elastic force as R_s. The EAS solution is invariant to that scaling.

A eliminates the bead constraints, including their dependence on rim height.
The last equation is therefore the rim's vertical force balance, not a fixed
boundary approximation. At zero velocity the rig's viscous friction and all
damping vanish. This model is restricted to a nonrotating rim on flat ground.

ANS strains are quadratic forms in each element's eight vector coordinates:
e_s = 0.5 H_s : (Q Q.T - Q0 Q0.T). Thus B_s = H_s Q and the exact elastic
tangent is B.T C B + sum_s stress_s H_s (tensor identity). EAS variables are
eliminated using their linear local equation, with the same +/-0.1 clipping
as the single-environment solver. Both follower-pressure geometry and the
volume derivative in p_live = p V_ref / V are differentiated. Ground contact
uses the current active set; derivatives at an active-set boundary are not
classical two-sided derivatives.

No finite differences are used for training. They belong in the tests.
"""

from dataclasses import dataclass

import numpy as np

from ._sparse_tangent import ShellSparsity, SparseTangent


def _shape(u, v):
    a = np.array([-1.0, 1.0, 1.0, -1.0])
    b = np.array([-1.0, -1.0, 1.0, 1.0])
    return (1 + a * u) * (1 + b * v) / 4, a * (1 + b * v) / 4, b * (1 + a * u) / 4


def _derivatives(u, v, w, thickness):
    n, du, dv = _shape(u, v)
    d = np.zeros((8, 3))
    d[::2, 0], d[::2, 1] = du, dv
    d[1::2, 0], d[1::2, 1], d[1::2, 2] = w * thickness / 2 * du, w * thickness / 2 * dv, thickness / 2 * n
    return d


def _strain_forms(d):
    return np.array(
        [np.outer(d[:, i], d[:, i]) for i in range(3)]
        + [np.outer(d[:, i], d[:, j]) + np.outer(d[:, j], d[:, i]) for i, j in ((0, 1), (0, 2), (1, 2))]
    )


def _voigt_transform(j0, cosine, sine):
    a1 = j0[:, 0] / np.linalg.norm(j0[:, 0])
    a3 = np.cross(j0[:, 0], j0[:, 1])
    a3 /= np.linalg.norm(a3)
    a2 = np.cross(a3, a1)
    beta = np.array([cosine * a1 + sine * a2, -sine * a1 + cosine * a2, a3]) @ np.linalg.inv(j0).T
    bases = np.zeros((6, 3, 3))
    for i in range(3):
        bases[i, i, i] = 1
    for s, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
        bases[s, i, j] = bases[s, j, i] = 0.5
    transformed = beta @ bases @ beta.T
    return np.array(
        [
            transformed[:, 0, 0],
            transformed[:, 1, 1],
            transformed[:, 2, 2],
            2 * transformed[:, 0, 1],
            2 * transformed[:, 0, 2],
            2 * transformed[:, 1, 2],
        ]
    )


def _skew(v):
    result = np.zeros((*v.shape[:-1], 3, 3))
    result[..., 0, 1], result[..., 0, 2] = -v[..., 2], v[..., 1]
    result[..., 1, 0], result[..., 1, 2] = v[..., 2], -v[..., 0]
    result[..., 2, 0], result[..., 2, 1] = -v[..., 1], v[..., 0]
    return result


class ANCFTireEquilibrium:
    """Static pressure derivative for one nonrotating, vertically free rim.

    The source solver is read only. Its terrain must be a plane and its bead
    directors must be fixed. This is not a trajectory derivative for MuJoCo.
    The float64 equations use sparse shell blocks and a rank-one cavity term. Set
    ``OPENBLAS_NUM_THREADS=1`` before starting Python to avoid oversubscription.

    Args:
        solver: Single-environment ANCF solver supplying reference geometry,
            material, cavity, gravity, and plane contact parameters.
        bead_indices: Indices of the nodes attached to the vertical rim.
        spindle_mass: Rigid rim mass [kg].
        load: Additional downward rim load [N].
        build_pressure: Fixed reference pressure subtracted from gas pressure [Pa].
    """

    @dataclass
    class Result:
        """Converged state and nominal-pressure sensitivity.

        Attributes:
            coordinates: Reduced coordinates, translations [m] and dimensionless directors.
            height: Equilibrium rim height [m].
            pressure: Nominal cavity pressure [Pa].
            dh_dp: Derivative of rim height with respect to pressure [m/Pa].
            residual_norm: Maximum absolute SI residual component, forces [N]
                and generalized director forces [N m].
            linear_residual: Relative sensitivity-equation residual.
            iterations: Accepted equilibrium Newton corrections.
            stiffness_scale: Uniform multiplier of the reference elastic moduli.
            dh_dscale: Derivative of rim height with respect to that multiplier [m].
        """

        coordinates: np.ndarray
        height: float
        pressure: float
        dh_dp: float
        residual_norm: float
        linear_residual: float
        iterations: int
        stiffness_scale: float = 1.0
        dh_dscale: float = 0.0

        def loss_gradient(self, target_height: float) -> float:
            """Derivative of half squared height error with respect to pressure [m^2/Pa]."""
            return (self.height - target_height) * self.dh_dp

    def __init__(self, solver, bead_indices, spindle_mass: float, load: float = 0.0, build_pressure: float = 30_000.0):
        if solver.n_envs != 1 or solver.terrain is not None:
            raise ValueError("Equilibrium differentiation supports one tire on the flat ground plane")
        if not np.isfinite([spindle_mass, load, build_pressure]).all() or min(spindle_mass, load, build_pressure) < 0:
            raise ValueError("Rim mass, downward load, and build pressure must be finite and nonnegative")
        model = solver.ancf
        self.nodes = model.elem_nodes.numpy().astype(int)
        self.rest = np.stack((model.node_x0.numpy(), model.node_D0.numpy()), axis=1).astype(np.float64)
        self.n = len(self.rest)
        self.ne = len(self.nodes)
        self.beads = np.asarray(bead_indices, dtype=int)
        if (
            self.beads.ndim != 1
            or not 0 < len(self.beads) < self.n
            or np.any((self.beads < 0) | (self.beads >= self.n))
        ):
            raise ValueError("Bead indices must select a nonempty proper subset of tire nodes")
        if len(np.unique(self.beads)) != len(self.beads):
            raise ValueError("Bead indices must be unique")
        fixed = np.zeros((self.n, 6), dtype=bool)
        fixed[self.beads] = True
        self.free = np.flatnonzero(~fixed.ravel())
        self.rim_dofs = 6 * self.beads + 1
        self.size = len(self.free) + 1
        self.base = self.rest.reshape(-1).copy()
        current = solver.node_x.numpy()
        centre = (current[self.beads] - self.rest[self.beads, 0]).mean(axis=0)
        if not np.allclose(current[self.beads] - centre, self.rest[self.beads, 0], atol=1e-4):
            raise ValueError("The equilibrium rim must be unrotated")
        if not np.allclose(solver.node_D.numpy()[self.beads], self.rest[self.beads, 1], atol=1e-4):
            raise ValueError("The equilibrium bead directors must be fixed at their reference orientation")
        self.base.reshape(self.n, 6)[:, [0, 2]] += centre[[0, 2]]
        self.mass = solver.lumped_mass.numpy().astype(np.float64)
        self.gravity = np.asarray(solver._gravity, dtype=np.float64)
        if np.linalg.norm(self.gravity[[0, 2]]) > 1e-12:
            raise ValueError("This equilibrium model requires vertical Y-up gravity")
        self.spindle_weight = -spindle_mass * self.gravity[1]
        self.rim_load = load + self.spindle_weight
        self.build_pressure = float(build_pressure)
        self.kn, self.ground = solver.kn, solver.ground_z
        self.vref = solver.cavity_volume_ref
        self.sign = solver.elem_psign.numpy().astype(np.float64)
        self.element_dofs = (self.nodes[..., None] * 6 + np.arange(6)).reshape(self.ne, 24)
        self._rows = np.broadcast_to(self.element_dofs[:, :, None], (self.ne, 24, 24)).ravel()
        self._cols = np.broadcast_to(self.element_dofs[:, None, :], (self.ne, 24, 24)).ravel()
        self._reduced_dofs = np.full(self.n * 6, -1, dtype=int)
        self._reduced_dofs[self.free] = np.arange(len(self.free))
        self._reduced_dofs[self.rim_dofs] = self.size - 1
        self._sparsity = ShellSparsity(self._reduced_dofs[self._rows], self._reduced_dofs[self._cols], self.size)
        self._prepare_material(model, solver.thickness_gp)

    def _prepare_material(self, model, thickness_gp):
        material = model.elem_mat.numpy().astype(np.float64)
        self.c = np.zeros((self.ne, 6, 6))
        for i in range(3):
            self.c[:, i, i] = material[:, i]
        for s, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 3):
            self.c[:, i, j] = self.c[:, j, i] = material[:, s]
        self.c[:, 3, 3], self.c[:, 4, 4], self.c[:, 5, 5] = material[:, 8], material[:, 7], material[:, 6]
        h = model.elem_h.numpy()
        cosine, sine = model.elem_fiber_cos.numpy(), model.elem_fiber_sin.numpy()
        zs, ws = np.polynomial.legendre.leggauss(thickness_gp)
        ng = 4 * thickness_gp
        self.hessian = np.empty((self.ne, ng, 6, 8, 8))
        self.weight = np.empty((self.ne, ng))
        self.eas = np.empty((self.ne, ng, 6, 5))
        for e in range(self.ne):
            q0 = self.rest[self.nodes[e]].reshape(8, 3)
            jc = q0.T @ _derivatives(0, 0, 0, h[e])
            tc = _voigt_transform(jc, cosine[e], sine[e])
            corners = [_strain_forms(_derivatives(u, v, 0, h[e]))[2] for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            tying = [_strain_forms(_derivatives(u, v, 0, h[e])) for u, v in ((0, -1), (0, 1), (-1, 0), (1, 0))]
            g = 0
            for u in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
                for v in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
                    for z, weight in zip(zs, ws, strict=True):
                        d = _derivatives(u, v, z, h[e])
                        j0 = q0.T @ d
                        det = abs(np.linalg.det(j0))
                        forms = _strain_forms(d)
                        forms[2] = np.einsum("a,aij->ij", _shape(u, v)[0], corners)
                        forms[4] = (1 - v) / 2 * tying[0][4] + (1 + v) / 2 * tying[1][4]
                        forms[5] = (1 - u) / 2 * tying[2][5] + (1 + u) / 2 * tying[3][5]
                        self.hessian[e, g] = np.einsum("st,tij->sij", _voigt_transform(j0, cosine[e], sine[e]), forms)
                        self.weight[e, g] = weight * det
                        self.eas[e, g] = (
                            tc[:, [0, 1, 2, 3, 3]] * np.array([u, v, z, u, v]) * abs(np.linalg.det(jc)) / det
                        )
                        g += 1
        rest = self.rest[self.nodes].reshape(self.ne, 8, 3)
        self.rest_strain = 0.5 * np.einsum("egsij,eic,ejc->egs", self.hessian, rest, rest, optimize=True)
        self.wc = self.weight[:, :, None, None] * self.c[:, None]
        self.gtcw = np.einsum("egsa,egst->egat", self.eas, self.wc, optimize=True)
        kaa = np.einsum("egas,egsb->eab", self.gtcw, self.eas, optimize=True)
        self.kaa_inv = np.linalg.inv(kaa)
        self._prepare_elastic_polynomial()
        self._pressure_shapes = (
            np.array(
                [_shape(u, v) for u in (-1 / np.sqrt(3), 1 / np.sqrt(3)) for v in (-1 / np.sqrt(3), 1 / np.sqrt(3))]
            )
            .transpose(1, 0, 2)
            .copy()
        )

    def _prepare_elastic_polynomial(self):
        # Subtract x0 from the other positions. The seven remaining vectors
        # have 28 distinct Gram products; strain is linear in those products.
        # Integrate the fixed material and EAS operators once, including all
        # quadrature points. This is the same ANS/EAS energy, without sampling
        # its quadrature again at every Newton/line-search evaluation.
        rows, cols = np.triu_indices(7)
        self._gram_rows, self._gram_cols = rows, cols
        forms = self.hessian[..., 1:, 1:][..., rows, cols] * np.where(rows == cols, 1, 2)
        weighted = (self.wc @ forms).reshape(self.ne, -1, 28)
        forms = forms.reshape(self.ne, -1, 28)
        coupling = self.gtcw.transpose(0, 2, 1, 3).reshape(self.ne, 5, -1) @ forms
        self._gram_eas = self.kaa_inv @ coupling
        self._gram_coupling = coupling
        self._gram_material = forms.transpose(0, 2, 1) @ weighted - coupling.transpose(0, 2, 1) @ self._gram_eas
        rest = self.rest[self.nodes].reshape(self.ne, 8, 3).copy()
        rest[:, 2::2] -= rest[:, :1]
        vectors = rest[:, 1:]
        self._rest_gram = 0.5 * (vectors @ vectors.transpose(0, 2, 1))[:, rows, cols]

    def coordinates(self, positions, directors, height):
        """Pack world ANCF positions [m], directors, and rim height [m]."""
        q = np.stack((positions, directors), axis=1).reshape(-1)
        return np.r_[q[self.free], height].astype(np.float64)

    def _full(self, z):
        q = self.base.copy()
        q[self.free] = z[:-1]
        q[self.rim_dofs] += z[-1]
        return q.reshape(self.n, 2, 3)

    def positions(self, coordinates: np.ndarray) -> np.ndarray:
        """Reconstruct world ANCF node positions [m] from reduced coordinates.

        Returns a new array with shape ``(node_count, 3)`` in the solver's
        Y-up frame, including the bead positions at the solved rim height.
        The source solver and supplied coordinates are unchanged.
        """
        coordinates = np.asarray(coordinates)
        if coordinates.shape != (self.size,) or not np.isfinite(coordinates).all():
            raise ValueError("Expected finite reduced equilibrium coordinates")
        return self._full(coordinates)[:, 0].copy()

    def _elastic(self, q, tangent):
        rows, cols = self._gram_rows, self._gram_cols
        qe = q[self.nodes].reshape(self.ne, 8, 3).copy()
        qe[:, 2::2] -= qe[:, :1]
        vectors = qe[:, 1:]
        gram = 0.5 * (vectors @ vectors.transpose(0, 2, 1))[:, rows, cols] - self._rest_gram
        raw_alpha = -(self._gram_eas @ gram[..., None]).squeeze(-1)
        self._eas_margin = float(np.min(np.abs(np.abs(raw_alpha) - 0.1)))
        correction = np.clip(raw_alpha, -0.1, 0.1) - raw_alpha
        stress = (
            self._gram_material @ gram[..., None] + self._gram_coupling.transpose(0, 2, 1) @ correction[..., None]
        ).squeeze(-1)
        geometric = np.zeros((self.ne, 7, 7))
        geometric[:, rows, cols] = stress * np.where(rows == cols, 1, 0.5)
        geometric[:, cols, rows] = geometric[:, rows, cols]
        fe = np.empty((self.ne, 8, 3))
        fe[:, 1:] = geometric @ vectors
        fe[:, 0] = -fe[:, 2::2].sum(axis=1)
        if not tangent:
            return fe.reshape(self.ne, 24), None
        derivative = np.zeros((self.ne, 28, 7, 3))
        pairs = np.arange(28)
        derivative[:, pairs, rows] += 0.5 * vectors[:, cols]
        derivative[:, pairs, cols] += 0.5 * vectors[:, rows]
        derivative = derivative.reshape(self.ne, 28, 21)
        coupling = self._gram_coupling @ derivative
        ke = derivative.transpose(0, 2, 1) @ (self._gram_material @ derivative)
        # Clipped EAS coordinates have zero derivative. Restore exactly the
        # corresponding rows removed by the preintegrated condensation.
        ke += coupling.transpose(0, 2, 1) @ ((np.abs(raw_alpha) >= 0.1)[..., None] * (self.kaa_inv @ coupling))
        ke += np.einsum("eij,ab->eiajb", geometric, np.eye(3)).reshape(self.ne, 21, 21)
        local = ke.reshape(self.ne, 7, 3, 7, 3)
        full = np.empty((self.ne, 8, 3, 8, 3))
        full[:, 1:, :, 1:] = local
        full[:, 0, :, 1:] = -local[:, 1::2].sum(axis=1)
        full[:, 1:, :, 0] = -local[:, :, :, 1::2].sum(axis=3)
        full[:, 0, :, 0] = local[:, 1::2, :, 1::2].sum(axis=(1, 3))
        return fe.reshape(self.ne, 24), full.reshape(self.ne, 24, 24)

    def _pressure(self, x, tangent):
        xe = x[self.nodes]
        a, b = xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 1]
        area = 0.5 * self.sign[:, None] * np.cross(a, b)
        offset = xe.mean(axis=1) - x.mean(axis=0)
        volume = np.sum(offset * area) / 3
        if volume <= 1e-9:
            raise ValueError("The cavity volume is nonpositive; equilibrium is invalid")
        volume_grad = None
        if tangent:
            dn = (
                0.5
                * self.sign[:, None, None, None]
                * (
                    -np.array([-1, 0, 1, 0])[None, :, None, None] * _skew(b)[:, None]
                    + np.array([0, -1, 0, 1])[None, :, None, None] * _skew(a)[:, None]
                )
            )
            dv = area[:, None] / 12 + np.einsum("eaci,ec->eai", dn, offset) / 3
            volume_grad = np.zeros((self.n, 3))
            np.add.at(volume_grad, self.nodes, dv)
            volume_grad -= area.sum(axis=0) / (3 * self.n)
        n, du, dv = self._pressure_shapes
        tu, tv = du @ xe, dv @ xe
        fe = np.zeros((self.ne, 4, 6))
        fe[:, :, :3] = self.sign[:, None, None] * (n.T @ np.cross(tu, tv))
        ke = np.zeros((self.ne, 4, 6, 4, 6)) if tangent else None
        if tangent:
            dn = -du[None, :, :, None, None] * _skew(tv)[:, :, None]
            dn += dv[None, :, :, None, None] * _skew(tu)[:, :, None]
            blocks = (n.T @ dn.reshape(self.ne, 4, 36)).reshape(self.ne, 4, 4, 3, 3)
            ke[:, :, :3, :, :3] = self.sign[:, None, None, None, None] * blocks.transpose(0, 1, 3, 2, 4)
        return volume, volume_grad, fe.reshape(self.ne, 24), None if ke is None else ke.reshape(self.ne, 24, 24)

    def evaluate(self, z, pressure, tangent=True, *, stiffness_scale=1.0):
        """Evaluate reduced static force residual and its exact local derivative."""
        q = self._full(z)
        fe, ke = self._elastic(q, tangent)
        # Uniform modulus scaling leaves the locally eliminated EAS coordinates unchanged.
        fe *= stiffness_scale
        if tangent:
            ke *= stiffness_scale
        volume, dv, fp, kp = self._pressure(q[:, 0], tangent)
        gauge = pressure * self.vref / volume - self.build_pressure
        residual = np.zeros(self.n * 6)
        unit_pressure = np.zeros_like(residual)
        np.add.at(residual, self.element_dofs, fe - gauge * fp)
        np.add.at(unit_pressure, self.element_dofs, fp)
        residual.reshape(self.n, 6)[:, :3] -= self.mass.reshape(self.n, 6)[:, :3] * self.gravity
        penetration = self.ground - q[:, 0, 1]
        residual[1::6] -= self.kn * np.maximum(penetration, 0.0)
        reduced = np.r_[residual[self.free], residual[self.rim_dofs].sum() + self.rim_load]
        rp_full = -self.vref / volume * unit_pressure
        rp = np.r_[rp_full[self.free], rp_full[self.rim_dofs].sum()]
        if not tangent:
            return reduced, None, rp
        dv_full = np.zeros((self.n, 6))
        dv_full[:, :3] = dv
        gas_derivative = pressure * self.vref / volume**2 * dv_full.ravel()
        active = np.flatnonzero(penetration > 0) * 6 + 1
        contact = np.zeros(self.size)
        mapped = self._reduced_dofs[active]
        np.add.at(contact, mapped[mapped >= 0], self.kn)
        j = SparseTangent(
            self._sparsity.assemble(ke - gauge * kp, contact),
            np.r_[unit_pressure[self.free], unit_pressure[self.rim_dofs].sum()],
            np.r_[gas_derivative[self.free], gas_derivative[self.rim_dofs].sum()],
        )
        return reduced, j, rp

    @staticmethod
    def _solve_scaled(j, rhs):
        return j.solve(rhs)

    def solve(
        self,
        pressure: float,
        initial: np.ndarray,
        tolerance: float = 1e-7,
        max_iterations: int = 30,
        *,
        stiffness_scale: float = 1.0,
    ) -> Result:
        """Converge static forces and differentiate nominal pressure [Pa] and elastic scale.

        A failed equilibrium or sensitivity solve raises rather than returning
        an unchecked gradient. The source dynamic simulation is never modified.
        ``stiffness_scale`` multiplies all elastic moduli, preserving density,
        thickness, Poisson ratios, and reference geometry.
        """
        if not np.isfinite(pressure) or pressure <= 0:
            raise ValueError("Pressure must be finite and positive")
        if not np.isfinite(stiffness_scale) or stiffness_scale <= 0:
            raise ValueError("Stiffness scale must be finite and positive")
        z = np.asarray(initial, dtype=np.float64).copy()
        if z.shape != (self.size,) or not np.isfinite(z).all():
            raise ValueError("Initial equilibrium coordinates have invalid shape or values")
        if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
            raise ValueError("Tolerance and iteration limit must be positive")
        for iteration in range(max_iterations):
            residual, jac, rp = self.evaluate(z, pressure, stiffness_scale=stiffness_scale)
            norm = np.linalg.norm(residual, np.inf)
            if norm <= tolerance:
                contact_margin = np.min(np.abs(self._full(z)[:, 0, 1] - self.ground))
                if contact_margin < 1e-9 or self._eas_margin < 1e-10:
                    raise RuntimeError("Equilibrium is on a contact/EAS switching boundary; derivative is not unique")
                sensitivity = self._solve_scaled(jac, -rp)
                linear_error = np.linalg.norm(jac @ sensitivity + rp) / max(np.linalg.norm(rp), 1e-30)
                elastic, _ = self._elastic(self._full(z), False)
                rs_full = np.zeros(self.n * 6)
                np.add.at(rs_full, self.element_dofs, elastic)
                rs = np.r_[rs_full[self.free], rs_full[self.rim_dofs].sum()]
                stiffness_sensitivity = self._solve_scaled(jac, -rs)
                linear_error = max(
                    linear_error,
                    np.linalg.norm(jac @ stiffness_sensitivity + rs) / max(np.linalg.norm(rs), 1e-30),
                )
                if not np.isfinite([sensitivity, stiffness_sensitivity]).all() or linear_error > 1e-6:
                    raise RuntimeError(f"Pressure sensitivity solve failed: relative residual {linear_error:.3g}")
                return self.Result(
                    z,
                    z[-1],
                    pressure,
                    sensitivity[-1],
                    norm,
                    linear_error,
                    iteration,
                    stiffness_scale,
                    stiffness_sensitivity[-1],
                )
            direction = self._solve_scaled(jac, -residual)
            step = 1.0
            merit = np.linalg.norm(residual)
            for _ in range(20):
                candidate = z + step * direction
                try:
                    candidate_residual, _, _ = self.evaluate(
                        candidate, pressure, tangent=False, stiffness_scale=stiffness_scale
                    )
                    if np.linalg.norm(candidate_residual) < (1 - 1e-4 * step) * merit:
                        z = candidate
                        break
                except ValueError:
                    pass
                step *= 0.5
            else:
                raise RuntimeError(f"Equilibrium line search stalled at residual {norm:.3g}")
        raise RuntimeError(f"Equilibrium did not converge in {max_iterations} iterations (residual {norm:.3g})")

    def pressure_step(self, current: Result, target_height: float, pressure_bounds: tuple[float, float]) -> Result:
        """Backtracked scalar Gauss-Newton step in log pressure [Pa].

        Use the analytic sensitivity, preserve positive pressure, and accept
        only a smaller converged-equilibrium height loss. No finite-difference
        pressure probes are used. Limit each log-pressure step to 0.5.
        """
        if not np.isfinite(target_height):
            raise ValueError("Target height must be finite")
        error = current.height - target_height
        derivative = current.pressure * current.dh_dp
        if abs(derivative) < 1e-12:
            raise RuntimeError("Height is insensitive to pressure at this equilibrium")
        step = float(np.clip(-error / derivative, -0.5, 0.5))
        lower, upper = pressure_bounds
        if not np.isfinite(pressure_bounds).all() or not 0 < lower < upper:
            raise ValueError("Pressure bounds must be positive and increasing")
        for _ in range(12):
            pressure = float(np.clip(current.pressure * np.exp(step), lower, upper))
            if abs(pressure - current.pressure) < 1e-8:
                raise RuntimeError("Target is outside the pressure bounds on this equilibrium branch")
            try:
                candidate = self.solve(pressure, current.coordinates, stiffness_scale=current.stiffness_scale)
                if abs(candidate.height - target_height) < abs(error):
                    return candidate
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass
            step *= 0.5
        raise RuntimeError("No improving pressure step; change the target or initial pressure")
