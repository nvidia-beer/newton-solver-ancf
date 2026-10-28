# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced sparse tire tangents, nonsymmetric solves, and physical derivatives."""

import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_ancf import ANCFTireEquilibrium, SolverANCFTire, examples, load_ancf_tire_usd
from newton_solver_ancf._src.solvers.ancf._sparse_tangent import ShellSparsity, SparseTangent


class TestSparseTireTangent(unittest.TestCase):
    def test_reduced_assembly_nonsymmetric_solve_and_transpose(self):
        # Two original coordinates move with one rim height; another is fixed.
        mapping = np.array([-1, 0, 1, 3, 2, 3])
        full = np.diag([2.0, 4.0, -3.0, 5.0, 6.0, 7.0])
        full[1, 2], full[2, 1] = 0.4, -0.8
        full[3, 4], full[4, 5] = 0.3, -0.2
        rows, columns = np.indices(full.shape)
        pattern = ShellSparsity(mapping[rows.ravel()], mapping[columns.ravel()], 4)
        projection = np.zeros((6, 4))
        projection[np.flatnonzero(mapping >= 0), mapping[mapping >= 0]] = 1.0
        left = np.array([0.1, 0.2, 0.3, 0.1])
        right = np.array([-0.3, 0.5, 0.1, -0.1])
        diagonal = np.array([0.2, 0.1, 0.0, 0.4])
        matrix = SparseTangent(pattern.assemble(full, diagonal), left, right)
        reference = projection.T @ full @ projection + np.diag(diagonal) + np.outer(left, right)
        rhs = np.array([[1.0, -0.4], [0.5, 2.0], [3.0, -1.0], [0.2, 1.0]])
        np.testing.assert_allclose(matrix @ rhs, reference @ rhs, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(matrix.diagonal(), np.diag(reference), rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(matrix.solve(rhs), np.linalg.solve(reference, rhs), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(matrix.T.solve(rhs), np.linalg.solve(reference.T, rhs), rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(matrix.solve(np.zeros(4)), np.zeros(4))

    def test_singular_solve_is_rejected(self):
        pattern = ShellSparsity(np.arange(3), np.arange(3), 3)
        matrix = SparseTangent(pattern.assemble(np.zeros(3), np.zeros(3)), np.zeros(3), np.zeros(3))
        with self.assertRaisesRegex(RuntimeError, "tangent is singular"):
            matrix.solve(np.ones(3))

    def test_cavity_can_stabilize_singular_local_matrix(self):
        pattern = ShellSparsity(np.arange(3), np.arange(3), 3)
        local = pattern.assemble(np.array([2.0, 0.0, -3.0]), np.zeros(3))
        matrix = SparseTangent(local, np.array([0.5, 1.0, 0.0]), np.array([0.0, 2.0, 0.0]))
        rhs = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(matrix @ matrix.solve(rhs), rhs, rtol=1e-13, atol=1e-13)

    def test_authored_shell_force_and_parameter_derivatives(self):
        path = Path(examples.__file__).parent / "assets" / "minimal_ball.usda"
        with wp.ScopedDevice("cpu"):
            shell, _ = load_ancf_tire_usd(str(path), device="cpu")
            builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
            model = builder.finalize(device="cpu")
            rest = shell.node_x0.numpy().astype(np.float64)
            solver = SolverANCFTire(model, shell, ground_z=float(rest[:, 1].min() + 0.08), kn=2e4)
            beads = np.argsort(rest[:, 1])[-4:]
            equilibrium = ANCFTireEquilibrium(solver, beads, spindle_mass=2.0, build_pressure=110_000.0)
            positions = rest.copy()
            positions[:, 1] += 0.03
            z = equilibrium.coordinates(positions, shell.node_D0.numpy(), 0.03)
            random = np.random.default_rng(9)
            z += random.normal(0.0, 1e-5, z.shape)
            pressure, scale = 120_000.0, 1.1
            residual, tangent, pressure_derivative = equilibrium.evaluate(z, pressure, stiffness_scale=scale)
            self.assertIsInstance(tangent, SparseTangent)
            self.assertLess(tangent.local.nnz, equilibrium.size**2 // 2)
            direction = random.normal(size=z.shape)
            direction /= np.linalg.norm(direction)
            epsilon = 1e-6
            upper = equilibrium.evaluate(z + epsilon * direction, pressure, tangent=False, stiffness_scale=scale)[0]
            lower = equilibrium.evaluate(z - epsilon * direction, pressure, tangent=False, stiffness_scale=scale)[0]
            numerical = (upper - lower) / (2 * epsilon)
            np.testing.assert_allclose(tangent @ direction, numerical, rtol=2e-5, atol=2e-5)
            upper = equilibrium.evaluate(z, pressure + 1.0, tangent=False, stiffness_scale=scale)[0]
            lower = equilibrium.evaluate(z, pressure - 1.0, tangent=False, stiffness_scale=scale)[0]
            np.testing.assert_allclose(pressure_derivative, (upper - lower) / 2, rtol=1e-9, atol=1e-10)
            elastic, _ = equilibrium._elastic(equilibrium._full(z), False)
            full_scale_derivative = np.zeros(equilibrium.n * 6)
            np.add.at(full_scale_derivative, equilibrium.element_dofs, elastic)
            scale_derivative = np.r_[
                full_scale_derivative[equilibrium.free], full_scale_derivative[equilibrium.rim_dofs].sum()
            ]
            upper = equilibrium.evaluate(z, pressure, tangent=False, stiffness_scale=scale + epsilon)[0]
            lower = equilibrium.evaluate(z, pressure, tangent=False, stiffness_scale=scale - epsilon)[0]
            np.testing.assert_allclose(scale_derivative, (upper - lower) / (2 * epsilon), rtol=1e-5, atol=1e-6)
            # The same nonsymmetric operator supplies pressure and material
            # responses; checking in physical units catches scaling mistakes.
            rhs = np.column_stack((-pressure_derivative, -scale_derivative))
            response = tangent.solve(rhs)
            np.testing.assert_allclose(tangent @ response, rhs, rtol=1e-7, atol=1e-8)
            self.assertTrue(np.isfinite(residual).all())


if __name__ == "__main__":
    unittest.main()
