# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse assembly and solves against independent small dense test oracles."""

import os
import unittest

import newton
import numpy as np
import warp as wp
from warp.sparse import bsr_get_diag

from newton_solver_ancf import SolverANCFCable
from newton_solver_ancf._src.solvers.implicit.sparse import (
    assemble_matrix,
    assembly_indices,
    checked_gmres,
    solve,
)

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


def array(value, dtype=wp.float64):
    return wp.array(value, dtype=dtype, device=DEVICE)


def dense(matrix):
    out = np.zeros((matrix.nrow, matrix.ncol))
    offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    for row in range(matrix.nrow):
        entries = slice(offsets[row], offsets[row + 1])
        out[row, columns[entries]] = values[entries]
    return out


class TestSparse(unittest.TestCase):
    def test_shared_nodes_scrambled_indices_and_storage_scaling(self):
        dofs = np.array([[5, 1, 3], [3, 0, 4], [1, 4, 2]])
        rng = np.random.default_rng(155)
        blocks = rng.normal(size=(3, 3, 3))
        blocks = blocks @ blocks.transpose(0, 2, 1)
        matrix = assemble_matrix(dofs, blocks, 6, DEVICE)
        expected = np.zeros((6, 6))
        for indices, block in zip(dofs, blocks, strict=True):
            expected[np.ix_(indices, indices)] += block
        np.testing.assert_allclose(dense(matrix), expected, atol=1.0e-14)
        slots, rows = assembly_indices(matrix, dofs)
        columns = matrix.columns.numpy()
        for element, indices in enumerate(dofs):
            for i, row in enumerate(indices):
                np.testing.assert_array_equal(rows.numpy()[slots.numpy()[element, i]], np.full(3, row))
                np.testing.assert_array_equal(columns[slots.numpy()[element, i]], indices)
        counts = []
        for n in (100, 200):
            topology = np.column_stack((np.arange(n - 1), np.arange(1, n)))
            mass = assemble_matrix(topology, np.tile(np.eye(2), (n - 1, 1, 1)), n, DEVICE)
            counts.append(int(mass.offsets.numpy()[n]))
            self.assertEqual(counts[-1], 3 * n - 2)
            self.assertEqual(mass.values.ndim, 1)
        self.assertLess(counts[1] / counts[0], 2.02)
        with self.assertRaisesRegex(ValueError, "every generalized"):
            assemble_matrix(np.array([[0, 1]]), np.eye(2)[None], 3, DEVICE)

    def test_invalid_csr_topology_is_rejected_before_scattering(self):
        dofs = np.array([[0, 1]])
        diagonal = assemble_matrix(np.array([[0], [1]]), np.ones((2, 1, 1)), 2, DEVICE)
        with self.assertRaisesRegex(ValueError, "every element clique"):
            assembly_indices(diagonal, dofs)
        matrix = assemble_matrix(dofs, np.eye(2)[None], 2, DEVICE)
        matrix.offsets.assign([0, 3, 2])
        with self.assertRaisesRegex(ValueError, "invalid CSR offsets"):
            assembly_indices(matrix, dofs)
        matrix.offsets.assign([0, 2, 4])
        matrix.columns.assign([0, 2, 0, 1])
        with self.assertRaisesRegex(ValueError, "out-of-range CSR columns"):
            assembly_indices(matrix, dofs)
        matrix.columns.assign([1, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "strictly sorted"):
            assembly_indices(matrix, dofs)

    def test_consistent_mass_freefall(self):
        fem = SolverANCFCable.create_model(
            np.array([[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.4, 0.0, 1.0]]),
            device=DEVICE,
        )
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        fem.register(builder)
        solver = SolverANCFCable(builder.finalize(device=DEVICE), fem)
        state = solver.initial_state()
        acceleration = np.zeros(fem.dof_count)
        acceleration[2::6] = float(solver.gravity[2])
        state.qdd.assign(acceleration)
        state = solver.initialize_history(state)
        dt, steps = 0.002, 3
        for _ in range(steps):
            state = solver.advance(state, dt)
        np.testing.assert_allclose(
            state.q.numpy(), fem.rest_q.numpy() + 0.5 * (steps * dt) ** 2 * acceleration, atol=1e-11
        )
        np.testing.assert_allclose(state.qd.numpy(), steps * dt * acceleration, atol=1e-9)
        np.testing.assert_allclose(state.qdd.numpy(), acceleration, atol=1e-7)

    def test_fixed_coordinates_shift_and_zero_rhs(self):
        rng = np.random.default_rng(37)
        block = rng.normal(size=(7, 7))
        block = block @ block.T + 0.3 * np.eye(7)
        matrix = assemble_matrix(np.arange(7)[None], block[None], 7, DEVICE)
        rhs = rng.normal(size=7)
        fixed = np.array([0, 1, 0, 0, 1, 0, 0], dtype=np.int32)
        free = fixed == 0
        before = matrix.values.numpy().copy()
        for shift in (0.0, 0.2):
            result = solve(matrix, array(rhs), array(fixed, int), regularization=shift, multiplier=-1.0)
            self.assertIsNotNone(result)
            expected = np.zeros(7)
            shifted = block + shift * np.diag(np.diag(block))
            expected[free] = np.linalg.solve(shifted[np.ix_(free, free)], -rhs[free])
            np.testing.assert_allclose(result.numpy(), expected, atol=1.0e-11)
        for method in ("cg", "gmres"):
            zero = solve(matrix, array(np.zeros(7)), array(fixed, int), method=method, transpose=True)
            self.assertIsNotNone(zero)
            np.testing.assert_array_equal(zero.numpy(), np.zeros(7))
        np.testing.assert_array_equal(matrix.values.numpy(), before)

    def test_curvature_failure_and_unshifted_transpose(self):
        indefinite = np.array([[1.0, 2.0], [2.0, 1.0]])
        matrix = assemble_matrix(np.array([[0, 1]]), indefinite[None], 2, DEVICE)
        rhs, fixed = array([1.0, -1.0]), array([0, 0], int)
        self.assertIsNone(solve(matrix, rhs, fixed))
        corrected = solve(matrix, rhs, fixed, regularization=2.0)
        np.testing.assert_allclose(corrected.numpy(), np.linalg.solve(indefinite + 2 * np.eye(2), rhs.numpy()))
        result = solve(matrix, rhs, fixed, method="gmres", transpose=True)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.numpy(), np.linalg.solve(indefinite.T, rhs.numpy()), atol=1.0e-12)
        nonsymmetric = np.array([[4.0, 2.0, 0.0], [-1.0, 3.0, 1.0], [0.3, 0.0, 2.0]])
        matrix = assemble_matrix(np.arange(3)[None], nonsymmetric[None], 3, DEVICE)
        result = solve(matrix, array([1.0, -2.0, 3.0]), array([0, 0, 0], int), method="gmres", transpose=True)
        np.testing.assert_allclose(result.numpy(), np.linalg.solve(nonsymmetric.T, [1.0, -2.0, 3.0]), atol=1.0e-11)
        np.testing.assert_array_equal(bsr_get_diag(matrix).numpy(), np.diag(nonsymmetric))

    def test_insufficient_iteration_budget_rejects_solution(self):
        n = 12
        block = np.diag(np.arange(1.0, n + 1.0))
        matrix = assemble_matrix(np.arange(n)[None], block[None], n, DEVICE)
        self.assertIsNone(checked_gmres(matrix, array(np.ones(n)), maxiter=2, restart=2))


if __name__ == "__main__":
    unittest.main()
