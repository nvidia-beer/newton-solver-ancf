# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Smooth floor energy, exact contact tangent, and implicit impact derivatives."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_ancf import SolverANCFCable, SolverANCFTetrahedron
from newton_solver_ancf._src.solvers.implicit.solver import _ground_terms

DEVICE = os.getenv("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel
def floor_terms(z: wp.array[wp.float64], values: wp.array[wp.vec3d]):
    i = wp.tid()
    values[i] = _ground_terms(wp.float64(0.3), z[i], wp.float64(1300.0), wp.float64(0.001))


@wp.kernel
def height_loss(q: wp.array[wp.float64], dofs_per_node: int, loss: wp.array[wp.float64]):
    node = wp.tid()
    wp.atomic_add(loss, 0, q[dofs_per_node * node + 2] / wp.float64(q.shape[0] // dofs_per_node))


def make_solver(fem, solver_type, **options):
    builder = newton.ModelBuilder()
    fem.register(builder)
    model = builder.finalize(device=DEVICE, requires_grad=True)
    return solver_type(model, fem, **options)


def tetrahedron(height=0.001, requires_grad=False):
    points = np.array([[0, 0, height], [0.04, 0, height], [0, 0.04, height], [0, 0, height + 0.04]])
    return SolverANCFTetrahedron.create_model(
        points,
        np.array([[0, 1, 2, 3]]),
        young_modulus=2.0e4,
        density=1000.0,
        device=DEVICE,
        requires_grad=requires_grad,
    )


class TestFloorContact(unittest.TestCase):
    def test_energy_gradient_and_curvature(self):
        z = np.array([0.295, 0.299, 0.3, 0.301, 0.305])

        def evaluate(points):
            values = wp.empty(len(points), dtype=wp.vec3d, device=DEVICE)
            wp.launch(floor_terms, len(points), inputs=[wp.array(points, dtype=wp.float64, device=DEVICE), values])
            return values.numpy().copy()

        values = evaluate(z)
        h = 1.0e-7
        differences = (evaluate(z + h) - evaluate(z - h)) / (2.0 * h)
        np.testing.assert_allclose(values[:, 1], differences[:, 0], rtol=2.0e-8)
        np.testing.assert_allclose(values[:, 2], differences[:, 1], rtol=2.0e-8)
        self.assertTrue((values[:, 0] > 0.0).all())
        self.assertTrue((values[:, 1] < 0.0).all())
        self.assertTrue((values[:, 2] > 0.0).all())

    def test_floor_contributes_only_to_position_z(self):
        fem = SolverANCFCable.create_model(np.array([[0.0, 0.0, 0.3], [0.2, 0.0, 0.3]]), device=DEVICE)
        free = make_solver(fem, SolverANCFCable)
        floor = make_solver(fem, SolverANCFCable, ground_height=0.3, ground_stiffness=1300, contact_smoothing=0.001)
        state = free.initial_state()
        load = wp.zeros(fem.dof_count, dtype=wp.float64, device=DEVICE)
        r0, h0, _, e0 = free._residual(fem, state.q, state, 0.01, load, tangent=True)
        r1, h1, _, e1 = floor._residual(fem, state.q, state, 0.01, load, tangent=True)
        expected_force, expected_diagonal = np.zeros(fem.dof_count), np.zeros(fem.dof_count)
        expected_force[2::6] = 0.8 * -0.325
        expected_diagonal[2::6] = 0.8 * 650.0
        np.testing.assert_allclose(r1.numpy() - r0.numpy(), expected_force, atol=1.0e-12)
        offsets, columns = h0.offsets.numpy(), h0.columns.numpy()[: h0.nnz]
        np.testing.assert_array_equal(h1.offsets.numpy(), offsets)
        np.testing.assert_array_equal(h1.columns.numpy()[: h1.nnz], columns)
        rows = np.repeat(np.arange(fem.dof_count), np.diff(offsets))
        expected_values = np.where(rows == columns, expected_diagonal[rows], 0.0)
        np.testing.assert_allclose(
            h1.values.numpy()[: h1.nnz] - h0.values.numpy()[: h0.nnz], expected_values, atol=1.0e-10
        )
        self.assertAlmostEqual(float((e1.numpy() - e0.numpy()).sum()), 0.8 * 0.000325, delta=1.0e-12)

    def test_native_impact_rollout_gradients(self):
        self._check_impact_rollout_gradients(tetrahedron(requires_grad=True), SolverANCFTetrahedron)

    def _check_impact_rollout_gradients(self, fem, solver_type):
        solver = make_solver(fem, solver_type, ground_height=0.0, mass_damping=2.0, tolerance=1.0e-10)
        native = solver.model
        stride = fem.dofs_per_node
        rest = fem.rest_q.numpy().copy()
        velocity = np.zeros(fem.dof_count)
        velocity[2::stride] = -0.1
        position_direction, velocity_direction = np.zeros_like(rest), np.zeros_like(rest)
        position_direction[2::stride] = 0.001
        velocity_direction[2::stride] = 0.1
        references = {name: array.numpy().copy() for name, array in fem.parameters.items()}

        def rollout(variable=None, epsilon=0.0, gradient=False):
            for name, array in fem.parameters.items():
                array.assign(references[name] * (1.0 + epsilon if variable == "material" else 1.0))
                array.grad.zero_()
            states = [native.state() for _ in range(9)]
            q0 = rest + (epsilon * position_direction if variable == "position" else 0.0)
            v0 = velocity + (epsilon * velocity_direction if variable == "velocity" else 0.0)
            states[0].ancf.q.assign(q0)
            states[0].ancf.qd.assign(v0)
            loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)

            def forward():
                for before, after in zip(states[:-1], states[1:], strict=True):
                    solver.step(before, after, None, None, 0.005)
                wp.launch(height_loss, fem.node_count, inputs=[states[-1].ancf.q, stride, loss], device=DEVICE)

            if gradient:
                with wp.Tape() as tape:
                    forward()
                tape.backward(loss)
                result = {
                    "position": float(states[0].ancf.q.grad.numpy() @ position_direction),
                    "velocity": float(states[0].ancf.qd.grad.numpy() @ velocity_direction),
                    "material": sum(
                        float(np.sum(array.grad.numpy() * references[name])) for name, array in fem.parameters.items()
                    ),
                }
                np.testing.assert_array_equal(states[0].ancf.q.numpy(), q0)
                np.testing.assert_array_equal(states[0].ancf.qd.numpy(), v0)
                self.assertLess(float(states[-1].ancf.q.numpy()[2::stride].min()), 0.0)
                return result
            forward()
            return float(loss.numpy()[0])

        for variable, derivative in rollout(gradient=True).items():
            self.assertGreater(abs(derivative), 1.0e-8)
            for epsilon in (1.0e-3, 3.0e-4):
                with self.subTest(variable=variable, epsilon=epsilon):
                    finite = (rollout(variable, epsilon) - rollout(variable, -epsilon)) / (2.0 * epsilon)
                    self.assertAlmostEqual(derivative, finite, delta=max(1.0e-9, 2.0e-3 * abs(finite)))

    def test_invalid_floor_settings(self):
        fem = tetrahedron()
        for options in (
            {"ground_height": np.nan},
            {"ground_stiffness": -1.0},
            {"contact_smoothing": 0.0},
            {"contact_smoothing": np.inf},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_solver(fem, SolverANCFTetrahedron, **options)


if __name__ == "__main__":
    unittest.main()
