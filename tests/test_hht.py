# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent oscillator accuracy and full-history HHT adjoint checks."""

import os
import unittest
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from newton_solver_ancf._src.solvers.implicit import ImplicitModel, ImplicitState, SolverImplicit
from newton_solver_ancf._src.solvers.implicit.sparse import assemble_matrix

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")


@dataclass
class SpringData:
    stiffness: wp.array

    @property
    def parameters(self):
        return {"stiffness": self.stiffness}


@wp.kernel
def spring_force(
    q: wp.array[wp.float64],
    stiffness: wp.array[wp.float64],
    energy: wp.array[wp.float64],
    gradient: wp.array2d[wp.float64],
):
    value = wp.float64(0.0)
    for i in range(3):
        gradient[0, i] = stiffness[0] * q[i]
        value += wp.float64(0.5) * stiffness[0] * q[i] * q[i]
    energy[0] = value


@wp.kernel(enable_backward=False)
def spring_tangent(stiffness: wp.array[wp.float64], tangent: wp.array3d[wp.float64]):
    i, j = wp.tid()
    tangent[0, i, j] = stiffness[0] if i == j else wp.float64(0.0)


@wp.kernel
def state_loss(
    q: wp.array[wp.float64],
    v: wp.array[wp.float64],
    a: wp.array[wp.float64],
    internal: wp.array[wp.float64],
    external: wp.array[wp.float64],
    weights: wp.array2d[wp.float64],
    loss: wp.array[wp.float64],
):
    i = wp.tid()
    wp.atomic_add(
        loss,
        0,
        q[i] * weights[0, i]
        + v[i] * weights[1, i]
        + a[i] * weights[2, i]
        + internal[i] * weights[3, i]
        + external[i] * weights[4, i],
    )


class Oscillator(SolverImplicit):
    """Three independent physical mass-spring coordinates; no ANCF approximation."""

    element_type = SpringData

    @staticmethod
    def _evaluate(model, q, *, tangent):
        stiffness = model.element_data.stiffness
        requires_grad = q.requires_grad or stiffness.requires_grad
        energy = wp.empty(1, dtype=wp.float64, device=q.device, requires_grad=requires_grad)
        gradient = wp.empty((1, 3), dtype=wp.float64, device=q.device, requires_grad=requires_grad)
        wp.launch(spring_force, 1, inputs=[q, stiffness, energy, gradient], device=q.device)
        hessian = None
        if tangent:
            hessian = wp.empty((1, 3, 3), dtype=wp.float64, device=q.device)
            wp.launch(spring_tangent, (3, 3), inputs=[stiffness, hessian], device=q.device)
        return energy, gradient, hessian


def oscillator(*, damping=0.0, requires_grad=False):
    mass = 2.0 * np.eye(3)
    model = ImplicitModel(
        formulation="oscillator",
        rest_q=wp.zeros(3, dtype=wp.float64, device=DEVICE),
        elements=wp.array([[0]], dtype=int, device=DEVICE),
        element_dofs=wp.array([[0, 1, 2]], dtype=int, device=DEVICE),
        lengths=wp.ones(1, dtype=wp.float64, device=DEVICE),
        mass_matrix=assemble_matrix(np.array([[0, 1, 2]]), mass[None], 3, DEVICE),
        node_count=1,
        element_count=1,
        dofs_per_node=3,
        nodes_per_element=1,
        device=wp.get_device(DEVICE),
        element_data=SpringData(wp.array([25.0], dtype=wp.float64, device=DEVICE, requires_grad=requires_grad)),
    )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    model.register(builder)
    return Oscillator(builder.finalize(device=DEVICE), model, mass_damping=damping, tolerance=1.0e-10)


def arrays(state):
    return state.q, state.qd, state.qdd, state.internal, state.external


class TestHHT(unittest.TestCase):
    def test_second_order_oscillator(self):
        solver = oscillator()
        omega = np.sqrt(25.0 / 2.0)
        final_time, q0, v0 = 0.8, 0.2, 0.1
        exact_q = q0 * np.cos(omega * final_time) + v0 * np.sin(omega * final_time) / omega
        exact_v = -q0 * omega * np.sin(omega * final_time) + v0 * np.cos(omega * final_time)
        errors = []
        for steps in (10, 20, 40):
            state = solver.initial_state()
            state.q.assign([q0, 0.0, 0.0])
            state.qd.assign([v0, 0.0, 0.0])
            # Supply a physically consistent initial acceleration, independently
            # of initialize_history, which only evaluates current force history.
            state.qdd.assign([-omega * omega * q0, 0.0, 0.0])
            state = solver.initialize_history(state)
            for _ in range(steps):
                state = solver.advance(state, final_time / steps)
            q, velocity = state.q.numpy()[0], state.qd.numpy()[0]
            errors.append(np.hypot(q - exact_q, (velocity - exact_v) / omega))
        ratios = np.asarray(errors[:-1]) / errors[1:]
        self.assertTrue(np.all(ratios > 3.5), f"HHT refinement errors {errors}, ratios {ratios}")
        self.assertLess(errors[-1], 1.0e-3)

    def test_multistep_state_history_and_time_varying_load_gradients(self):
        solver = oscillator(damping=0.15)
        rng = np.random.default_rng(162)
        values = {
            "q": np.array([0.2, -0.03, 0.04]),
            "v": np.array([0.1, 0.02, -0.01]),
            "a": np.array([-2.5, 0.375, -0.5]),
            "internal": np.array([5.03, -0.744, 0.997]),
            "external": np.array([0.1, 0.2, -0.1]),
        }
        for i in range(4):
            values[f"load{i}"] = rng.normal(size=3) * (i + 1)
        directions = {key: rng.normal(size=3) for key in values}
        weights = wp.array(
            rng.normal(size=(5, 3)) * np.array([1.0, 0.1, 0.01, 0.02, 0.03])[:, None],
            dtype=wp.float64,
            device=DEVICE,
        )

        def run(variable=None, epsilon=0.0, gradient=False):
            inputs = {
                key: wp.array(
                    value + (epsilon * directions[key] if key == variable else 0.0),
                    dtype=wp.float64,
                    device=DEVICE,
                    requires_grad=gradient,
                )
                for key, value in values.items()
            }
            initial = ImplicitState(*(inputs[key] for key in ("q", "v", "a", "internal", "external")))
            loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=gradient)

            def forward():
                state = initial
                for i in range(4):
                    state = solver.advance(state, 0.03, external_force=inputs[f"load{i}"])
                wp.launch(state_loss, 3, inputs=[*arrays(state), weights, loss], device=DEVICE)
                return state

            if not gradient:
                forward()
                return float(loss.numpy()[0])
            with wp.Tape() as tape:
                final = forward()
            tape.backward(loss)
            np.testing.assert_allclose(final.external.numpy(), values["load3"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                final.internal.numpy(), 25 * final.q.numpy() + 0.3 * final.qd.numpy(), rtol=1e-12
            )
            for name in ("q", "v", "a", "internal", "external"):
                np.testing.assert_array_equal(inputs[name].numpy(), values[name])
            return {name: float(value.grad.numpy() @ directions[name]) for name, value in inputs.items()}

        for variable, derivative in run(gradient=True).items():
            self.assertGreater(abs(derivative), 1.0e-8, variable)
            for epsilon in (1.0e-4, 3.0e-5):
                with self.subTest(input=variable, epsilon=epsilon):
                    finite = (run(variable, epsilon) - run(variable, -epsilon)) / (2 * epsilon)
                    self.assertAlmostEqual(derivative, finite, delta=max(2.0e-9, 1.0e-5 * abs(finite)))

    def test_initial_force_history_material_gradient(self):
        solver = oscillator(damping=0.15, requires_grad=True)
        parameter = solver.fem.parameters["stiffness"]
        weights = wp.array(np.ones((5, 3)) * 0.01, dtype=wp.float64, device=DEVICE)

        def run(stiffness, gradient=False):
            parameter.assign([stiffness])
            parameter.grad.zero_()
            state = solver.initial_state(requires_grad=gradient)
            state.q.assign([0.2, 0.04, -0.03])
            state.qd.assign([0.1, -0.02, 0.01])
            # Keep the specified acceleration fixed while differentiating k.
            state.qdd.assign([-2.5, -0.5, 0.375])
            loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=gradient)

            def forward():
                end = solver.initialize_history(state)
                for _ in range(4):
                    end = solver.advance(end, 0.03)
                wp.launch(state_loss, 3, inputs=[*arrays(end), weights, loss], device=DEVICE)

            if gradient:
                with wp.Tape() as tape:
                    forward()
                tape.backward(loss)
                return float(parameter.grad.numpy()[0])
            forward()
            return float(loss.numpy()[0])

        exact = run(25.0, gradient=True)
        self.assertGreater(abs(exact), 1.0e-6)
        for epsilon in (1.0e-3, 3.0e-4):
            finite = (run(25.0 + epsilon) - run(25.0 - epsilon)) / (2 * epsilon)
            self.assertAlmostEqual(exact, finite, delta=max(1.0e-10, 1.0e-5 * abs(finite)))
