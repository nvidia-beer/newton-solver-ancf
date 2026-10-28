"""Physics, exact element Hessians, implicit gradients and public rigid coupling."""

import importlib.util
import os
import unittest
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from cable_fit import experiment, optimize

from newton_solver_ancf import (
    ANCFState,
    SolverANCF,
    SolverANCFBeam,
    SolverANCFCable,
    SolverANCFShell,
    SolverANCFTetrahedron,
)
from newton_solver_ancf.examples._tip_box import TipBox

DEVICE = os.getenv("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel
def tip_value(q: wp.array[wp.float64], index: int, out: wp.array[wp.float64]):
    out[0] = q[index]


@wp.kernel
def functional(q: wp.array[wp.float64], weights: wp.array[wp.float64], out: wp.array[wp.float64]):
    i = wp.tid()
    wp.atomic_add(out, 0, q[i] * weights[i])


def make_solver(fem, solver_type, **kwargs):
    builder = newton.ModelBuilder()
    fem.register(builder)
    return solver_type(builder.finalize(device=fem.device, requires_grad=True), fem, **kwargs)


def models(requires_grad=False):
    line = np.array([[0.0, 0.0, 0.3], [0.2, 0.0, 0.3]])
    yield (
        SolverANCFCable.create_model(line, device=DEVICE, requires_grad=requires_grad),
        SolverANCFCable._evaluate,
        SolverANCFCable,
    )
    yield (
        SolverANCFBeam.create_model(line, device=DEVICE, requires_grad=requires_grad),
        SolverANCFBeam._evaluate,
        SolverANCFBeam,
    )
    points = np.array([[0.0, 0.0, 0.3], [0.0, 0.1, 0.3], [0.2, 0.0, 0.3], [0.2, 0.1, 0.3]])
    quads = np.array([[0, 2, 3, 1]], dtype=np.int32)
    yield (
        SolverANCFShell.create_model(points, quads, device=DEVICE, requires_grad=requires_grad),
        SolverANCFShell._evaluate,
        SolverANCFShell,
    )
    points = np.array([[0.0, 0.0, 0.3], [0.2, 0.0, 0.3], [0.0, 0.1, 0.3], [0.0, 0.0, 0.4]])
    yield (
        SolverANCFTetrahedron.create_model(
            points, np.array([[0, 1, 2, 3]]), device=DEVICE, requires_grad=requires_grad
        ),
        SolverANCFTetrahedron._evaluate,
        SolverANCFTetrahedron,
    )


class TestANCF(unittest.TestCase):
    def test_typed_solvers_reject_other_element_families(self):
        for fem, _, typed in models():
            builder = newton.ModelBuilder()
            fem.register(builder)
            model = builder.finalize(device=DEVICE)
            self.assertIsInstance(typed(model, fem), SolverANCF)
            with self.assertRaisesRegex(TypeError, "abstract"):
                SolverANCF(model, fem)
            wrong = SolverANCFBeam if typed is SolverANCFCable else SolverANCFCable
            with self.assertRaisesRegex(ValueError, "requires"):
                wrong(model, fem)

    def test_four_exact_element_hessians(self):
        for model, evaluate, _ in models():
            with self.subTest(formulation=model.formulation):
                energy, gradient, hessian = evaluate(model, model.rest_q)
                self.assertLess(abs(float(energy.numpy().sum())), 1.0e-10)
                self.assertLess(np.max(np.abs(gradient.numpy())), 1.0e-7)
                rng = np.random.default_rng(12)
                direction = rng.normal(size=model.dof_count)
                direction.reshape(-1, model.dofs_per_node)[:, :3] *= 0.02
                deformed = model.rest_q.numpy() + 0.02 * direction
                q = wp.array(deformed, dtype=wp.float64, device=DEVICE)
                _, gradient, hessian = evaluate(model, q)
                h = 1.0e-5
                plus = wp.array(deformed + h * direction, dtype=wp.float64, device=DEVICE)
                minus = wp.array(deformed - h * direction, dtype=wp.float64, device=DEVICE)
                fd = (
                    evaluate(model, plus, tangent=False)[1].numpy() - evaluate(model, minus, tangent=False)[1].numpy()
                ) / (2 * h)
                dofs = model.element_dofs.numpy()
                exact = np.einsum("eij,ej->ei", hessian.numpy(), direction[dofs])
                np.testing.assert_allclose(exact, fd, rtol=2.0e-4, atol=2.0e-6)
                energy_fd = (
                    evaluate(model, plus, tangent=False)[0].numpy().sum()
                    - evaluate(model, minus, tangent=False)[0].numpy().sum()
                ) / (2 * h)
                energy_exact = float(np.sum(gradient.numpy() * direction[dofs]))
                self.assertAlmostEqual(energy_exact / energy_fd, 1.0, delta=2.0e-5)

    def test_four_family_implicit_gradients(self):
        for fem, _, solver_type in models(requires_grad=True):
            self._check_implicit_gradients(fem, solver_type)

    def _check_implicit_gradients(self, fem, solver_type):
        solver = make_solver(
            fem,
            solver_type,
            fixed_dofs=range(fem.dofs_per_node),
            gravity=(0, 0, -0.1),
            mass_damping=0.2,
            tolerance=1.0e-10,
        )
        rng = np.random.default_rng(13)
        base = {
            "q": fem.rest_q.numpy().copy(),
            "v": np.zeros(fem.dof_count),
            "a": rng.normal(size=fem.dof_count) * 0.01,
            "internal": rng.normal(size=fem.dof_count) * 0.01,
            "external": rng.normal(size=fem.dof_count) * 0.01,
            "load": rng.normal(size=fem.dof_count) * 0.01,
            "boundary": fem.rest_q.numpy().copy(),
        }
        directions = {
            key: rng.normal(size=fem.dof_count) * scale
            for key, scale in (
                ("q", 0.001),
                ("v", 0.01),
                ("a", 0.01),
                ("internal", 0.01),
                ("external", 0.01),
                ("load", 0.01),
                ("boundary", 0.001),
            )
        }
        parameter = next(iter(fem.parameters.values()))
        reference = parameter.numpy().copy()
        material_direction = 0.03 * reference
        weights = wp.array(rng.normal(size=fem.dof_count), dtype=wp.float64, device=DEVICE)

        def run(perturb=None, epsilon=0.0, gradient=False):
            for array in fem.parameters.values():
                array.grad.zero_()
            parameter.assign(reference + (epsilon * material_direction if perturb == "material" else 0.0))
            arrays = {
                key: wp.array(
                    value + (epsilon * directions[key] if perturb == key else 0.0),
                    dtype=wp.float64,
                    device=DEVICE,
                    requires_grad=gradient,
                )
                for key, value in base.items()
            }
            loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=gradient)

            def forward():
                state = solver.advance(
                    ANCFState(arrays["q"], arrays["v"], arrays["a"], arrays["internal"], arrays["external"]),
                    0.005,
                    external_force=arrays["load"],
                    boundary_q=arrays["boundary"],
                )
                wp.launch(functional, fem.dof_count, inputs=[state.q, weights, loss], device=DEVICE)

            if gradient:
                with wp.Tape() as tape:
                    forward()
                tape.backward(loss)
                derivatives = {
                    key: float(np.sum(array.grad.numpy() * directions[key])) for key, array in arrays.items()
                }
                derivatives["material"] = float(np.sum(parameter.grad.numpy() * material_direction))
                return derivatives
            forward()
            return float(loss.numpy()[0])

        for variable, analytic in run(gradient=True).items():
            with self.subTest(formulation=fem.formulation, variable=variable):
                epsilon = 1.0e-4
                finite = (run(variable, epsilon) - run(variable, -epsilon)) / (2.0 * epsilon)
                self.assertAlmostEqual(analytic, finite, delta=max(1.0e-8, 1.0e-3 * abs(finite)))

    def test_material_rollout_gradient(self):
        rollout = experiment(DEVICE)
        target = rollout(2.0)
        self.assertEqual(target.shape, (8, 18))
        value, gradient = rollout(1.0, target, True)
        self.assertGreater(value, 0.0)
        for epsilon in (3.0e-4, 1.0e-4):
            finite = (rollout(np.exp(epsilon), target) - rollout(np.exp(-epsilon), target)) / (2 * epsilon)
            self.assertAlmostEqual(gradient / finite, 1.0, delta=2.0e-3)

    def test_material_recovery(self):
        scale, initial, final = optimize(DEVICE)
        self.assertAlmostEqual(scale, 2.0, delta=2.0e-3)
        self.assertLess(final, initial * 0.01)

    def test_equilibrium_load_gradient(self):
        model = SolverANCFCable.create_model(np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]), device=DEVICE)
        solver = make_solver(model, SolverANCFCable, fixed_dofs=range(6), gravity=(0, 0, 0), tolerance=1.0e-10)
        self.assertIsInstance(solver, newton.solvers.SolverBase)
        native = solver.model
        start, result = native.state(), native.state()
        control = native.control()
        load = control.ancf.load
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        with wp.Tape() as tape:
            solver.step(start, result, control, None, 0.01)
            wp.launch(tip_value, 1, inputs=[result.ancf.q, 8, loss], device=DEVICE)
        tape.backward(loss)
        np.testing.assert_array_equal(start.ancf.q.numpy(), model.rest_q.numpy())
        solver.reset(result)
        np.testing.assert_array_equal(result.ancf.q.numpy(), model.rest_q.numpy())
        gradient = float(load.grad.numpy()[8])
        self.assertGreater(gradient, 0.0)
        values = []
        for sign in (-1, 1):
            array = np.zeros(model.dof_count)
            array[8] = sign * 1.0e-5
            result = solver.advance(
                solver.initial_state(), 0.01, external_force=wp.array(array, dtype=wp.float64, device=DEVICE)
            )
            values.append(result.q.numpy()[8])
        self.assertAlmostEqual(gradient / ((values[1] - values[0]) / 2.0e-5), 1.0, delta=1.0e-4)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"),
        "Install the examples extra for MuJoCo coupling tests",
    )
    def test_public_mujoco_tip_coupling(self):
        model = SolverANCFCable.create_model(
            np.array([[0.0, 0.0, 0.4], [0.1, 0.0, 0.4], [0.2, 0.0, 0.4]]), device=DEVICE, radius=0.005
        )
        solver = make_solver(model, SolverANCFCable, fixed_dofs=range(6), mass_damping=2.0)
        example = SimpleNamespace(solver=solver, state=solver.initial_state(), sim_time=0.0)
        box = TipBox(example, 0.03, (0.005, 0.1))
        for _ in range(3):
            box.step(1 / 240)
        box.test_final()
        self.assertGreater(np.linalg.norm(box.coupler.last_wrench.numpy()), 1.0e-5)
        box.set_mass(0.04)
        box.step(1 / 240)
        box.test_final()
        with self.assertRaisesRegex(RuntimeError, "backward"):
            with wp.Tape():
                box.step(1 / 240)


if __name__ == "__main__":
    unittest.main()
