"""Analytic mass and uniform-acceleration checks for the reference shell paths."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_ancf import ANCFShellModel, SolverANCFTire, isotropic_ancf_material


class TestReferenceShell(unittest.TestCase):
    def test_single_and_batched_uniform_gravity(self):
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        material = isotropic_ancf_material(1.0e5, 0.3, 1000.0, 0.0)
        fields = ("C11", "C22", "C33", "C12", "C13", "C23", "G23", "G13", "G12", "rho", "alpha_damp")
        row = [getattr(material, name) for name in fields]
        positions = np.array([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]], dtype=np.float32)
        dt = 1.0 / 600.0
        for count in (1, 2):
            with self.subTest(environments=count):
                builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
                model = builder.finalize(device=device)
                shell = ANCFShellModel(
                    n_nodes=4,
                    n_elems=1,
                    node_x0=wp.array(positions, dtype=wp.vec3, device=device),
                    node_D0=wp.array(np.tile([0, 0, 1], (4, 1)), dtype=wp.vec3, device=device),
                    elem_nodes=wp.array([[0, 1, 2, 3]], dtype=wp.int32, device=device),
                    elem_h=wp.array([0.05], dtype=float, device=device),
                    elem_mat=wp.array([row], dtype=float, device=device),
                    elem_eas_alpha=wp.zeros((1, 5), device=device),
                    elem_fiber_cos=wp.ones(1, device=device),
                    elem_fiber_sin=wp.zeros(1, device=device),
                    device=device,
                )
                solver = SolverANCFTire(model, shell, n_envs=count, ground_z=-10.0, nr_max_iter=2, pcg_max_iter=10)
                # capture_graph prepares this immutable timestep-dependent mass.
                # Set the same public workspace for an uncaptured CPU kernel check.
                solver.lumped_mass_scaled_tiled.assign(solver.lumped_mass_tiled.numpy() / (solver.hht_beta * dt * dt))
                solver.step(None, None, None, None, dt)

                masses = solver.lumped_mass.numpy().reshape((-1, 6))
                np.testing.assert_allclose(masses[:, :3].sum(axis=0), 1000.0 * 1.0 * 0.05, rtol=2.0e-6)
                # The source starts with zero acceleration and force history;
                # its first HHT acceleration is (1 + alpha) times gravity.
                gravity = model.gravity.numpy()[0]
                acceleration = (1.0 + solver.hht_alpha) * gravity
                expected = np.tile(acceleration, (4 * count, 1))
                np.testing.assert_allclose(solver.node_xdd.numpy(), expected, rtol=2.0e-5, atol=1.0e-7)
                np.testing.assert_allclose(
                    solver.node_xd.numpy(), solver.hht_gamma * dt * expected, rtol=2.0e-5, atol=1.0e-8
                )
                np.testing.assert_allclose(
                    solver.node_x.numpy(),
                    np.tile(positions, (count, 1)) + solver.hht_beta * dt * dt * expected,
                    atol=2.0e-7,
                )
                np.testing.assert_allclose(solver.node_D.numpy(), np.tile([0, 0, 1], (4 * count, 1)), atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
