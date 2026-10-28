# SPDX-License-Identifier: Apache-2.0
"""Reference kinematics and stock public articulation workspaces."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_ancf._src.solvers.coupled.articulation import (
    compute_body_spatial_inertia,
    jcalc_integrate,
)

DEVICE = os.getenv("NEWTON_TEST_DEVICE", "cpu")


@wp.kernel
def integrate(
    kind: int,
    linear: int,
    angular: int,
    q: wp.array[float],
    v: wp.array[float],
    a: wp.array[float],
    dt: float,
    q_out: wp.array[float],
    v_out: wp.array[float],
):
    jcalc_integrate(-1, wp.transform_identity(), wp.vec3(0.0), kind, q, v, a, 0, 0, linear, angular, dt, q_out, v_out)


class TestReferenceArticulation(unittest.TestCase):
    def test_reference_joint_integration(self):
        dt = 0.02
        velocity = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3], dtype=np.float32)
        acceleration = np.array([-0.1, 0.2, 0.3, -0.2, 0.1, 0.05], dtype=np.float32)
        for kind, linear, angular, count in (
            (newton.JointType.PRISMATIC, 1, 0, 1),
            (newton.JointType.REVOLUTE, 0, 1, 1),
            (newton.JointType.D6, 3, 3, 6),
            (newton.JointType.BALL, 0, 3, 3),
        ):
            with self.subTest(kind=kind):
                position = np.linspace(0.05, 0.35, 7, dtype=np.float32)
                if kind == newton.JointType.BALL:
                    position[:4] = [0.0, 0.0, 0.0, 1.0]
                q = wp.array(position, device=DEVICE)
                v = wp.array(velocity, device=DEVICE)
                a = wp.array(acceleration, device=DEVICE)
                q_out, v_out = wp.clone(q), wp.clone(v)
                wp.launch(integrate, dim=1, inputs=[kind, linear, angular, q, v, a, dt, q_out, v_out], device=DEVICE)
                expected_v = velocity[:count] + dt * acceleration[:count]
                np.testing.assert_allclose(v_out.numpy()[:count], expected_v, rtol=1e-6, atol=1e-7)
                expected_q = position.copy()
                if kind == newton.JointType.BALL:
                    expected_q[:4] = np.r_[0.5 * dt * expected_v, 1.0]
                    expected_q[:4] /= np.linalg.norm(expected_q[:4])
                else:
                    expected_q[:count] += dt * expected_v
                np.testing.assert_allclose(q_out.numpy(), expected_q, rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(q.numpy(), position)
                np.testing.assert_array_equal(v.numpy(), velocity)

    def test_stock_jacobian_mass_preallocated_workspaces(self):
        builder = newton.ModelBuilder()
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, -1.0)), 0.7)
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.3, -0.1, 0.5), rotation),
            com=wp.vec3(0.02, -0.01, 0.03),
            mass=2.5,
            inertia=wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4),
            lock_inertia=True,
        )
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        model = builder.finalize(device=DEVICE)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        expected_j = newton.eval_jacobian(model, state).numpy().copy()
        expected_h = newton.eval_mass_matrix(model, state).numpy().copy()
        jacobian = wp.empty(expected_j.shape, dtype=float, device=DEVICE)
        mass = wp.empty(expected_h.shape, dtype=float, device=DEVICE)
        motions = wp.empty(model.joint_dof_count, dtype=wp.spatial_vector, device=DEVICE)
        inertias = wp.empty(model.body_count, dtype=wp.spatial_matrix, device=DEVICE)
        newton.eval_jacobian(model, state, J=jacobian, joint_S_s=motions)
        newton.eval_mass_matrix(model, state, H=mass, J=jacobian, body_I_s=inertias, joint_S_s=motions)
        expected_inertias = inertias.numpy().copy()
        wp.launch(
            compute_body_spatial_inertia,
            dim=model.body_count,
            inputs=[model.body_inertia, model.body_mass, state.body_q, inertias],
            device=DEVICE,
        )
        np.testing.assert_allclose(jacobian.numpy(), expected_j, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(mass.numpy(), expected_h, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(inertias.numpy(), expected_inertias, rtol=1e-6, atol=1e-7)
        j = jacobian.numpy()[0]
        np.testing.assert_allclose(mass.numpy()[0], j.T @ inertias.numpy()[0] @ j, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
