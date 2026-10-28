# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Split phases preserve stock forward dynamics and native COM-wrench semantics."""

import importlib.util
import os
import unittest

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo as StockMuJoCo

from newton_solver_ancf import SolverMuJoCo

DEVICE = os.getenv("NEWTON_TEST_DEVICE", "cpu")


def model(device, count=2):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for i in range(count):
        body = builder.add_link(
            xform=wp.transform((float(i), 0.0, 0.5), wp.quat_identity()),
            mass=2.0,
            com=wp.vec3(0.08, -0.04, 0.03),
            inertia=wp.mat33(np.diag([0.2, 0.3, 0.4])),
        )
        builder.add_articulation([builder.add_joint_free(body)])
    return builder.finalize(device=device)


def initial(model):
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state


@wp.kernel
def add_forces(
    mapping: wp.array2d[int],
    forces: wp.array[wp.spatial_vector],
    applied: wp.array2d[wp.spatial_vector],
):
    world, body = wp.tid()
    index = mapping[world, body]
    if index >= 0:
        applied[world, body] += forces[index]


@unittest.skipUnless(
    importlib.util.find_spec("mujoco") and importlib.util.find_spec("mujoco_warp"),
    "Install the examples MuJoCo dependencies",
)
class TestMuJoCo(unittest.TestCase):
    def test_cpu_inherited_step_and_explicit_split_rejection(self):
        native = model("cpu", 1)
        solver = SolverMuJoCo(native, use_mujoco_cpu=True)
        start, out = initial(native), native.state()
        start.body_f.assign([[4, -2, 0, 0, 0, 0.6]])
        solver.step(start, out, native.control(), None, 0.001)
        self.assertGreater(np.linalg.norm(out.body_qd.numpy()), 0)
        solver.reset(start, flags=0, time=1.25)
        self.assertEqual(solver.mj_data.time, 1.25)
        for time in (-0.1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "time"):
                solver.reset(start, flags=0, time=time)
        mask = wp.array([True], dtype=wp.bool, device="cpu")
        with self.assertRaisesRegex(ValueError, "world_mask"):
            solver.reset(start, world_mask=mask, flags=0, time=0.0)
        with self.assertRaisesRegex(NotImplementedError, "CUDA backend"):
            solver.step_kinematics(start, out, None, None, 0.001)
        with self.assertRaisesRegex(NotImplementedError, "CUDA backend"):
            solver.step_dynamics(out)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "Split stepping requires CUDA")
    def test_split_matches_stock_with_multiple_inserted_wrenches(self):
        native = model(DEVICE)
        dt = 1 / 240
        base = np.array([[0.3, -0.2, 0.1, 0.1, 0.0, -0.1], [-0.4, 0.1, 0.2, 0.0, 0.1, 0.2]], np.float32)
        extra = np.array([[2.0, 1.0, -0.4, 0.2, -0.3, 0.1], [-1.0, 0.3, 1.2, -0.2, 0.1, 0.4]], np.float32)
        for integrator in ("euler", "implicitfast"):
            with self.subTest(integrator=integrator):
                stock = StockMuJoCo(native, integrator=integrator, update_data_interval=1)
                split = SolverMuJoCo(native, integrator=integrator, update_data_interval=1)
                left, right = initial(native), initial(native)
                velocities = np.array([0.2, -0.1, 0.3, -0.2, 0.3, 0.1] * 2, np.float32)
                for state in (left, right):
                    state.joint_qd.assign(velocities)
                    newton.eval_fk(native, state.joint_q, state.joint_qd, state)
                before = [getattr(right, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")]
                left.body_f.assign(base + 2 * extra)
                right.body_f.assign(base)
                expected, actual = native.state(), native.state()
                stock.step(left, expected, native.control(), None, dt)
                split.step_kinematics(right, actual, native.control(), None, dt)
                self.assertEqual(split._step, 0)
                np.testing.assert_array_equal(split.mjw_data.time.numpy(), 0.0)
                self.assertIs(split.xpos, split.mjw_data.xpos)
                self.assertIs(split.xquat, split.mjw_data.xquat)
                self.assertIs(split.cvel, split.mjw_data.cvel)
                force = wp.array(extra, dtype=wp.spatial_vector, device=DEVICE)
                for _ in range(2):
                    wp.launch(
                        add_forces,
                        split.xfrc_applied.shape,
                        inputs=[split.mjc_body_to_newton, force, split.xfrc_applied],
                    )
                split.step_dynamics(actual)
                self.assertEqual(split._step, 1)
                np.testing.assert_allclose(split.mjw_data.time.numpy(), dt, atol=1e-9)
                for index, name in enumerate(("joint_q", "joint_qd", "body_q", "body_qd")):
                    np.testing.assert_array_equal(getattr(right, name).numpy(), before[index])
                    np.testing.assert_allclose(
                        getattr(actual, name).numpy(), getattr(expected, name).numpy(), rtol=2e-6, atol=2e-7
                    )
                np.testing.assert_array_equal(right.body_f.numpy(), base)
                with self.assertRaisesRegex(RuntimeError, "step_kinematics"):
                    split.step_dynamics(actual)

    @unittest.skipUnless(DEVICE.startswith("cuda"), "Split stepping requires CUDA")
    def test_com_wrench_and_phase_validation(self):
        native = model(DEVICE, 1)
        solver = SolverMuJoCo(native, integrator="euler")
        start, out = initial(native), native.state()
        with self.assertRaisesRegex(RuntimeError, "step_kinematics"):
            solver.step_dynamics(out)
        for dt in (0.0, -0.1, float("nan")):
            with self.assertRaisesRegex(ValueError, "dt"):
                solver.step_kinematics(start, out, None, None, dt)
        with self.assertRaisesRegex(RuntimeError, "backward"):
            with wp.Tape():
                solver.step_kinematics(start, out, None, None, 0.001)
        dt = 0.001
        solver.step_kinematics(start, out, None, None, dt)
        force = wp.array([[4, -2, 0, 0, 0, 0.6]], dtype=wp.spatial_vector, device=DEVICE)
        wp.launch(add_forces, solver.xfrc_applied.shape, inputs=[solver.mjc_body_to_newton, force, solver.xfrc_applied])
        solver.step_dynamics(out)
        expected = dt * np.array([[2.0, -1.0, 0.0, 0.0, 0.0, 1.5]])
        np.testing.assert_allclose(out.body_qd.numpy(), expected, rtol=2e-5, atol=1e-8)
        rk4 = SolverMuJoCo(native, integrator="rk4")
        with self.assertRaisesRegex(NotImplementedError, "RK4"):
            rk4.step_kinematics(start, out, None, None, dt)
        external = SolverMuJoCo(native, use_mujoco_contacts=False)
        with self.assertRaisesRegex(ValueError, "contact storage"):
            external.step_kinematics(start, out, None, None, dt)
        solver.step_kinematics(start, out, None, None, dt)
        with self.assertRaisesRegex(RuntimeError, "step_dynamics or reset"):
            solver.step(start, out, None, None, dt)
        self.assertEqual(solver._step, 1)
        solver.reset(start, flags=0, time=1.25)
        np.testing.assert_array_equal(solver.mjw_data.time.numpy(), 1.25)
        with self.assertRaisesRegex(RuntimeError, "step_kinematics"):
            solver.step_dynamics(out)
        solver.step(start, out, None, None, dt)
        self.assertEqual(solver._step, 2)
        np.testing.assert_allclose(solver.mjw_data.time.numpy(), 1.25 + dt, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
