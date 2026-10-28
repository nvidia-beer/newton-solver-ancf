# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A free MuJoCo box clamped to the end of a nonlinear FEM cantilever."""

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from newton_solver_ancf import MuJoCoAttachment


class TipBox:
    """Advance the box and FEM together, retaining reciprocal force and torque."""

    def __init__(self, example, mass: float, mass_range: tuple[float, float]):
        self.example = example
        self.mass_range = mass_range
        self.mass = float(mass)
        old_solver = example.solver
        self.uncoupled_solver = old_solver
        fem = old_solver.fem
        reference = fem.rest_q.numpy().reshape(fem.node_count, fem.dofs_per_node)
        shell = fem.formulation == "shell3423"
        self.nodes = getattr(
            example,
            "tip_nodes",
            np.arange(fem.node_count - (2 if shell else 1), fem.node_count, dtype=np.int32),
        )
        self.half_extents = np.array([0.025, 0.044 if shell else 0.025, 0.025])
        center = reference[self.nodes, :3].mean(axis=0) + np.array([self.half_extents[0], 0.0, 0.0])
        self.positions = reference[self.nodes, :3] - center
        self.slopes = reference[self.nodes, 3:].reshape(len(self.nodes), -1, 3)
        tip_dofs = (fem.dofs_per_node * self.nodes[:, None] + np.arange(fem.dofs_per_node)).ravel()
        example.solver = type(old_solver)(
            old_solver.model,
            fem,
            fixed_dofs=np.union1d(old_solver.fixed_dofs, tip_dofs),
            gravity=tuple(old_solver.gravity),
            mass_damping=old_solver.mass_damping,
            max_iterations=old_solver.max_iterations,
            tolerance=old_solver.tolerance,
        )

        self.coupled_solver = example.solver
        builder = newton.ModelBuilder()
        self.body = builder.add_link(
            xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
            mass=self.mass,
            inertia=wp.mat33(self._inertia(self.mass)),
            label="tip_box",
        )
        builder.add_articulation([builder.add_joint_free(self.body)])
        builder.add_shape_box(
            self.body,
            hx=float(self.half_extents[0]),
            hy=float(self.half_extents[1]),
            hz=float(self.half_extents[2]),
            cfg=builder.ShapeConfig(density=0.0),
            color=(1.0, 0.45, 0.05),
        )
        self.model = builder.finalize(device=fem.device)
        self.rigid_solver = SolverMuJoCo(
            self.model,
            use_mujoco_cpu=not fem.device.is_cuda,
            use_mujoco_contacts=True,
            update_data_interval=1,
            integrator="implicitfast",
            solver="newton",
            iterations=20,
            ls_iterations=30,
            njmax=16,
            nconmax=8,
        )
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.coupler = self._make_coupler()
        self.last_residual = 0.0
        self.last_iterations = 0

    def _inertia(self, mass):
        x, y, z = self.half_extents
        return np.diag(mass / 3.0 * np.array([y * y + z * z, x * x + z * z, x * x + y * y]))

    def _make_coupler(self):
        return MuJoCoAttachment(self.coupled_solver, self.rigid_solver, self.nodes, self.positions, self.slopes)

    def set_mass(self, mass: float):
        """Change uniform box density while keeping its pose and velocity."""
        if not np.isfinite(mass) or not self.mass_range[0] <= mass <= self.mass_range[1]:
            raise ValueError(f"mass must lie in {self.mass_range} kg")
        if mass == self.mass:
            return
        inertia = self._inertia(mass)
        self.model.body_mass.assign(np.array([mass], dtype=np.float32))
        self.model.body_inv_mass.assign(np.array([1.0 / mass], dtype=np.float32))
        self.model.body_inertia.assign(inertia[None].astype(np.float32))
        self.model.body_inv_inertia.assign(np.linalg.inv(inertia)[None].astype(np.float32))
        self.rigid_solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        # Refresh the cached rigid trial data after an interactive parameter edit.
        self.coupler = self._make_coupler()
        self.mass = float(mass)

    def reset(self, *, coupled: bool):
        """Reset both physical states and select the original or clamped solver."""
        self.example.solver = self.coupled_solver if coupled else self.uncoupled_solver
        self.example.state = self.example.solver.initial_state()
        self.example.sim_time = 0.0
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.coupler = self._make_coupler()
        self.last_residual, self.last_iterations = 0.0, 0

    def step(self, dt):
        self.example.state, self.state = self.coupler.step(self.example.state, self.state, dt)
        self.last_residual = self.coupler.last_residual
        self.last_iterations = self.coupler.last_iterations

    def test_final(self):
        assert np.isfinite(self.state.body_q.numpy()).all()
        assert self.last_residual <= self.coupler.tolerance
