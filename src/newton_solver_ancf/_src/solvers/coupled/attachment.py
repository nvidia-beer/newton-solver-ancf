# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Small, forward-only attachment adapter for a single free MuJoCo body."""

import newton
import numpy as np
import warp as wp
from warp._src import context as wp_context


@wp.kernel(module="unique", enable_backward=True)
def _attachment_boundary(
    reference: wp.array[wp.float64],
    body_q: wp.array[wp.transform],
    node_attachment: wp.array[int],
    body: int,
    offsets: wp.array[wp.vec3],
    slopes: wp.array[wp.vec3],
    dofs_per_node: int,
    out: wp.array[wp.float64],
):
    i = wp.tid()
    attachment = node_attachment[i // dofs_per_node]
    value = reference[i]
    if attachment >= 0:
        pose = body_q[body]
        point = wp.transform_point(pose, offsets[attachment])
        axis = i % dofs_per_node
        if axis < 3:
            value = wp.float64(point[axis])
        else:
            gradient = attachment * (dofs_per_node // 3 - 1) + axis // 3 - 1
            direction = wp.transform_vector(pose, slopes[gradient])
            value = wp.float64(direction[axis % 3])
    out[i] = value


@wp.kernel(module="unique", enable_backward=True)
def _attachment_wrenches(
    reaction: wp.array[wp.float64],
    boundary: wp.array[wp.float64],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    nodes: wp.array[int],
    body: int,
    dofs_per_node: int,
    out: wp.array[wp.spatial_vector],
):
    attachment = wp.tid()
    j = dofs_per_node * nodes[attachment]
    force = -wp.vec3(float(reaction[j]), float(reaction[j + 1]), float(reaction[j + 2]))
    position = wp.vec3(float(boundary[j]), float(boundary[j + 1]), float(boundary[j + 2]))
    center = wp.transform_point(body_q[body], body_com[body])
    torque = wp.cross(position - center, force)
    for gradient in range(1, dofs_per_node // 3):
        k = j + 3 * gradient
        director_force = wp.vec3(float(reaction[k]), float(reaction[k + 1]), float(reaction[k + 2]))
        director = wp.vec3(float(boundary[k]), float(boundary[k + 1]), float(boundary[k + 2]))
        torque -= wp.cross(director, director_force)
    out[attachment] = wp.spatial_vector(force, torque)


@wp.kernel(module="unique", enable_backward=True)
def _sum_attachment_wrenches(
    attachments: wp.array[wp.spatial_vector],
    external: wp.array[wp.spatial_vector],
    attachment_body: int,
    out: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    value = external[body]
    if body == attachment_body:
        for attachment in range(attachments.shape[0]):
            value += attachments[attachment]
    out[body] = value


def _coordinates(pose, length):
    q = np.asarray(pose[3:], dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    norm = np.linalg.norm(q[:3])
    rotation = 2 * q[:3] if norm < 1.0e-12 else 2 * np.arctan2(norm, q[3]) * q[:3] / norm
    return np.r_[pose[:3] / length, rotation]


def _pose(coordinates, length):
    angle = np.linalg.norm(coordinates[3:])
    scale = 0.5 if angle < 1.0e-12 else np.sin(0.5 * angle) / angle
    return np.r_[length * coordinates[:3], scale * coordinates[3:], np.cos(0.5 * angle)].astype(np.float32)


class MuJoCoAttachment:
    """Converge a single free body's position/director attachment to a FEM tip.

    Each trial uses the SAME time-n states, a converged FEM solve and its full
    reciprocal force and torque. MuJoCo alone advances the body. A six-coordinate
    Broyden iteration accelerates this interface fixed point; the returned state
    must satisfy the measured attachment tolerance. This adapter has no coupled
    backward implementation and explicitly rejects a Warp tape.
    """

    def __init__(self, solver, rigid_solver, nodes, positions, slopes, *, iterations=50, tolerance=1.0e-5):
        self.solver, self.rigid_solver = solver, rigid_solver
        self.model = rigid_solver.model
        fem = solver.fem
        if self.model.body_count != 1:
            raise ValueError("The minimal attachment adapter requires one free MuJoCo body")
        if solver.device != self.model.device:
            raise ValueError("FEM and MuJoCo devices must match")
        if rigid_solver.update_data_interval != 1:
            raise ValueError("MuJoCo attachment requires update_data_interval=1")
        if self.model.joint_count != 1 or self.model.joint_type.numpy()[0] != int(newton.JointType.FREE):
            raise ValueError("MuJoCo attachment requires one free joint")
        if int(self.model.body_flags.numpy()[0]) & int(newton.BodyFlags.KINEMATIC):
            raise ValueError("The attached body must be dynamic")
        self.nodes = np.asarray(nodes, dtype=np.int32)
        tip = (fem.dofs_per_node * self.nodes[:, None] + np.arange(fem.dofs_per_node)).ravel()
        if not np.isin(tip, solver.fixed_dofs).all():
            raise ValueError("All attached FEM coordinates must be prescribed")
        node_attachment = np.full(fem.node_count, -1, dtype=np.int32)
        node_attachment[self.nodes] = np.arange(len(self.nodes))
        self._nodes = wp.array(self.nodes, dtype=int, device=fem.device)
        self._node_attachment = wp.array(node_attachment, dtype=int, device=fem.device)
        self._positions = wp.array(positions, dtype=wp.vec3, device=fem.device)
        self._slopes = wp.array(np.asarray(slopes).reshape(-1, 3), dtype=wp.vec3, device=fem.device)
        self._slope_norms = np.linalg.norm(np.asarray(slopes), axis=2)
        self.length = float(np.mean(fem.lengths.numpy()))
        self.iterations, self.tolerance = iterations, tolerance
        self.last_residual = 0.0
        self.last_iterations = 0
        self._jacobian = None
        self._dt = None
        self.control = self.model.control()
        self.last_wrench = wp.zeros(1, dtype=wp.spatial_vector, device=fem.device)

    def _boundary(self, pose):
        fem = self.solver.fem
        boundary = wp.empty_like(fem.rest_q)
        wp.launch(
            _attachment_boundary,
            fem.dof_count,
            inputs=[
                fem.rest_q,
                pose,
                self._node_attachment,
                0,
                self._positions,
                self._slopes,
                fem.dofs_per_node,
                boundary,
            ],
            device=fem.device,
        )
        return boundary

    def step(self, fem_state, rigid_state, dt):
        if wp_context.runtime.tape is not None:
            raise RuntimeError("MuJoCo attachment backward is not implemented; differentiate standalone FEM steps")
        fem = self.solver.fem
        device = fem.device
        start = self.model.state()
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            wp.copy(getattr(start, name), getattr(rigid_state, name))
        external = wp.clone(rigid_state.body_f)
        pose = wp.clone(rigid_state.body_q)
        current = _coordinates(pose.numpy()[0], self.length)
        jacobian = -np.eye(6) if self._jacobian is None or self._dt != dt else self._jacobian.copy()
        previous_coordinates = previous_residual = None
        guess = fem_state.q
        for iteration in range(self.iterations):
            pose.assign(_pose(current, self.length)[None])
            boundary = self._boundary(pose)
            structural = self.solver.advance(fem_state, dt, boundary_q=boundary, initial_guess=guess)
            guess = structural.q
            reaction = self.solver.reaction(fem_state, structural, dt)
            wrenches = wp.empty(len(self.nodes), dtype=wp.spatial_vector, device=device)
            wp.launch(
                _attachment_wrenches,
                len(self.nodes),
                inputs=[reaction, boundary, pose, self.model.body_com, self._nodes, 0, fem.dofs_per_node, wrenches],
                device=device,
            )
            wp.launch(_sum_attachment_wrenches, 1, inputs=[wrenches, external, 0, start.body_f], device=device)
            result = self.model.state()
            # Public reset clears native warm-start/history while flags=0 preserves
            # the retained input coordinates. update_data_interval=1 reimports them.
            self.rigid_solver.reset(start, flags=0)
            self.rigid_solver.step(start, result, self.control, None, dt)
            target = _coordinates(result.body_q.numpy()[0], self.length)
            residual = target - current
            actual = self._boundary(result.body_q).numpy().reshape(fem.node_count, fem.dofs_per_node)[self.nodes]
            wanted = boundary.numpy().reshape(fem.node_count, fem.dofs_per_node)[self.nodes]
            error = float(np.max(np.linalg.norm(actual[:, :3] - wanted[:, :3], axis=1)) / self.length)
            if fem.dofs_per_node > 3:
                errors = np.linalg.norm((actual[:, 3:] - wanted[:, 3:]).reshape(len(self.nodes), -1, 3), axis=2)
                error = max(error, float(np.max(errors / self._slope_norms)))
            self.last_residual, self.last_iterations = error, iteration + 1
            if not np.isfinite(residual).all():
                raise RuntimeError("MuJoCo attachment produced a non-finite state")
            if previous_coordinates is not None:
                delta = current - previous_coordinates
                change = residual - previous_residual
                norm = float(delta @ delta)
                if norm > 1.0e-20:
                    jacobian += np.outer(change - jacobian @ delta, delta) / norm
            if error <= self.tolerance:
                self._jacobian, self._dt = jacobian, dt
                wp.copy(self.last_wrench, start.body_f)
                # A returned state's body_f remains external, never reapply last
                # step's attachment reaction on the next step.
                wp.copy(result.body_f, external)
                return structural, result
            previous_coordinates, previous_residual = current.copy(), residual.copy()
            try:
                correction = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError:
                jacobian = -np.eye(6)
                correction = 0.2 * residual
            norm = np.linalg.norm(correction)
            current += correction * min(1.0, 0.1 / max(norm, 1.0e-30))
        self._jacobian = None
        raise RuntimeError(f"MuJoCo attachment did not converge: mismatch {self.last_residual:.3e}")
