# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A rectangular ANCF3423 shell strip bends under gravity with one edge clamped.

The 0.3 m by 0.08 m strip is 0.01 m thick and uses four quadrilateral shell
elements by default. Position and thickness-director coordinates are fixed
at both nodes of the root edge. The other nodes move under gravity with
mass-proportional damping. --debug_arrows displays the nodal directors.

"""

from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_ancf import SolverANCFShell, load_element_mesh
from newton_solver_ancf.examples._element_colors import ElementSurface
from newton_solver_ancf.examples._fixed_markers import FixedNodeMarkers
from newton_solver_ancf.examples._node_arrows import NodeGradientArrows


@wp.kernel
def _render_positions(
    q: wp.array[wp.float64],
    face_nodes: wp.array[int],
    corners_per_face: int,
    render_offset: wp.vec3,
    positions: wp.array[wp.vec3],
    face_positions: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    index = wp.tid()
    if index < len(positions):
        positions[index] = (
            wp.vec3(wp.float32(q[6 * index]), wp.float32(q[6 * index + 1]), wp.float32(q[6 * index + 2]))
            + render_offset
        )
    node = face_nodes[index]
    corner = index % corners_per_face
    face = index - corner
    next_node = face_nodes[face + (corner + 1) % corners_per_face]
    previous_node = face_nodes[face + (corner + corners_per_face - 1) % corners_per_face]
    position = wp.vec3(float(q[6 * node]), float(q[6 * node + 1]), float(q[6 * node + 2]))
    next_position = wp.vec3(float(q[6 * next_node]), float(q[6 * next_node + 1]), float(q[6 * next_node + 2]))
    previous_position = wp.vec3(
        float(q[6 * previous_node]), float(q[6 * previous_node + 1]), float(q[6 * previous_node + 2])
    )
    face_positions[index] = position + render_offset
    normals[index] = wp.normalize(wp.cross(next_position - position, previous_position - position))


@wp.kernel
def _render_edges(
    positions: wp.array[wp.vec3],
    edges: wp.array[wp.vec2i],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    edge = wp.tid()
    starts[edge] = positions[edges[edge][0]]
    ends[edge] = positions[edges[edge][1]]


class Shell:
    """A clamped rectangular strip with gravity and mass-proportional damping."""

    def __init__(self, viewer, *, substeps=2, young_modulus=2.0e7, debug_arrows=False, render_offset=(0.0, 0.0, 0.0)):
        self.viewer = viewer
        self._render_offset = wp.vec3(render_offset)
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = substeps
        self.sim_time = 0.0
        if self.sim_substeps < 1:
            raise ValueError("substeps must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        positions, elements = load_element_mesh(
            Path(__file__).with_name("assets") / "element_meshes.usdc", "/Elements/Shell/e4"
        )
        self.fem_model = SolverANCFShell.create_model(
            positions,
            elements,
            thickness=0.01,
            young_modulus=young_modulus,
            poisson_ratio=0.3,
            density=700.0,
            device=wp.get_device(),
        )
        builder = newton.ModelBuilder()
        self.fem_model.register(builder)
        native_model = builder.finalize(device=self.fem_model.device)
        self.solver = SolverANCFShell(
            native_model,
            self.fem_model,
            fixed_dofs=np.arange(12),
            mass_damping=6.0,
            max_iterations=35,
            tolerance=1.0e-8,
        )
        self.state = self.solver.initial_state()
        self.node_arrows = NodeGradientArrows(
            self.fem_model, axes=(2,), enabled=debug_arrows, render_offset=render_offset
        )
        self._rest = self.fem_model.rest_q.numpy().copy()
        self._fixed_points = wp.array(
            positions[:2].astype(np.float32) + np.asarray(render_offset, dtype=np.float32),
            dtype=wp.vec3,
            device=self.fem_model.device,
        )
        self._fixed_colors = wp.full(2, wp.vec3(1.0, 0.45, 0.05), dtype=wp.vec3, device=self.fem_model.device)
        self._fixed_markers = FixedNodeMarkers(self._fixed_points, self._fixed_colors, 0.008)
        self._vertices = wp.empty(len(positions), dtype=wp.vec3, device=self.fem_model.device)
        self._node_colors = wp.full(len(positions), wp.vec3(0.8, 0.8, 0.8), dtype=wp.vec3, device=self.fem_model.device)
        self._corners_per_face = elements.shape[1]
        self._surface_nodes = wp.array(elements.reshape(-1), dtype=int, device=self.fem_model.device)
        self._surface_vertices = wp.empty(elements.size, dtype=wp.vec3, device=self.fem_model.device)
        self._normals = wp.empty_like(self._surface_vertices)
        face_triangles = [0, 1, 2] if self._corners_per_face == 3 else [0, 1, 2, 0, 2, 3]
        self._triangles = wp.array(
            (self._corners_per_face * np.arange(len(elements))[:, None] + face_triangles).ravel(),
            dtype=int,
            device=self.fem_model.device,
        )
        self._surface = ElementSurface(
            np.repeat(np.arange(len(elements)), self._corners_per_face), len(elements), self.fem_model.device
        )
        edges = set()
        for element in elements:
            edges.update(tuple(sorted(edge)) for edge in zip(element, np.roll(element, -1), strict=True))
        self._edges = wp.array(sorted(edges), dtype=wp.vec2i, device=self.fem_model.device)
        self._edge_starts = wp.empty(len(edges), dtype=wp.vec3, device=self.fem_model.device)
        self._edge_ends = wp.empty_like(self._edge_starts)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state = self.solver.advance(self.state, self.sim_dt)
        self.sim_time += self.frame_dt

    def test_final(self, *, check_gravity=True):
        coordinates = self.state.q.numpy()
        if not np.isfinite(coordinates).all() or not np.isfinite(self.state.qd.numpy()).all():
            raise AssertionError("shell state contains non-finite values")
        np.testing.assert_allclose(coordinates[:12], self._rest[:12], atol=1.0e-12, rtol=0.0)
        if np.max(np.linalg.norm(coordinates.reshape(-1, 6)[:, :3] - self._rest.reshape(-1, 6)[:, :3], axis=1)) > 0.4:
            raise AssertionError("shell displacement exceeded its cantilever length")
        if check_gravity and self.sim_time > 0.05 and np.mean(coordinates.reshape(-1, 6)[-2:, 2]) >= 0.3:
            raise AssertionError("gravity did not bend the shell cantilever downward")

    def render(self, *, show_edges=False):
        wp.launch(
            _render_positions,
            dim=len(self._surface_nodes),
            inputs=[self.state.q, self._surface_nodes, self._corners_per_face, self._render_offset],
            outputs=[self._vertices, self._surface_vertices, self._normals],
            device=self.fem_model.device,
            record_tape=False,
        )
        wp.launch(
            _render_edges,
            dim=len(self._edges),
            inputs=[self._vertices, self._edges],
            outputs=[self._edge_starts, self._edge_ends],
            device=self.fem_model.device,
            record_tape=False,
        )
        self._surface.render(
            self.viewer,
            "shell_elements",
            self._surface_vertices,
            self._triangles,
            self._normals,
            backface_culling=False,
        )
        self.viewer.log_lines(
            "element_edges",
            self._edge_starts,
            self._edge_ends,
            colors=(0.8, 0.8, 0.8),
            hidden=not show_edges or not self.viewer.layer.visible,
        )
        self.viewer.log_points(
            "nodes", self._vertices, radii=0.0035, colors=self._node_colors, hidden=not self.viewer.layer.visible
        )
        self._fixed_markers.render(self.viewer)
        self.node_arrows.render(self.viewer, self.state.q)
