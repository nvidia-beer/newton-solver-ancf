# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A solid cantilever bends under gravity using cubic ANCF tetrahedra, with one end face clamped.

The 0.3 m by 0.05 m by 0.04 m solid is divided into grid cells, each split
into six four-node cubic ANCF/FN tetrahedra. Every node carries its position
and three position-gradient vectors; all coordinates of the four root-face
nodes are fixed. Curved exterior faces use the same cubic interpolation as
the mechanics. --debug_arrows displays the nodal gradients.

"""

from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_ancf import SolverANCFTetrahedron, boundary_faces, load_element_mesh
from newton_solver_ancf.examples._element_colors import ElementSurface
from newton_solver_ancf.examples._fixed_markers import FixedNodeMarkers
from newton_solver_ancf.examples._node_arrows import NodeGradientArrows


@wp.kernel
def _render_surface(
    q: wp.array[wp.float64],
    vector_indices: wp.array2d[int],
    shape: wp.array2d[wp.float64],
    tangent_u: wp.array2d[wp.float64],
    tangent_v: wp.array2d[wp.float64],
    render_offset: wp.vec3,
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    vertex = wp.tid()
    position, du, dv = wp.vec3d(0.0), wp.vec3d(0.0), wp.vec3d(0.0)
    for coefficient in range(16):
        index = 3 * vector_indices[vertex, coefficient]
        vector = wp.vec3d(q[index], q[index + 1], q[index + 2])
        position += shape[vertex, coefficient] * vector
        du += tangent_u[vertex, coefficient] * vector
        dv += tangent_v[vertex, coefficient] * vector
    vertices[vertex] = wp.vec3(position) + render_offset
    normals[vertex] = wp.vec3(wp.normalize(wp.cross(du, dv)))


@wp.kernel
def _render_annotations(
    q: wp.array[wp.float64],
    vertices: wp.array[wp.vec3],
    edge_indices: wp.array[wp.vec2i],
    render_offset: wp.vec3,
    nodes: wp.array[wp.vec3],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    index = wp.tid()
    if index < nodes.shape[0]:
        nodes[index] = wp.vec3(float(q[12 * index]), float(q[12 * index + 1]), float(q[12 * index + 2])) + render_offset
    if index < edge_indices.shape[0]:
        edge = edge_indices[index]
        starts[index] = vertices[edge[0]]
        ends[index] = vertices[edge[1]]


class Tetrahedron:
    """A tetrahedral beam clamped at four nodes, with cubic exterior surfaces."""

    def __init__(self, viewer, *, substeps=4, young_modulus=2.0e6, debug_arrows=False, render_offset=(0.0, 0.0, 0.0)):
        if substeps < 1:
            raise ValueError("substeps must be positive")
        self.viewer = viewer
        self.sim_substeps = substeps
        self._render_offset = wp.vec3(render_offset)
        self.sim_time, self.frame_dt = 0.0, 1.0 / 60.0
        self.sim_dt = self.frame_dt / substeps
        positions, elements = load_element_mesh(
            Path(__file__).with_name("assets") / "element_meshes.usdc", "/Elements/Tetrahedron/e4"
        )
        self.fem_model = SolverANCFTetrahedron.create_model(
            positions,
            elements,
            young_modulus=young_modulus,
            poisson_ratio=0.3,
            density=700.0,
            device=wp.get_device(),
        )
        builder = newton.ModelBuilder()
        self.fem_model.register(builder)
        native_model = builder.finalize(device=self.fem_model.device)
        self.solver = SolverANCFTetrahedron(
            native_model,
            self.fem_model,
            fixed_dofs=np.arange(48),
            mass_damping=3.0,
            max_iterations=35,
            tolerance=1.0e-8,
        )
        self.state = self.solver.initial_state()
        self.tip_nodes = np.arange(len(positions) - 4, len(positions), dtype=np.int32)
        self._reference = self.fem_model.rest_q.numpy()
        self.node_arrows = NodeGradientArrows(
            self.fem_model, axes=(0, 1, 2), enabled=debug_arrows, render_offset=render_offset
        )
        self._make_surface(positions, elements)
        device = self.fem_model.device
        self._nodes = wp.empty(len(positions), dtype=wp.vec3, device=device)
        self._node_colors = wp.full(len(positions), wp.vec3(0.8, 0.8, 0.8), dtype=wp.vec3, device=device)
        self._fixed_points = wp.array(
            positions[:4].astype(np.float32) + np.asarray(render_offset, dtype=np.float32),
            dtype=wp.vec3,
            device=device,
        )
        self._fixed_colors = wp.full(4, wp.vec3(1.0, 0.45, 0.05), dtype=wp.vec3, device=device)
        self._fixed_markers = FixedNodeMarkers(self._fixed_points, self._fixed_colors, 0.008)

    def _make_surface(self, positions, elements):
        subdivisions = 6
        boundary, owners = boundary_faces(elements)
        faces = [
            (int(owner), tuple(int(np.flatnonzero(elements[owner] == node)[0]) for node in face))
            for face, owner in zip(boundary, owners, strict=True)
        ]
        shapes, vectors, tangent_u, tangent_v, edges, triangles, vertex_elements = [], [], [], [], [], [], []
        for element, nodes in enumerate(elements):
            reference = positions[nodes]
            vector_indices = (4 * nodes[:, None] + np.arange(4)).ravel()
            for owner, face in faces:
                if owner != element:
                    continue
                indices = {}
                a, b, c = face
                for u in range(subdivisions + 1):
                    for v in range(subdivisions + 1 - u):
                        barycentric = np.zeros(4)
                        barycentric[a] = 1.0 - (u + v) / subdivisions
                        barycentric[b], barycentric[c] = u / subdivisions, v / subdivisions
                        shape, derivative = SolverANCFTetrahedron.shape_functions(barycentric, reference)
                        indices[u, v] = len(shapes)
                        shapes.append(shape)
                        vertex_elements.append(element)
                        vectors.append(vector_indices)
                        tangent_u.append(derivative @ (reference[b] - reference[a]))
                        tangent_v.append(derivative @ (reference[c] - reference[a]))
                for u in range(subdivisions):
                    for v in range(subdivisions - u):
                        triangles.extend((indices[u, v], indices[u + 1, v], indices[u, v + 1]))
                        if u + v < subdivisions - 1:
                            triangles.extend((indices[u + 1, v], indices[u + 1, v + 1], indices[u, v + 1]))
                for i in range(subdivisions):
                    edges.extend(
                        (
                            (indices[i, 0], indices[i + 1, 0]),
                            (indices[0, i], indices[0, i + 1]),
                            (indices[i, subdivisions - i], indices[i + 1, subdivisions - i - 1]),
                        )
                    )
        device = self.fem_model.device
        self._shape = wp.array(np.array(shapes), dtype=wp.float64, device=device)
        self._vector_indices = wp.array(np.array(vectors), dtype=int, device=device)
        self._tangent_u = wp.array(np.array(tangent_u), dtype=wp.float64, device=device)
        self._tangent_v = wp.array(np.array(tangent_v), dtype=wp.float64, device=device)
        self._vertices = wp.empty(len(shapes), dtype=wp.vec3, device=device)
        self._normals = wp.empty_like(self._vertices)
        self._triangles = wp.array(triangles, dtype=int, device=device)
        self._surface = ElementSurface(vertex_elements, len(elements), device)
        self._edges = wp.array(edges, dtype=wp.vec2i, device=device)
        self._edge_starts = wp.empty(len(edges), dtype=wp.vec3, device=device)
        self._edge_ends = wp.empty_like(self._edge_starts)
        self._edge_colors = wp.full(len(edges), wp.vec3(0.8, 0.8, 0.8), dtype=wp.vec3, device=device)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state = self.solver.advance(self.state, self.sim_dt)
        self.sim_time += self.frame_dt

    def render(self, *, show_edges=False):
        wp.launch(
            _render_surface,
            dim=len(self._vertices),
            inputs=[
                self.state.q,
                self._vector_indices,
                self._shape,
                self._tangent_u,
                self._tangent_v,
                self._render_offset,
            ],
            outputs=[self._vertices, self._normals],
            device=self.fem_model.device,
            record_tape=False,
        )
        wp.launch(
            _render_annotations,
            dim=max(len(self._nodes), len(self._edges)),
            inputs=[self.state.q, self._vertices, self._edges, self._render_offset],
            outputs=[self._nodes, self._edge_starts, self._edge_ends],
            device=self.fem_model.device,
            record_tape=False,
        )
        self._surface.render(
            self.viewer,
            "tetrahedral_beam",
            self._vertices,
            self._triangles,
            self._normals,
        )
        self.viewer.log_lines(
            "element_edges",
            self._edge_starts,
            self._edge_ends,
            self._edge_colors,
            hidden=not show_edges or not self.viewer.layer.visible,
        )
        self.viewer.log_points(
            "nodes", self._nodes, radii=0.003, colors=self._node_colors, hidden=not self.viewer.layer.visible
        )
        self._fixed_markers.render(self.viewer)
        self.node_arrows.render(self.viewer, self.state.q)

    def test_final(self, *, check_gravity=True):
        q, qd = self.state.q.numpy(), self.state.qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise AssertionError("tetrahedron state contains non-finite values")
        np.testing.assert_allclose(q[:48], self._reference[:48], atol=1.0e-12, rtol=0.0)
        positions = q.reshape(-1, 12)[:, :3]
        reference = self._reference.reshape(-1, 12)[:, :3]
        if np.max(np.linalg.norm(positions - reference, axis=1)) > 0.4:
            raise AssertionError("tetrahedron displacement exceeded its cantilever length")
        local = q[self.fem_model.element_dofs.numpy()].reshape(-1, 16, 3)
        # Independently sample corners and the interior through the public basis API.
        # The solver also rejects inverted states at its integration points.
        samples = np.vstack((np.eye(4), np.full((1, 4), 0.25)))
        corners = reference[self.fem_model.elements.numpy()]
        _, derivatives = SolverANCFTetrahedron.shape_functions(samples[None], corners[:, None])
        deformation = np.einsum("eai,eqaj->eqij", local, derivatives)
        jacobian = np.linalg.det(deformation)
        if not np.isfinite(jacobian).all() or np.any(jacobian <= 0.0):
            raise AssertionError("tetrahedron contains an inverted or non-finite deformation gradient")
        if (
            check_gravity
            and self.sim_time > 0.05
            and positions[self.tip_nodes, 2].mean() >= reference[self.tip_nodes, 2].mean()
        ):
            raise AssertionError("gravity did not deflect the tetrahedron downward")
