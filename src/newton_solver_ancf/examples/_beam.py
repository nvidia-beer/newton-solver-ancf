# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A rectangular ANCF beam bends under gravity with one end clamped; choose 3243 or 3333 elements.

The default 0.4 m beam has six shear-deformable elements. The 3243 formulation
uses two nodes per element; 3333 uses three. All coordinates at the root node
are fixed, and mass-proportional damping reduces the motion. Gray shading
and outlines identify elements; --debug_arrows displays nodal gradients.

Used by the combined ``elements`` example; choose its beam with --formulation.
"""

from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_ancf import SolverANCFBeam, load_element_mesh
from newton_solver_ancf.examples._element_colors import ElementSurface
from newton_solver_ancf.examples._fixed_markers import FixedNodeMarkers
from newton_solver_ancf.examples._node_arrows import NodeGradientArrows


@wp.kernel
def _render_surface(
    q: wp.array[wp.float64],
    element_dofs: wp.array2d[int],
    vertex_elements: wp.array[int],
    shapes: wp.array2d[wp.float64],
    tangent_u_shapes: wp.array2d[wp.float64],
    tangent_v_shapes: wp.array2d[wp.float64],
    render_offset: wp.vec3,
    vertices: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    vertex = wp.tid()
    element = vertex_elements[vertex]
    position = wp.vec3d(0.0)
    tangent_u = wp.vec3d(0.0)
    tangent_v = wp.vec3d(0.0)
    for shape in range(shapes.shape[1]):
        coordinate = wp.vec3d(
            q[element_dofs[element, 3 * shape]],
            q[element_dofs[element, 3 * shape + 1]],
            q[element_dofs[element, 3 * shape + 2]],
        )
        position += shapes[vertex, shape] * coordinate
        tangent_u += tangent_u_shapes[vertex, shape] * coordinate
        tangent_v += tangent_v_shapes[vertex, shape] * coordinate
    normal = wp.normalize(wp.cross(tangent_u, tangent_v))
    vertices[vertex] = wp.vec3(position) + render_offset
    normals[vertex] = wp.vec3(wp.float32(normal[0]), wp.float32(normal[1]), wp.float32(normal[2]))


@wp.kernel
def _render_boundaries(
    vertices: wp.array[wp.vec3],
    boundary_vertices: wp.array[wp.vec4i],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    boundary = wp.tid()
    indices = boundary_vertices[boundary]
    center = (vertices[indices[0]] + vertices[indices[1]] + vertices[indices[2]] + vertices[indices[3]]) * 0.25
    for side in range(4):
        # Lift outlines off the surface to avoid depth flicker.
        starts[4 * boundary + side] = center + 1.01 * (vertices[indices[side]] - center)
        ends[4 * boundary + side] = center + 1.01 * (vertices[indices[(side + 1) % 4]] - center)


class Beam:
    def __init__(
        self,
        viewer,
        *,
        substeps=4,
        formulation="3243",
        young_modulus=2.0e7,
        debug_arrows=False,
        render_offset=(0.0, 0.0, 0.0),
    ):
        elements = 6
        if substeps < 1:
            raise ValueError("substeps must be positive")
        self.viewer = viewer
        self.formulation = formulation
        self.sim_substeps = substeps
        self._render_offset = wp.vec3(render_offset)
        self.sim_time, self.frame_dt = 0.0, 1.0 / 60.0
        self.sim_dt = self.frame_dt / substeps
        self.length, self.width, self.height = 0.4, 0.025, 0.025
        positions, _ = load_element_mesh(
            Path(__file__).with_name("assets") / "element_meshes.usdc", f"/Elements/Beam{formulation}/e6"
        )
        self.fem_model = SolverANCFBeam.create_model(
            positions,
            formulation=formulation,
            young_modulus=young_modulus,
            width=self.width,
            height=self.height,
            density=1000.0,
        )
        builder = newton.ModelBuilder()
        self.fem_model.register(builder)
        native_model = builder.finalize(device=self.fem_model.device)
        self.solver = SolverANCFBeam(
            native_model,
            self.fem_model,
            fixed_dofs=np.arange(self.fem_model.dofs_per_node),
            mass_damping=3.0,
        )
        self.state = self.solver.initial_state()
        self.node_arrows = NodeGradientArrows(
            self.fem_model,
            axes=(0, 1, 2) if formulation == "3243" else (1, 2),
            enabled=debug_arrows,
            render_offset=render_offset,
        )
        self._reference = self.state.q.numpy().copy()
        self._fixed_points = wp.array(
            positions[:1].astype(np.float32) + np.asarray(render_offset, dtype=np.float32),
            dtype=wp.vec3,
            device=self.fem_model.device,
        )
        self._fixed_colors = wp.full(1, wp.vec3(1.0, 0.45, 0.05), dtype=wp.vec3, device=self.fem_model.device)
        self._fixed_markers = FixedNodeMarkers(self._fixed_points, self._fixed_colors, 0.018)
        self._samples = 4
        self._build_render_mesh()
        boundary_edges = 4 * (elements + 1)
        self._boundary_starts = wp.empty(boundary_edges, dtype=wp.vec3, device=self.fem_model.device)
        self._boundary_ends = wp.empty_like(self._boundary_starts)
        self._boundary_colors = wp.full(
            boundary_edges, wp.vec3(0.8, 0.8, 0.8), dtype=wp.vec3, device=self.fem_model.device
        )

    def _build_render_mesh(self):
        shapes, tangent_u_shapes, tangent_v_shapes, vertex_elements = [], [], [], []
        triangles, boundaries = [], []
        corners = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
        side_tangents = ((1, 0), (2, 0), (0, 1), (0, 2))
        lengths = self.fem_model.lengths.numpy()

        def add_vertex(element, xi, corner, tangent_axes):
            eta, zeta = corners[corner]
            shape, derivative = SolverANCFBeam.shape_functions(
                self.formulation, xi, eta, zeta, lengths[element], self.width, self.height
            )
            shapes.append(shape)
            tangent_u_shapes.append(derivative[:, tangent_axes[0]])
            tangent_v_shapes.append(derivative[:, tangent_axes[1]])
            vertex_elements.append(element)

        for element in range(self.fem_model.element_count):
            first = len(shapes)
            faces = []
            # Each face owns its edge vertices: rectangular corners must not share shading normals.
            for side, tangent_axes in enumerate(side_tangents):
                start = len(shapes) - first
                for sample in range(self._samples + 1):
                    xi = -1.0 + 2.0 * sample / self._samples
                    add_vertex(element, xi, side, tangent_axes)
                    add_vertex(element, xi, (side + 1) % 4, tangent_axes)
                for sample in range(self._samples):
                    a, b = start + 2 * sample, start + 2 * sample + 1
                    faces.extend((a, b, b + 2, a, b + 2, a + 2))

            opposite_side = first + 4 * (self._samples + 1)
            boundary = (first, first + 1, opposite_side, opposite_side + 1)
            boundaries.append(boundary)
            if element == self.fem_model.element_count - 1:
                boundaries.append(tuple(index + 2 * self._samples for index in boundary))

            for xi, tangent_axes, cap_faces in (
                (-1.0, (2, 1), (0, 2, 1, 0, 3, 2)),
                (1.0, (1, 2), (0, 1, 2, 0, 2, 3)),
            ):
                if (xi == -1.0 and element == 0) or (xi == 1.0 and element == self.fem_model.element_count - 1):
                    start = len(shapes) - first
                    for corner in range(4):
                        add_vertex(element, xi, corner, tangent_axes)
                    faces.extend(start + index for index in cap_faces)
            triangles.extend(first + index for index in faces)

        device = self.fem_model.device
        self._render_shapes = wp.array(np.asarray(shapes), dtype=wp.float64, device=device)
        self._render_tangent_u = wp.array(np.asarray(tangent_u_shapes), dtype=wp.float64, device=device)
        self._render_tangent_v = wp.array(np.asarray(tangent_v_shapes), dtype=wp.float64, device=device)
        self._vertex_elements = wp.array(vertex_elements, dtype=int, device=device)
        self._vertices = wp.empty(len(shapes), dtype=wp.vec3, device=device)
        self._normals = wp.empty_like(self._vertices)
        self._triangles = wp.array(triangles, dtype=int, device=device)
        self._surface = ElementSurface(vertex_elements, self.fem_model.element_count, device)
        self._boundary_vertices = wp.array(boundaries, dtype=wp.vec4i, device=device)

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
                self.fem_model.element_dofs,
                self._vertex_elements,
                self._render_shapes,
                self._render_tangent_u,
                self._render_tangent_v,
                self._render_offset,
            ],
            outputs=[self._vertices, self._normals],
            device=self.fem_model.device,
            record_tape=False,
        )
        wp.launch(
            _render_boundaries,
            dim=len(self._boundary_vertices),
            inputs=[self._vertices, self._boundary_vertices],
            outputs=[self._boundary_starts, self._boundary_ends],
            device=self.fem_model.device,
            record_tape=False,
        )
        self._surface.render(
            self.viewer,
            "beam_elements",
            self._vertices,
            self._triangles,
            self._normals,
        )
        self.viewer.log_lines(
            "element_boundaries",
            self._boundary_starts,
            self._boundary_ends,
            self._boundary_colors,
            hidden=not show_edges or not self.viewer.layer.visible,
        )
        self._fixed_markers.render(self.viewer)
        self.node_arrows.render(self.viewer, self.state.q)

    def test_final(self, *, check_gravity=True):
        q = self.state.q.numpy().reshape(-1, self.fem_model.dofs_per_node)
        assert np.isfinite(q).all() and np.isfinite(self.state.qd.numpy()).all(), "Beam state is not finite"
        np.testing.assert_allclose(q[0], self._reference[: self.fem_model.dofs_per_node], atol=1.0e-12)
        assert np.max(np.abs(q[:, :3])) < 1.0, "Beam left the scene"
        if check_gravity and self.sim_time > 0.02:
            assert q[-1, 2] < 0.25, "Cantilever did not deflect under gravity"
