# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""An ANCF cable with axial and bending stiffness sags under gravity, with one end clamped.

The initially horizontal, 0.5 m cable uses eight two-node cubic Hermite
elements by default. The root position and tangent are fixed. Axial stretching
and bending resist deformation, and mass-proportional damping reduces motion.
The material model has no torsional stiffness. --debug_arrows shows the
nodal tangents.

"""

from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.math import quat_between_vectors_robust

from newton_solver_ancf import SolverANCFCable, load_element_mesh
from newton_solver_ancf.examples._fixed_markers import FixedNodeMarkers
from newton_solver_ancf.examples._node_arrows import NodeGradientArrows


@wp.kernel
def _render_centerline(
    q: wp.array[wp.float64],
    lengths: wp.array[wp.float64],
    samples: int,
    render_offset: wp.vec3,
    centers: wp.array[wp.vec3],
    nodes: wp.array[wp.vec3],
):
    point = wp.tid()
    element = wp.min(point // samples, lengths.shape[0] - 1)
    xi = wp.float64(point - element * samples) / wp.float64(samples)
    xi2, xi3 = xi * xi, xi * xi * xi
    left = 6 * element
    right = left + 6
    p0 = wp.vec3d(q[left], q[left + 1], q[left + 2])
    d0 = wp.vec3d(q[left + 3], q[left + 4], q[left + 5])
    p1 = wp.vec3d(q[right], q[right + 1], q[right + 2])
    d1 = wp.vec3d(q[right + 3], q[right + 4], q[right + 5])
    position = (
        (wp.float64(1.0) - wp.float64(3.0) * xi2 + wp.float64(2.0) * xi3) * p0
        + lengths[element] * (xi - wp.float64(2.0) * xi2 + xi3) * d0
        + (wp.float64(3.0) * xi2 - wp.float64(2.0) * xi3) * p1
        + lengths[element] * (-xi2 + xi3) * d1
    )
    # Keep intermediate centers local so the display offset cannot alter
    # segment directions and cap overlap through float32 cancellation.
    centers[point] = wp.vec3(position)
    if point % samples == 0:
        nodes[point // samples] = wp.vec3(position) + render_offset


@wp.func
def _joint_overlap(direction: wp.vec3, neighbor: wp.vec3, radius: float):
    cosine = wp.clamp(wp.dot(direction, wp.normalize(neighbor)), -1.0, 1.0)
    # Overlap cylinder ends across a bend so their flat caps leave no visible cracks.
    return radius * wp.min(wp.sqrt((1.0 - cosine) / wp.max(1.0 + cosine, 1.0e-6)), 1.0)


@wp.kernel
def _render_cylinders(
    centers: wp.array[wp.vec3],
    radius: float,
    render_offset: wp.vec3,
    transforms: wp.array[wp.transform],
    scales: wp.array[wp.vec3],
):
    segment = wp.tid()
    start, end = centers[segment], centers[segment + 1]
    delta = end - start
    length = wp.length(delta)
    direction = wp.vec3(0.0, 0.0, 1.0)
    if length > 1.0e-9:
        direction = delta / length
    start_overlap, end_overlap = float(0.0), float(0.0)
    if segment > 0:
        start_overlap = _joint_overlap(direction, start - centers[segment - 1], radius)
    if segment + 2 < centers.shape[0]:
        end_overlap = _joint_overlap(direction, centers[segment + 2] - end, radius)
    center = 0.5 * (start + end) + 0.5 * (end_overlap - start_overlap) * direction
    rotation = quat_between_vectors_robust(wp.vec3(0.0, 0.0, 1.0), direction)
    transforms[segment] = wp.transform(center + render_offset, rotation)
    scales[segment] = wp.vec3(radius, radius, 0.5 * (length + start_overlap + end_overlap))


class Cable:
    def __init__(self, viewer, *, substeps=4, debug_arrows=False, render_offset=(0.0, 0.0, 0.0)):
        elements = 8
        if substeps < 1:
            raise ValueError("substeps must be positive")
        self.viewer = viewer
        self.sim_substeps = substeps
        self._render_offset = wp.vec3(render_offset)
        self.sim_time, self.frame_dt = 0.0, 1.0 / 60.0
        self.sim_dt = self.frame_dt / substeps
        positions, _ = load_element_mesh(
            Path(__file__).with_name("assets") / "element_meshes.usdc", "/Elements/Cable/e8"
        )
        self.cable_model = SolverANCFCable.create_model(positions, young_modulus=2.0e7, radius=0.005, density=1000.0)
        builder = newton.ModelBuilder()
        self.cable_model.register(builder)
        native_model = builder.finalize(device=self.cable_model.device)
        self.solver = SolverANCFCable(
            native_model,
            self.cable_model,
            fixed_dofs=range(6),
            gravity=(0, 0, -9.81),
            mass_damping=2.0,
            max_iterations=30,
            tolerance=1.0e-8,
        )
        self.state = self.solver.initial_state()
        self.node_arrows = NodeGradientArrows(
            self.cable_model, axes=(0,), enabled=debug_arrows, render_offset=render_offset
        )
        self._reference = self.state.q.numpy().copy()
        self._fixed_points = wp.array(
            positions[:1].astype(np.float32) + np.asarray(render_offset, dtype=np.float32),
            dtype=wp.vec3,
            device=self.cable_model.device,
        )
        self._fixed_colors = wp.full(1, wp.vec3(1.0, 0.45, 0.05), dtype=wp.vec3, device=self.cable_model.device)
        self._fixed_markers = FixedNodeMarkers(self._fixed_points, self._fixed_colors, 0.008)
        self._samples = 12
        point_count = elements * self._samples + 1
        segment_count = point_count - 1
        element_ids = np.arange(segment_count) // self._samples
        shades = np.linspace(0.25, 0.8, elements)
        self._colors = wp.array(
            np.repeat(shades[element_ids, None], 3, axis=1), dtype=wp.vec3, device=self.cable_model.device
        )
        self._centers = wp.empty(point_count, dtype=wp.vec3, device=self.cable_model.device)
        self._cylinder_transforms = wp.empty(segment_count, dtype=wp.transform, device=self.cable_model.device)
        self._cylinder_scales = wp.empty(segment_count, dtype=wp.vec3, device=self.cable_model.device)
        self._cylinder_materials = wp.full(
            segment_count, wp.vec4(0.5, 0.0, 0.0, 0.0), dtype=wp.vec4, device=self.cable_model.device
        )
        self._nodes = wp.empty(len(positions), dtype=wp.vec3, device=self.cable_model.device)
        self._node_colors = wp.full(len(positions), wp.vec3(0.9), dtype=wp.vec3, device=self.cable_model.device)
        self.viewer.log_geo("cable_cylinder", newton.GeoType.CYLINDER, (1.0, 1.0), 0.0, True, hidden=True)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state = self.solver.advance(self.state, self.sim_dt)
        self.sim_time += self.frame_dt

    def render(self, *, show_edges=False):
        # The cable's cylindrical segments already distinguish its elements.
        wp.launch(
            _render_centerline,
            dim=len(self._centers),
            inputs=[self.state.q, self.cable_model.lengths, self._samples, self._render_offset],
            outputs=[self._centers, self._nodes],
            device=self.cable_model.device,
            record_tape=False,
        )
        wp.launch(
            _render_cylinders,
            dim=len(self._cylinder_transforms),
            inputs=[self._centers, 0.005, self._render_offset],
            outputs=[self._cylinder_transforms, self._cylinder_scales],
            device=self.cable_model.device,
            record_tape=False,
        )
        self.viewer.log_instances(
            "cable",
            "cable_cylinder",
            self._cylinder_transforms,
            self._cylinder_scales,
            self._colors,
            self._cylinder_materials,
            hidden=not self.viewer.layer.visible,
        )
        self.viewer.log_points(
            "nodes", self._nodes, radii=0.007, colors=self._node_colors, hidden=not self.viewer.layer.visible
        )
        self._fixed_markers.render(self.viewer)
        self.node_arrows.render(self.viewer, self.state.q)

    def test_final(self, *, check_gravity=True):
        coordinates, rates = self.state.q.numpy(), self.state.qd.numpy()
        assert np.isfinite(coordinates).all() and np.isfinite(rates).all(), "Cable state is not finite"
        np.testing.assert_allclose(coordinates[:6], self._reference[:6], atol=1.0e-12)
        assert np.linalg.norm(coordinates.reshape(-1, 6)[-1, :3] - self._reference[:3]) < 0.65, (
            "Cable stretched excessively"
        )
        if check_gravity and self.sim_time > 0.0:
            assert coordinates[-4] < self._reference[-4], "Cable did not bend under gravity"
