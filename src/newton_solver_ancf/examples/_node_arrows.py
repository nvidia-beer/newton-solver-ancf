# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reusable, device-resident visualization of ANCF nodal gradients."""

import warp as wp
from newton.viewer import ViewerBase

from newton_solver_ancf import ANCFModel


@wp.kernel
def _update_arrows(
    q: wp.array[wp.float64],
    dofs_per_node: int,
    gradient_count: int,
    length_scale: wp.float64,
    render_offset: wp.vec3,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    node, gradient = wp.tid()
    base = node * dofs_per_node
    offset = base + 3 * (gradient + 1)
    position = wp.vec3d(q[base], q[base + 1], q[base + 2])
    direction = wp.vec3d(q[offset], q[offset + 1], q[offset + 2])
    tip = position + length_scale * direction
    arrow = node * gradient_count + gradient
    starts[arrow] = wp.vec3(position) + render_offset
    ends[arrow] = wp.vec3(tip) + render_offset


class NodeGradientArrows:
    """Display actual nodal gradients without normalizing their magnitudes.

    Colors identify material-coordinate gradients: red X, green Y, blue Z.
    Positions stay at the actual nodes; one constant length scale converts
    dimensionless gradients into visible arrow lengths.
    """

    def __init__(
        self,
        model: ANCFModel,
        axes: tuple[int, ...],
        length_scale: float = 0.04,
        *,
        enabled=False,
        render_offset=(0.0, 0.0, 0.0),
    ):
        if model.dofs_per_node != 3 * (1 + len(axes)) or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("axes must identify every gradient in the nodal coordinate layout")
        self.model = model
        self.gradient_count = len(axes)
        self.length_scale = wp.float64(length_scale)
        self.render_offset = wp.vec3(render_offset)
        self.enabled = enabled
        self._visible = False
        colors = ((0.95, 0.15, 0.15), (0.2, 0.9, 0.3), (0.25, 0.45, 1.0))
        self.starts = wp.empty(model.node_count * len(axes), dtype=wp.vec3, device=model.device)
        self.ends = wp.empty_like(self.starts)
        self.colors = wp.array(
            [colors[axis] for _ in range(model.node_count) for axis in axes], dtype=wp.vec3, device=model.device
        )

    def render(self, viewer: ViewerBase, q: wp.array[wp.float64]):
        if not self.enabled or not viewer.layer.visible:
            if self._visible:
                viewer.log_arrows("node_gradients", self.starts, self.ends, self.colors, hidden=True)
                self._visible = False
            return
        # Cached buffers and one device kernel per rendered frame: no state
        # copies to the host and no work inside the physics substeps.
        wp.launch(
            _update_arrows,
            dim=(self.model.node_count, self.gradient_count),
            inputs=[q, self.model.dofs_per_node, self.gradient_count, self.length_scale, self.render_offset],
            outputs=[self.starts, self.ends],
            device=self.model.device,
            record_tape=False,
        )
        viewer.log_arrows("node_gradients", self.starts, self.ends, self.colors)
        self._visible = True
