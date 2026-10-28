# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-element gray shades for a single batched surface mesh."""

import numpy as np
import warp as wp


class ElementSurface:
    """Draw a palette mesh with an explicit texture-enabled instance material."""

    def __init__(self, vertex_elements, element_count, device):
        indices = np.asarray(vertex_elements)
        uvs = np.column_stack(((indices + 0.5) / element_count, np.full(len(indices), 0.5)))
        shades = np.round(255 * np.linspace(0.25, 0.8, element_count)).astype(np.uint8)
        self.uvs = wp.array(uvs, dtype=wp.vec2, device=device)
        self.texture = np.repeat(shades[None, :, None], 3, axis=2)
        self.transforms = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        self.scales = wp.full(1, wp.vec3(1.0), dtype=wp.vec3, device=device)
        self.colors = wp.full(1, wp.vec3(1.0), dtype=wp.vec3, device=device)
        self.materials = wp.full(1, wp.vec4(0.5, 0.0, 0.0, 1.0), dtype=wp.vec4, device=device)

    def render(self, viewer, name, points, indices, normals, *, backface_culling=True):
        prototype = f"{name}/geometry"
        viewer.log_mesh(
            prototype,
            points,
            indices,
            normals=normals,
            uvs=self.uvs,
            texture=self.texture,
            hidden=True,
            backface_culling=backface_culling,
        )
        viewer.log_instances(
            name,
            prototype,
            self.transforms,
            self.scales,
            self.colors,
            self.materials,
            hidden=not viewer.layer.visible,
        )
