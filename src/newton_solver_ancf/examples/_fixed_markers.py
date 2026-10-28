# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent fixed-node markers with explicit layer visibility updates."""


class FixedNodeMarkers:
    """Submit immutable markers once, then only when their layer changes.

    Viewer backends retain named geometry between frames. Visibility still
    needs an explicit update because custom points do not pass through
    ``log_state``. Keeping the layer object also handles removal and recreation.
    """

    def __init__(self, points, colors, radius):
        self.points, self.colors, self.radius = points, colors, radius
        self._viewer = None
        self._layer = None
        self._visible = None

    def render(self, viewer):
        layer = viewer.layer
        visible = layer.visible
        if viewer is self._viewer and layer is self._layer and visible == self._visible:
            return
        viewer.log_points("fixed_nodes", self.points, radii=self.radius, colors=self.colors, hidden=not visible)
        self._viewer, self._layer, self._visible = viewer, layer, visible
