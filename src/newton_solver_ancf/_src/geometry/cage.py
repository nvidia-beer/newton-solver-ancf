# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Coarse tetrahedral reference cages and geometric surface embedding.

Occupied volume and surface embedding are geometric quantities; their
conversion to density, stiffness, or solver coordinates belongs to callers.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from .mesh import _indices, _vertices
from .usd import Mesh

_HEX_TETRAHEDRA = np.array(((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)))


@dataclass(frozen=True)
class TetrahedralCage:
    """Reference geometry and point-to-tetrahedron coordinates, all in metres.

    ``volume_fraction`` is the source occupancy of each cage tetrahedron.
    ``surface_elements`` chooses one cage element per surface vertex;
    ``barycentric`` stores its four geometric coordinates in that element.
    No material, mass, shape-function, or solver-state data is stored.
    """

    positions: np.ndarray
    elements: np.ndarray
    volume_fraction: np.ndarray
    surface_positions: np.ndarray
    surface_triangles: np.ndarray
    surface_elements: np.ndarray
    barycentric: np.ndarray
    source_volume: float


def _positive_tetrahedra(points, elements):
    elements = np.asarray(elements, dtype=np.int32).copy()
    vertices = np.asarray(points)[elements]
    determinant = np.linalg.det((vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1))
    if not np.isfinite(determinant).all() or np.any(determinant == 0):
        raise ValueError("the source or cage contains a degenerate tetrahedron")
    negative = determinant < 0
    elements[negative, 1], elements[negative, 2] = elements[negative, 2].copy(), elements[negative, 1].copy()
    return elements


def _volumes(points, elements):
    vertices = points[elements]
    return np.abs(np.linalg.det((vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1))) / 6


def _clip_polyhedron(faces, axis, coordinate, keep_above):
    """Clip a convex polyhedron, retaining the new planar cap."""
    clipped, cap = [], []
    sign = 1.0 if keep_above else -1.0
    for polygon in faces:
        result = []
        previous = polygon[-1]
        previous_distance = sign * (previous[axis] - coordinate)
        for current in polygon:
            distance = sign * (current[axis] - coordinate)
            if (distance >= 0.0) != (previous_distance >= 0.0):
                point = previous + previous_distance / (previous_distance - distance) * (current - previous)
                result.append(point)
                cap.append(point)
            if distance >= 0.0:
                result.append(current)
            previous, previous_distance = current, distance
        if len(result) >= 3:
            clipped.append(np.asarray(result))
    if len(cap) >= 3:
        cap = np.unique(np.round(cap, decimals=14), axis=0)
        if len(cap) >= 3:
            plane = np.delete(cap - cap.mean(axis=0), axis, axis=1)
            order = np.argsort(np.arctan2(plane[:, 1], plane[:, 0]))
            clipped.append(cap[order])
    return clipped


def _polyhedron_volume(faces):
    if not faces:
        return 0.0
    center = np.concatenate(faces).mean(axis=0)
    volume = 0.0
    for polygon in faces:
        edges = polygon - center
        for index in range(1, len(edges) - 1):
            volume += abs(np.dot(edges[0], np.cross(edges[index], edges[index + 1]))) / 6.0
    return volume


def _cell_source_volumes(points, tetrahedra, origin, spacing, dimensions):
    """Integrate exact tetrahedron/box intersections, without sampling or discarding source volume."""
    vertices = points[tetrahedra]
    volumes = np.abs(np.linalg.det((vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1))) / 6.0
    if np.any(volumes <= 0.0):
        raise ValueError("source tetrahedra must have positive volume")
    cells = np.clip(np.floor((vertices - origin) / spacing).astype(int), 0, dimensions - 1)
    minimum, maximum = cells.min(axis=1), cells.max(axis=1)
    contained = np.all(minimum == maximum, axis=1)
    result = np.zeros(tuple(dimensions), dtype=np.float64)
    np.add.at(result, tuple(minimum[contained].T), volumes[contained])
    for index in np.flatnonzero(~contained):
        corners = vertices[index]
        source_faces = [corners[list(face)] for face in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))]
        for cell in product(*(range(minimum[index, axis], maximum[index, axis] + 1) for axis in range(3))):
            lower, upper = origin + np.asarray(cell) * spacing, origin + (np.asarray(cell) + 1) * spacing
            faces = source_faces
            for axis in range(3):
                if corners[:, axis].min() < lower[axis]:
                    faces = _clip_polyhedron(faces, axis, lower[axis], True)
                if corners[:, axis].max() > upper[axis]:
                    faces = _clip_polyhedron(faces, axis, upper[axis], False)
                if not faces:
                    break
            result[cell] += _polyhedron_volume(faces)
    if not np.isclose(result.sum(), volumes.sum(), rtol=2.0e-10, atol=1.0e-14):
        raise ValueError("tetrahedron/cell intersections did not conserve source source volume")
    if result.max() > np.prod(spacing) * (1.0 + 2.0e-8):
        raise ValueError("source tetrahedra overlap: a cell contains more material than its volume")
    return result


def _cartesian_cage(points, tetrahedra, resolution):
    origin, extent = points.min(axis=0), np.ptp(points, axis=0)
    dimensions = np.maximum(1, np.ceil(resolution * extent / extent.max()).astype(int))
    spacing = extent / dimensions
    cell_volumes = _cell_source_volumes(points, tetrahedra, origin, spacing, dimensions)
    occupied = np.argwhere(cell_volumes > np.prod(spacing) * 1.0e-12)
    corners = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
    mirror = np.array((1, 0, 3, 2, 5, 4, 7, 6))
    nodes, positions, elements, fractions = {}, [], [], []
    for cell in occupied:
        ids = []
        for coordinate in cell + corners:
            key = tuple(coordinate)
            if key not in nodes:
                nodes[key] = len(positions)
                positions.append(origin + coordinate * spacing)
            ids.append(nodes[key])
        local = mirror[_HEX_TETRAHEDRA] if cell[0] % 2 else _HEX_TETRAHEDRA
        elements.extend(np.asarray(ids)[local])
        fractions.extend([cell_volumes[tuple(cell)] / np.prod(spacing)] * 6)
    positions = np.asarray(positions)
    return positions, _positive_tetrahedra(positions, elements), np.asarray(fractions)


def _embedding(positions, elements, surface):
    corners = positions[elements]
    inverse = np.linalg.inv((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1))
    owners = np.empty(len(surface), dtype=np.int32)
    coordinates = np.empty((len(surface), 4))
    for index, point in enumerate(surface):
        local = np.einsum("eij,ej->ei", inverse, point - corners[:, 0])
        barycentric = np.column_stack((1 - local.sum(axis=1), local))
        owner = int(np.argmax(barycentric.min(axis=1)))
        if barycentric[owner].min() < -2e-8:
            raise ValueError("the cage does not enclose its detailed surface")
        owners[index], coordinates[index] = owner, barycentric[owner]
    return owners, coordinates


def tetrahedral_cage(mesh: Mesh, resolution: int = 6) -> TetrahedralCage:
    """Enclose a tetrahedral asset with a Cartesian cage and its original surface.

    Source-tetrahedron/box intersections determine each cell's occupied volume.
    ``resolution >= 2`` counts Cartesian cells along the longest source axis.
    Occupied cells split into six conforming, positively oriented tetrahedra.
    The result has float64 coordinates and int32 connectivity/parent indices.
    Scene transforms and solver interpolation are the caller's responsibility.
    """
    if not isinstance(resolution, (int, np.integer)) or isinstance(resolution, bool) or resolution < 2:
        raise ValueError("resolution must be an integer of at least two")
    points = _vertices(mesh.vertices)
    indices = _indices(mesh.indices, 4, len(points))
    triangles = _indices(mesh.surface_triangles, 3, len(points))
    if not len(indices) or not len(triangles) or np.any(np.ptp(points, axis=0) <= 0):
        raise ValueError("the source must have volume cells, boundary triangles, and span all three axes")
    source_tets = _positive_tetrahedra(points, indices)
    source_volume = float(_volumes(points, source_tets).sum())
    positions, elements, fractions = _cartesian_cage(points, source_tets, resolution)
    surface_ids, surface_triangles = np.unique(triangles, return_inverse=True)
    surface = points[surface_ids].copy()
    owners, barycentric = _embedding(positions, elements, surface)
    return TetrahedralCage(
        positions,
        elements,
        fractions,
        surface,
        surface_triangles.reshape(-1, 3).astype(np.int32),
        owners,
        barycentric,
        source_volume,
    )
