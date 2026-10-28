# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Geometry in metres, int32 connectivity, and float64 reference quadrature.

Hex8 uses VTK order: (-,-,-), (+,-,-), (+,+,-), (-,+,-),
(-,-,+), (+,-,+), (+,+,+), (-,+,+). Tet4 orientation is
det([x1-x0, x2-x0, x3-x0]) > 0. No material or solver state lives here.
"""

from itertools import product

import numpy as np

_SIGNS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=np.float64,
)
_HEX_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
_TET_FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))


def _vertices(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite with shape (V, 3)")
    return vertices


def _indices(indices, width=None, vertex_count=None):
    indices = np.asarray(indices)
    widths = (4, 8) if width is None else (width,)
    if indices.ndim != 2 or indices.shape[1] not in widths or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"indices must be an integer array with shape (E, {widths})")
    if indices.size and (indices.min() < 0 or indices.max() > np.iinfo(np.int32).max):
        raise ValueError("indices must be nonnegative int32 node IDs")
    if vertex_count is not None and indices.size and indices.max() >= vertex_count:
        raise ValueError("an element references a missing vertex")
    if np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
        raise ValueError("each element must reference distinct vertices")
    return indices.astype(np.int32, copy=False)


def boundary_faces(elements):
    """Return (outward boundary faces, owning element IDs) for Tet4 or Hex8.

    Input cells must be positively oriented. Only connectivity is inspected;
    coordinates and geometric orientation cannot be checked here. Duplicate
    cells, faces with more than two owners, and inconsistent interior face
    winding raise ValueError. This does not detect geometric intersections.
    """
    elements = _indices(elements)
    if len(np.unique(np.sort(elements, axis=1), axis=0)) != len(elements):
        raise ValueError("duplicate elements are not a valid volume mesh")
    local_faces = _TET_FACES if elements.shape[1] == 4 else _HEX_FACES
    faces = {}
    for owner, element in enumerate(elements):
        for local_face in local_faces:
            face = tuple(element[list(local_face)])
            key = tuple(sorted(face))
            if key not in faces:
                faces[key] = (face, owner)
                continue
            previous, _ = faces[key]
            if previous is None:
                raise ValueError("nonmanifold face has more than two owning elements")
            reversed_face = face[::-1]
            if not any(previous == reversed_face[k:] + reversed_face[:k] for k in range(len(face))):
                raise ValueError("interior face winding is inconsistent; check element orientation")
            faces[key] = (None, -1)
    boundary = [(face, owner) for face, owner in faces.values() if face is not None]
    return (
        np.array([face for face, _ in boundary], dtype=np.int32).reshape(-1, len(local_faces[0])),
        np.array([owner for _, owner in boundary], dtype=np.int32),
    )


def _shape(points):
    factors = 1.0 + points[:, None, :] * _SIGNS[None, :, :]
    values = np.prod(factors, axis=2) / 8.0
    derivatives = np.empty((len(points), 8, 3))
    for d in range(3):
        other = [axis for axis in range(3) if axis != d]
        derivatives[:, :, d] = _SIGNS[None, :, d] * np.prod(factors[:, :, other], axis=2) / 8.0
    return values, derivatives


def _jacobians(local, derivatives):
    jacobians = np.einsum("hai,gaj->hgij", local, derivatives)
    determinants = np.linalg.det(jacobians)
    scale = np.prod(np.linalg.norm(jacobians, axis=2), axis=2)
    if np.any(~np.isfinite(determinants)) or np.any(determinants <= 1.0e-12 * scale):
        raise ValueError("Hex8 has a nonpositive or numerically singular reference Jacobian")
    return jacobians, determinants


def hex_quadrature(vertices, indices, order=2):
    """Return (N[G,8], dN_dX[H,G,8,3], volume_weights[H,G]) for Hex8.

    ``order`` is 1, 2, or 3 Gauss points per reference axis; G=order**3.
    Gradients are in inverse metres and weights in cubic metres. Corner and
    quadrature Jacobians must be positive. These samples are a quality check,
    not a proof of global validity of an arbitrarily warped hexahedron.
    """
    vertices, indices = _vertices(vertices), _indices(indices, 8)
    if indices.size and indices.max() >= len(vertices):
        raise ValueError("an element references a missing vertex")
    if not isinstance(order, (int, np.integer)) or isinstance(order, bool) or order not in (1, 2, 3):
        raise ValueError("quadrature order must be 1, 2, or 3")
    abscissae, weights = np.polynomial.legendre.leggauss(order)
    points = np.array(list(product(abscissae, repeat=3)))
    gauss_weights = np.prod(np.array(list(product(weights, repeat=3))), axis=1)
    values, derivatives = _shape(points)
    local = vertices[indices] - vertices[indices[:, :1]]
    _jacobians(local, _shape(_SIGNS)[1])
    jacobians, determinants = _jacobians(local, derivatives)
    gradients = np.einsum("gaj,hgji->hgai", derivatives, np.linalg.inv(jacobians))
    return values, gradients, determinants * gauss_weights
