# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Static USD volume geometry, independent of any solver or material schema."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import _indices, _vertices, boundary_faces, hex_quadrature


@dataclass(frozen=True)
class Mesh:
    """One USD volume object in world-space metres, with Z as the up axis.

    ``name`` is the unique USD prim path. Connectivity references ``vertices``:
    four columns mean positively oriented Tet4; eight mean VTK-ordered Hex8.
    Surface triangles use the same vertices and outward winding. No material,
    density, chamber, contact, or attachment attributes are imported.
    """

    name: str
    vertices: np.ndarray
    indices: np.ndarray
    surface_triangles: np.ndarray

    @property
    def element_type(self):
        return "tet" if self.indices.shape[1] == 4 else "hex"


def _surface(elements, authored):
    """Keep the authored boundary triangulation, correcting its winding."""
    faces, _ = boundary_faces(elements)
    if faces.shape[1] == 3:
        choices = faces
        default = faces
        owners = np.arange(len(faces))
        diagonals = np.zeros(len(faces), dtype=int)
        per_face = 1
    else:
        # Either diagonal of a quad is valid, with consistent outward winding.
        choices = faces[:, [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]].reshape(-1, 3)
        default = faces[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
        owners = np.repeat(np.arange(len(faces)), 4)
        diagonals = np.tile([0, 0, 1, 1], len(faces))
        per_face = 2
    if authored is None or len(authored) == 0:
        return default.copy()
    authored = _indices(authored, 3)
    oriented = {tuple(sorted(face)): i for i, face in enumerate(choices)}
    try:
        selection = np.array([oriented[tuple(sorted(face))] for face in authored])
    except KeyError as error:
        raise ValueError("An authored surface triangle is not on the volume boundary") from error
    if len(np.unique(selection)) != len(selection):
        raise ValueError("The authored surface contains duplicate triangles")
    if np.any(np.bincount(owners[selection], minlength=len(faces)) != per_face):
        raise ValueError("The authored surface must cover every boundary face exactly once")
    if per_face == 2:
        diagonal_count = np.bincount(owners[selection], weights=diagonals[selection], minlength=len(faces))
        if np.any(diagonal_count == 1):
            raise ValueError("The authored surface mixes both diagonals of a boundary quad")
    return choices[selection].copy()


def load_mesh(path: str | Path) -> list[Mesh]:
    """Load all Tet4/Hex8 objects from a static USD file in traversal order.

    Reads ``TetMesh`` points/tetVertexIndices/surfaceFaceVertexIndices and the
    reference assets' ``Mesh`` + ``customData.hex_mesh.hex_vertex_indices``
    schema. Other prims, including surface-only meshes, are not volume objects.
    Standard composed USD transforms, stage units and Y/Z up-axis metadata are
    applied at default time. Reflection reverses connectivity so volumes stay
    positive. Missing surface triangles are derived from the volume boundary.

    OpenUSD is imported only here: install ``newton-solver-ancf[examples]`` to load assets.
    The function imports geometry only; arbitrary customData is not returned.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from pxr import Usd, UsdGeom
    except ModuleNotFoundError as error:
        if error.name != "pxr":
            raise
        raise ImportError("USD loading requires the optional 'newton-solver-ancf[examples]' dependency") from error
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD stage {path}")
    metres = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = str(UsdGeom.GetStageUpAxis(stage))
    if not np.isfinite(metres) or metres <= 0 or up not in ("Y", "Z"):
        raise ValueError("The USD stage must have positive units and Y or Z up")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    objects = []
    for prim in stage.Traverse():
        hex_data = prim.GetCustomDataByKey("hex_mesh") if prim.GetTypeName() == "Mesh" else None
        tetrahedral = prim.GetTypeName() == "TetMesh"
        if not tetrahedral and not hex_data:
            continue
        name = str(prim.GetPath())
        points = prim.GetAttribute("points").Get()
        if points is None:
            raise ValueError(f"{name}: missing default-time points")
        vertices = _vertices(points).copy()
        transform = np.asarray(transforms.GetLocalToWorldTransform(prim), dtype=np.float64)
        vertices = (vertices @ transform[:3, :3] + transform[3, :3]) * metres
        if up == "Y":
            vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
        vertices = _vertices(vertices)
        if tetrahedral:
            indices = _indices(prim.GetAttribute("tetVertexIndices").Get(), 4, len(vertices)).copy()
            authored = prim.GetAttribute("surfaceFaceVertexIndices").Get()
            edges = vertices[indices[:, 1:]] - vertices[indices[:, :1]]
            determinant = np.linalg.det(edges)
            scale = np.prod(np.linalg.norm(edges, axis=2), axis=1)
            if np.any(~np.isfinite(determinant)) or np.any(np.abs(determinant) <= 1.0e-12 * scale):
                raise ValueError(f"{name}: a tetrahedron is degenerate")
            inverted = determinant < 0
            indices[inverted] = indices[inverted][:, [0, 2, 1, 3]]
        else:
            raw = np.asarray(hex_data.get("hex_vertex_indices", []))
            if raw.ndim != 1 or raw.size % 8:
                raise ValueError(f"{name}: hex_vertex_indices must contain eight indices per element")
            indices = _indices(raw.reshape(-1, 8), 8, len(vertices)).copy()
            if np.linalg.det(transform[:3, :3]) < 0:
                indices = indices[:, [1, 0, 3, 2, 5, 4, 7, 6]]
            hex_quadrature(vertices, indices)
            counts = prim.GetAttribute("faceVertexCounts").Get()
            authored = prim.GetAttribute("faceVertexIndices").Get()
            if authored is not None and len(authored):
                if counts is None or np.any(np.asarray(counts) != 3) or len(authored) != 3 * len(counts):
                    raise ValueError(f"{name}: authored Hex8 surface must contain triangles")
                authored = np.asarray(authored).reshape(-1, 3)
        if not len(indices):
            raise ValueError(f"{name}: volume connectivity is empty")
        objects.append(Mesh(name, vertices, indices, _surface(indices, authored)))
    if not objects:
        raise ValueError(f"No Tet4 or Hex8 volume geometry found in {path}")
    return objects
