# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read authored double-precision element geometry."""

from pathlib import Path

import numpy as np

from .mesh import _indices, _vertices


def _stage(path):
    from pxr import Usd, UsdGeom

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD stage {path}")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("Authored ANCF reference arrays must use world-space metres and Z up")
    return stage


def _array(prim, name, dtype, width=None):
    value = prim.GetAttribute(f"ancf:{name}").Get()
    if value is None:
        raise ValueError(f"{prim.GetPath()}: missing ancf:{name}")
    array = np.asarray(value, dtype=dtype)
    if width is not None:
        if array.size % width:
            raise ValueError(f"ancf:{name} must have {width} values per entry")
        array = array.reshape(-1, width)
    if not np.isfinite(array).all():
        raise ValueError(f"ancf:{name} contains non-finite values")
    return array.copy()


def load_element_mesh(path: str | Path, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read authored positions [m] and integer connectivity from a USD prim.

    The ``ancf:positions`` double3 array preserves the original reference
    coordinates; ``ancf:elements`` is flat connectivity with the width stored
    in ``ancf:nodesPerElement``. No mesh is generated or resampled.
    """
    stage = _stage(path)
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No authored element mesh at {prim_path}")
    positions = _vertices(_array(prim, "positions", np.float64, 3))
    width = prim.GetAttribute("ancf:nodesPerElement").Get()
    if width not in (2, 4, 8):
        raise ValueError("ancf:nodesPerElement must be 2, 4 or 8")
    elements = _indices(_array(prim, "elements", np.int32, width), width, len(positions))
    return positions, elements
