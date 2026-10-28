# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Baked example geometry, reference dimensions and source-volume preservation."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from newton_solver_ancf import boundary_faces, examples, load_element_mesh

ASSETS = Path(examples.__file__).with_name("assets")


@unittest.skipUnless(importlib.util.find_spec("pxr"), "Install the examples extra for USD assets")
class TestBakedGeometry(unittest.TestCase):
    def test_element_geometry_and_orientation(self):
        cases = (
            ("Cable/e8", 9, 8, (0.0, 0.0, 0.6), (0.5, 0.0, 0.6)),
            ("Beam3243/e6", 7, 6, (0.0, 0.0, 0.25), (0.4, 0.0, 0.25)),
            ("Beam3333/e6", 13, 12, (0.0, 0.0, 0.25), (0.4, 0.0, 0.25)),
            ("Shell/e4", 10, 4, (0.0, -0.04, 0.3), (0.3, 0.04, 0.3)),
            ("Tetrahedron/e4", 20, 24, (0.0, -0.025, 0.28), (0.3, 0.025, 0.32)),
        )
        for prim, nodes, count, lower, upper in cases:
            with self.subTest(mesh=prim):
                points, elements = load_element_mesh(ASSETS / "element_meshes.usdc", f"/Elements/{prim}")
                self.assertEqual(points.shape, (nodes, 3))
                self.assertEqual(len(elements), count)
                self.assertEqual(points.dtype, np.float64)
                self.assertEqual(elements.dtype, np.int32)
                np.testing.assert_allclose(points.min(axis=0), lower, rtol=0, atol=1.0e-16)
                np.testing.assert_allclose(points.max(axis=0), upper, rtol=0, atol=1.0e-16)
                if prim.startswith("Tetrahedron"):
                    local = points[elements]
                    volumes = np.linalg.det(local[:, 1:] - local[:, :1]) / 6.0
                    self.assertTrue(np.all(volumes > 0.0))
                    self.assertAlmostEqual(volumes.sum(), 0.3 * 0.05 * 0.04, delta=1.0e-16)
                    faces, owners = boundary_faces(elements)
                    self.assertEqual(len(faces), 36)
                    self.assertTrue(np.all((owners >= 0) & (owners < count)))
                elif prim.startswith("Shell"):
                    local = points[elements]
                    normals = np.cross(local[:, 1] - local[:, 0], local[:, 3] - local[:, 0])
                    self.assertTrue(np.all(normals[:, 2] > 0.0))


if __name__ == "__main__":
    unittest.main()
