# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference-27 Baymax cage, material volume and detailed surface placement."""

import importlib.util
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from newton_solver_ancf import SolverANCFTetrahedron, load_mesh, tetrahedral_cage
from newton_solver_ancf.examples import example_baymax_drop


@unittest.skipUnless(importlib.util.find_spec("pxr"), "Install the examples extra for USD assets")
class TestBaymaxInflatable(unittest.TestCase):
    def test_reference_geometry_defaults_and_affine_surface(self):
        args = example_baymax_drop.Example.create_parser().parse_args([])
        self.assertEqual(args.substeps, 1)
        self.assertEqual((args.resolution, args.young_modulus, args.density, args.drop_height), (6, 5.0e4, 200.0, 0.15))
        path = Path(example_baymax_drop.__file__).with_name("assets") / "tet_baymax.usda"
        mesh = load_mesh(path)[0]
        mesh = replace(mesh, vertices=mesh.vertices[:, (0, 2, 1)] * (1.0, -1.0, 1.0))
        cage = tetrahedral_cage(mesh, args.resolution)
        self.assertEqual((len(mesh.vertices), len(mesh.indices)), (8965, 38730))
        self.assertEqual((len(cage.positions), len(cage.elements)), (84, 204))
        self.assertEqual((len(cage.surface_positions), len(cage.surface_triangles)), (4943, 9882))
        self.assertAlmostEqual(cage.source_volume, 0.03757605496349353, delta=1.0e-12)
        corners = cage.positions[cage.elements]
        volumes = np.linalg.det(np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)) / 6.0
        self.assertTrue((volumes > 0).all())
        self.assertTrue((cage.volume_fraction > 0).all())
        self.assertTrue((cage.volume_fraction <= 1.0 + 2.0e-8).all())
        self.assertAlmostEqual(float(volumes @ cage.volume_fraction), cage.source_volume, delta=1.0e-12)
        triangles = cage.surface_positions[cage.surface_triangles]
        signed_volume = float(
            np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        )
        self.assertAlmostEqual(signed_volume, cage.source_volume, delta=1.0e-12)
        shift = np.r_[
            -0.5 * (cage.surface_positions[:, :2].min(0) + cage.surface_positions[:, :2].max(0)),
            args.drop_height - cage.surface_positions[:, 2].min(),
        ]
        positions = cage.positions + shift
        self.assertAlmostEqual(float((cage.surface_positions + shift)[:, 2].min()), 0.15)
        parents = cage.elements[cage.surface_elements]
        shape, _ = SolverANCFTetrahedron.shape_functions(cage.barycentric, positions[parents], derivatives=False)
        for deformation, translation in (
            (np.eye(3), np.zeros(3)),
            (np.array([[1.1, 0.03, 0], [0, 0.9, 0.02], [0.01, 0, 1.05]]), np.array([0.02, -0.03, 0.04])),
        ):
            q = np.empty((len(positions), 4, 3))
            q[:, 0] = positions @ deformation.T + translation
            q[:, 1:] = deformation.T
            interpolated = np.einsum("pa,pai->pi", shape, q[parents].reshape(-1, 16, 3))
            expected = (cage.surface_positions + shift) @ deformation.T + translation
            np.testing.assert_allclose(interpolated, expected, rtol=0, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
