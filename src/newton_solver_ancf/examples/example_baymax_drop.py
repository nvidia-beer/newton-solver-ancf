# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Baymax drops onto a frictionless floor with its tetrahedral cage and detailed surface visible.

The default resolution-6 mechanical cage has 84 nodes and 204 cubic ANCF/FN
solid elements. The separate 9,882-triangle USD surface follows the cage's
interpolation. The filled elastic body starts 0.15 m above the floor, with
no fixed nodes. Floor forces act at cage nodes. Use --no-show-cage or the
viewer checkbox to hide the simulation mesh.
"""

from argparse import BooleanOptionalAction
from dataclasses import replace
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_ancf import SolverANCFTetrahedron, load_mesh, tetrahedral_cage


@wp.kernel
def _interpolate(
    q: wp.array[wp.float64],
    vectors: wp.array2d[int],
    shape: wp.array2d[wp.float64],
    points: wp.array[wp.vec3],
):
    i = wp.tid()
    point = wp.vec3d(0.0)
    for j in range(16):
        index = 3 * vectors[i, j]
        point += shape[i, j] * wp.vec3d(q[index], q[index + 1], q[index + 2])
    points[i] = wp.vec3(point)


@wp.kernel
def _cage_edges(
    q: wp.array[wp.float64],
    edges: wp.array[wp.vec2i],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    i = wp.tid()
    a, b = 12 * edges[i][0], 12 * edges[i][1]
    starts[i] = wp.vec3(float(q[a]), float(q[a + 1]), float(q[a + 2]))
    ends[i] = wp.vec3(float(q[b]), float(q[b + 1]), float(q[b + 2]))


class Example:
    """Drop one filled elastic solid; the floor contacts its mechanical cage."""

    def __init__(self, viewer, args=None):
        args = newton.examples.default_args(self.create_parser()) if args is None else args
        positive = (args.young_modulus, args.density, args.ground_stiffness)
        if (
            not np.isfinite((*positive, args.gravity, args.drop_height)).all()
            or min(positive) <= 0
            or min(args.gravity, args.drop_height) < 0
            or args.substeps < 1
            or args.resolution < 2
        ):
            raise ValueError(
                "positive material/stiffness/substeps, resolution >= 2 and nonnegative gravity/drop required"
            )
        self.viewer, self.args = viewer, args
        self.sim_time, self.frame_dt = 0.0, 1.0 / 60.0
        self.sim_dt = self.frame_dt / args.substeps
        print(f"Preparing Baymax geometry (resolution {args.resolution})...", flush=True)
        mesh = load_mesh(Path(__file__).with_name("assets") / "tet_baymax.usda")[0]
        # Place Baymax upright; the USD declares Z-up.
        mesh = replace(mesh, vertices=mesh.vertices[:, (0, 2, 1)] * (1.0, -1.0, 1.0))
        cage = self.cage = tetrahedral_cage(mesh, args.resolution)
        shift = np.r_[
            -0.5 * (cage.surface_positions[:, :2].min(0) + cage.surface_positions[:, :2].max(0)),
            args.drop_height - cage.surface_positions[:, 2].min(),
        ]
        positions = self.reference_positions = cage.positions + shift
        print(f"Building Baymax: {len(positions)} nodes / {len(cage.elements)} tetrahedra", flush=True)
        self.fem = SolverANCFTetrahedron.create_model(
            positions,
            cage.elements,
            young_modulus=args.young_modulus * cage.volume_fraction,
            poisson_ratio=0.3,
            density=args.density * cage.volume_fraction,
            device=wp.get_device(),
        )
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -args.gravity))
        builder.add_ground_plane()
        self.fem.register(builder)
        self.model = builder.finalize(device=self.fem.device)
        self.display_state = self.model.state()
        self.solver = SolverANCFTetrahedron(
            self.model,
            self.fem,
            ground_height=0.0,
            ground_stiffness=args.ground_stiffness,
            mass_damping=5.0,
            tolerance=1.0e-7,
            max_iterations=35,
        )
        self.state = self.solver.initial_state()
        parents = cage.elements[cage.surface_elements]
        shape, _ = SolverANCFTetrahedron.shape_functions(cage.barycentric, positions[parents], derivatives=False)
        self.shape = wp.array(shape, dtype=wp.float64, device=self.fem.device)
        self.vectors = wp.array(
            (4 * parents[:, :, None] + np.arange(4)).reshape(-1, 16), dtype=int, device=self.fem.device
        )
        self.points = wp.empty(len(shape), dtype=wp.vec3, device=self.fem.device)
        self.triangles = wp.array(cage.surface_triangles.ravel(), dtype=int, device=self.fem.device)
        edges = sorted({tuple(sorted((tet[a], tet[b]))) for tet in cage.elements for a in range(4) for b in range(a)})
        self.edges = wp.array(edges, dtype=wp.vec2i, device=self.fem.device)
        self.starts = wp.empty(len(edges), dtype=wp.vec3, device=self.fem.device)
        self.ends = wp.empty_like(self.starts)
        self.colors = wp.full(len(edges), wp.vec3(0.2, 0.2, 0.2), dtype=wp.vec3, device=self.fem.device)
        viewer.set_model(self.model)
        viewer.set_camera(pos=(1.0, -1.7, 0.95), pitch=-18.0, yaw=120.0)

    def step(self):
        for _ in range(self.args.substeps):
            self.state = self.solver.advance(self.state, self.sim_dt)
        self.sim_time += self.frame_dt

    def render(self):
        wp.launch(
            _interpolate,
            len(self.points),
            inputs=[self.state.q, self.vectors, self.shape, self.points],
            device=self.fem.device,
            record_tape=False,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.display_state)
        self.viewer.log_mesh("Baymax", self.points, self.triangles, color=(0.72, 0.74, 0.78))
        if self.args.show_cage:
            wp.launch(
                _cage_edges,
                len(self.edges),
                inputs=[self.state.q, self.edges, self.starts, self.ends],
                device=self.fem.device,
                record_tape=False,
            )
        self.viewer.log_lines("Baymax_cage", self.starts, self.ends, self.colors, hidden=not self.args.show_cage)
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.text("ANCF Baymax — elastic solid")
        imgui.text(f"{self.fem.element_count} tetrahedra / {self.fem.node_count} nodes")
        imgui.text(f"Reference mass: {self.args.density * self.cage.source_volume:.3f} kg")
        imgui.text_wrapped(
            "Fixed reference geometry and mass. Frictionless contact at cage nodes; no gas pressure or coupling."
        )
        if imgui.button("Drop again"):
            self.state = self.solver.initial_state()
            self.sim_time = 0.0
        _, self.args.show_cage = imgui.checkbox("Show mechanics cage", self.args.show_cage)

    def test_final(self):
        q, qd = self.state.q.numpy(), self.state.qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise AssertionError("Baymax has non-finite state")
        samples = np.vstack((np.eye(4), np.full((1, 4), 0.25)))
        _, derivatives = SolverANCFTetrahedron.shape_functions(
            samples[None], self.reference_positions[self.cage.elements][:, None]
        )
        derivatives = np.concatenate((self.fem.element_data.derivatives.numpy(), derivatives), axis=1)
        vectors = q.reshape(-1, 4, 3)[self.cage.elements].reshape(-1, 16, 3)
        determinants = np.linalg.det(np.einsum("eai,epaj->epij", vectors, derivatives))
        if not np.isfinite(determinants).all() or np.min(determinants) <= 0:
            raise AssertionError("Baymax has an inverted sampled deformation")
        lowest = float(q.reshape(-1, 12)[:, 2].min())
        if lowest < -0.02:
            raise AssertionError(f"Baymax floor penetration exceeds 2 cm ({lowest:.5f} m)")
        print(f"Baymax: min sampled det(F)={determinants.min():.6f}, lowest cage node={lowest:.6f} m")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--resolution",
            type=int,
            default=6,
            help="Mechanics resolution along the longest axis; 6 gives 204 tetrahedra.",
        )
        parser.add_argument("--substeps", type=int, default=1)
        parser.add_argument("--young-modulus", "--young_modulus", type=float, default=5.0e4)
        parser.add_argument("--density", type=float, default=200.0)
        parser.add_argument("--gravity", type=float, default=9.81)
        parser.add_argument("--drop-height", "--drop_height", type=float, default=0.15)
        parser.add_argument("--ground-stiffness", "--ground_stiffness", type=float, default=2.0e4)
        parser.add_argument(
            "--show-cage",
            "--show_cage",
            action=BooleanOptionalAction,
            default=True,
            help="Show the mechanical cage over the detailed surface (default: on).",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
