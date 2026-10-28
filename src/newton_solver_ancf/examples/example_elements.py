# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Four clamped ANCF cantilevers bend under gravity: cable, beam, shell and cubic tetrahedra.

The independent specimens share a viewer, with a cable at the front followed
by a rectangular beam, shell strip and tetrahedral solid. Each retains its
own geometry and material, using four HHT substeps per frame.
Display offsets align the clamps;
they do not change the simulated coordinates or create contact between objects.
Orange markers identify fixed nodes. Gray shading identifies elements; the UI
toggles outlines and nodal gradient arrows. Each flexibility slider scales that
specimen's compliance and restarts it without changing its geometry or mass.

Run: python -m newton_solver_ancf.examples elements --formulation 3333 --debug-arrows
"""

import argparse
import math

import newton
import newton.examples

from newton_solver_ancf.examples._beam import Beam
from newton_solver_ancf.examples._cable import Cable
from newton_solver_ancf.examples._shell import Shell
from newton_solver_ancf.examples._tetrahedron import Tetrahedron


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.show_arrows = args.debug_arrows
        self.show_edges = False
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.display_state = self.model.state()
        viewer.activate("Scene")
        viewer.set_model(self.model)

        settings = {"debug_arrows": args.debug_arrows, "substeps": args.substeps}
        self.elements = {}
        viewer.activate("Cable")
        self.elements["Cable"] = Cable(viewer, render_offset=(0.0, -0.36, 0.0), **settings)
        beam_name = f"Beam {args.formulation}"
        viewer.activate(beam_name)
        self.elements[beam_name] = Beam(
            viewer, formulation=args.formulation, render_offset=(0.0, -0.12, 0.35), **settings
        )
        viewer.activate("Shell 3423")
        self.elements["Shell 3423"] = Shell(viewer, render_offset=(0.0, 0.12, 0.3), **settings)
        viewer.activate("Cubic tetrahedra")
        self.elements["Cubic tetrahedra"] = Tetrahedron(viewer, render_offset=(0.0, 0.36, 0.32), **settings)
        self.flexibility = dict.fromkeys(self.elements, 1.0)
        self._reference_parameters = {
            name: {key: value.numpy().copy() for key, value in element.solver.fem.parameters.items()}
            for name, element in self.elements.items()
        }
        self._reference_moduli = {"Cable": 20.0, beam_name: 20.0, "Shell 3423": 20.0, "Cubic tetrahedra": 2.0}
        viewer.activate("Scene")
        viewer.set_camera(pos=(1.1, -1.5, 1.3), pitch=-28.0, yaw=120.0)

    def set_flexibility(self, name, value):
        """Scale compliance at fixed geometry, mass and Poisson ratio; restart one specimen."""
        if not math.isfinite(value) or not 0.2 <= value <= 5.0:
            raise ValueError("flexibility must be between 0.2 and 5.0")
        element = self.elements[name]
        for key, parameter in element.solver.fem.parameters.items():
            parameter.assign(self._reference_parameters[name][key] / value)
        self.flexibility[name] = value
        # HHT stores previous elastic forces. Restart all state histories together
        # when the material changes, rather than mixing old and new stiffnesses.
        element.state = element.solver.initial_state()
        element.sim_time = 0.0

    def step(self):
        for element in self.elements.values():
            element.step()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.activate("Scene")
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.display_state)
        for name, element in self.elements.items():
            self.viewer.activate(name)
            element.node_arrows.enabled = self.show_arrows
            element.render(show_edges=self.show_edges)
        self.viewer.activate("Scene")
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.text("Flexibility: higher is softer")
        imgui.text_wrapped("1 = original material. Changing a slider restarts that specimen.")
        for name in self.elements:
            imgui.text(name)
            changed, value = imgui.slider_float(f"##flexibility_{name}", self.flexibility[name], 0.2, 5.0)
            if changed:
                self.set_flexibility(name, min(5.0, max(0.2, value)))
            modulus = self._reference_moduli[name] / self.flexibility[name]
            imgui.text(f"Young's modulus: {modulus:.2f} MPa")
        if imgui.button("Restore original materials"):
            for name in self.elements:
                self.set_flexibility(name, 1.0)
        imgui.separator()
        _, self.show_arrows = imgui.checkbox("Nodal arrows", self.show_arrows)
        _, self.show_edges = imgui.checkbox("Element outlines", self.show_edges)
        imgui.text_wrapped("Arrows show nodal gradients: red X, green Y, blue Z.")
        imgui.text("Front to back: cable, beam, shell, tetrahedra.")
        imgui.text_wrapped("Orange nodes are clamped. Use Layers to show or hide each specimen.")
        imgui.text_wrapped("Independent gravity-loaded specimens; the ground is a visual reference.")

    def test_final(self):
        for element in self.elements.values():
            element.test_final()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--formulation", choices=("3243", "3333"), default="3243", help="Beam formulation.")
        parser.add_argument(
            "--substeps",
            type=int,
            default=4,
            help="Substeps per 60 Hz frame for every specimen (default: 4).",
        )
        parser.add_argument(
            "--debug-arrows",
            "--debug_arrows",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Show nodal gradients (enabled by default; also controlled in the UI).",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
